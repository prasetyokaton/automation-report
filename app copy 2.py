# app.py
# Dependencies: streamlit, pandas, plotly, openpyxl
# Run: streamlit run app.py

import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


st.set_page_config(page_title="SOV & Time Series Combo", layout="wide")
st.title("📈 Automation Report Insights")

# ============== Helpers ==============

@st.cache_data(show_spinner=False)
def read_excel_all_sheets(file):
    return pd.read_excel(file, sheet_name=None)

@st.cache_data(show_spinner=False)
def read_excel_sheet(file, sheet_name):
    return pd.read_excel(file, sheet_name=sheet_name)

def find_campaign_column(columns):
    norm = {c.lower().strip(): c for c in columns}
    for key in ["campaigns", "campaign"]:
        if key in norm: return norm[key]
    return None

def find_date_column(df):
    # Prioritaskan nama umum + tipe datetime
    candidates_by_name = ["date", "created", "created at", "created_at",
                          "post date", "post_date", "published", "timestamp",
                          "time", "tanggal"]
    by_name = []
    norm = {c.lower().strip(): c for c in df.columns}
    for nm in candidates_by_name:
        if nm in norm: by_name.append(norm[nm])
    # Tipe datetime
    by_dtype = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    # Gabungkan unik, pertahankan prioritas by_name
    seen, ordered = set(), []
    for c in by_name + by_dtype:
        if c not in seen:
            ordered.append(c); seen.add(c)
    return ordered[0] if ordered else None

def coerce_numeric(series: pd.Series) -> pd.Series:
    # Bersihkan angka bertitik/koma dan simbol
    s = pd.to_numeric(
        series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
        errors="coerce"
    )
    return s

def prep_sov(df, campaign_col):
    ser = (
        df[campaign_col]
        .astype(str).str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .dropna()
    )
    sov = (ser.to_frame(name=campaign_col)
             .groupby(campaign_col, dropna=True)
             .size()
             .reset_index(name="Post")
             .sort_values("Post", ascending=False)
             .reset_index(drop=True))
    total = int(sov["Post"].sum()) if len(sov) else 0
    sov["Share (%)"] = (sov["Post"] / total * 100).round(2) if total else 0.0
    return sov, total

def topn_with_others(sov, campaign_col, top_n: int):
    if len(sov) <= top_n: return sov.copy()
    top = sov.head(top_n).copy()
    others_post = int(sov["Post"].iloc[top_n:].sum())
    if others_post > 0:
        others_share = round(others_post / sov["Post"].sum() * 100, 2)
        top.loc[len(top)] = ["Others", others_post, others_share]
    return top

def pick_col(df, aliases):
    """Return actual column name (case-insensitive) if exists; else None."""
    lowmap = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lowmap:
            return lowmap[a.lower()]
    return None

def topn_generic(df, label_col, value_col, top_n: int):
    """Return top-N by value_col plus 'Others' row."""
    if df.empty or value_col not in df.columns:
        return df.copy()
    data = df[[label_col, value_col]].sort_values(value_col, ascending=False)
    if len(data) <= top_n:
        return data.reset_index(drop=True)
    top = data.head(top_n).copy()
    others_val = data[value_col].iloc[top_n:].sum()
    if others_val > 0:
        top.loc[len(top)] = ["Others", others_val]
    return top.reset_index(drop=True)

def build_timeseries(df, date_col, left_metric_key, right_metric_key, date_min=None, date_max=None):
    # Siapkan kolom tanggal harian
    dates = pd.to_datetime(df[date_col], errors="coerce")
    base = df.copy()
    base["__date"] = dates.dt.date
    base = base.dropna(subset=["__date"])

    # Filter tanggal
    if date_min: base = base[base["__date"] >= date_min]
    if date_max: base = base[base["__date"] <= date_max]

    # Definisi agregasi
    def agg_metric(frame, key):
        if key == "__count_post__":
            return frame.groupby("__date").size().rename("value")
        else:
            vals = coerce_numeric(frame[key])
            tmp = frame.assign(_val=vals)
            return tmp.groupby("__date")["_val"].sum(min_count=1).rename("value")

    left_series = agg_metric(base, left_metric_key)
    right_series = agg_metric(base, right_metric_key)

    ts = pd.DataFrame({
        "Date": left_series.index
    })
    ts["Left"] = left_series.values
    # Sinkronisasi tanggal (gabung outer)
    ts = pd.merge(ts, right_series.rename_axis("Date").reset_index(), on="Date", how="outer")
    ts.rename(columns={"value": "Right"}, inplace=True)
    ts = ts.sort_values("Date").reset_index(drop=True)
    return ts

def _calc_line_label_shift(bar_val, idx, q2, q3):
    """
    Bikin offset label line agar tidak menimpa bar:
    - bar tinggi (>= q3) -> dorong label lebih jauh ke atas
    - bar sedang (>= q2) -> dorong sedang
    - bar kecil (> 0)    -> dorong sedikit
    - selang-seling xshift untuk kurangi bentrok horizontal
    """
    if pd.isna(bar_val) or bar_val <= 0:
        base_y = 10
    elif bar_val >= q3:
        base_y = 28
    elif bar_val >= q2:
        base_y = 20
    else:
        base_y = 14
    xshift = 8 if (idx % 2 == 0) else -8
    return base_y, xshift



def normalize_channel(val: str):
    if pd.isna(val): return None
    s = str(val).strip().lower()
    # social
    if s in {"youtube", "you tube", "yt"}: return "YouTube"
    if s in {"tiktok", "tik tok", "tik-tok"}: return "Tiktok"
    if s in {"instagram", "ig"}: return "Instagram"
    if s in {"twitter", "x", "tw"}: return "Twitter"
    if s in {"facebook", "fb"}: return "Facebook"
    if s in {"forum", "kaskus", "reddit"}: return "Forum"
    # mainstream
    if s in {"online media", "online", "portal", "news", "website"}: return "Online Media"
    if s in {"printed", "print", "koran", "majalah", "newspaper", "tabloid"}: return "Printed"
    if s in {"tv", "television", "televisi"}: return "TV"
    return str(val).strip().title()

def fmt_id_int(x):
    try: return f"{int(round(float(x))) :,}".replace(",", ".")
    except: return str(x)

def _agg_sum_by_channel(base: pd.DataFrame, colname: str) -> pd.Series:
    if colname not in base.columns:
        return pd.Series(dtype=float, name=colname)
    vals = coerce_numeric(base[colname])
    return base.assign(_val=vals).groupby("__chan")["_val"].sum(min_count=1).rename(colname)

def build_channel_combo_v2(
    df_in: pd.DataFrame,
    channel_list: list,
    channel_col: str,
    left_key: str, left_title: str,
    right_key: str, right_title: str,
    title: str
):
    # fallback channel list
    if not channel_list:
        channel_list = ["YouTube","Tiktok","Instagram","Twitter","Facebook","Forum"] \
            if title.lower().startswith("social") else ["Online Media","Printed","TV"]

    base = df_in.copy()
    base["__chan"] = base[channel_col].map(normalize_channel)

    # count (post/articles)
    post = base.groupby("__chan").size().rename("__post__")

    # left metric
    if left_key in {"__count_post__","__count_articles__"}:
        left_ser = post.rename(left_title)
    else:
        left_ser = _agg_sum_by_channel(base, left_key).rename(left_title)

    # right metric
    right_ser = _agg_sum_by_channel(base, right_key).rename(right_title)

    # align pada urutan channel_list
    idx = pd.Index(channel_list, name="__chan")
    agg = (
        pd.concat([post, left_ser, right_ser], axis=1)
        .reindex(idx)
        .fillna(0.0)
        .reset_index()
    )
    agg.rename(columns={"__post__": "Post"}, inplace=True)

    # figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    BAR_COLOR, LINE_COLOR = "#f2b01e", "#1f77b4"

    # BAR (Left) — label di DALAM, bawah
    fig.add_bar(
        x=agg["__chan"], y=agg[left_title], name=left_title,
        marker_color=BAR_COLOR, opacity=0.95,
        text=[fmt_id_int(v) for v in agg[left_title]],
        texttemplate="%{text}",
        textposition="inside",
        insidetextanchor="start",   # tempel di base (bawah)
        textfont=dict(size=10),
        hovertemplate=f"<b>%{{x}}</b><br>{left_title}: %{{y:,}}<extra></extra>",
        secondary_y=False
    )

    # LINE (Right) — label floating (annotation ala Chart 2)
    fig.add_scatter(
        x=agg["__chan"], y=agg[right_title], name=right_title,
        mode="lines+markers", line=dict(width=3, color=LINE_COLOR), marker=dict(size=7),
        hovertemplate=f"<b>%{{x}}</b><br>{right_title}: %{{y:,}}<extra></extra>",
        secondary_y=True
    )
    for i, (xi, yv) in enumerate(zip(agg["__chan"], agg[right_title])):
        fig.add_annotation(
            x=xi, y=yv, yref="y2",
            text=fmt_id_int(yv),
            showarrow=False,
            bgcolor="rgba(220,220,220,0.85)",
            font=dict(size=10),
            borderpad=3,
            yshift=6,
            xshift=(7 if i % 2 == 0 else -7)
        )

    # layout & axes
    left_max  = float(agg[left_title].max()) if len(agg) else 0.0
    right_max = float(agg[right_title].max()) if len(agg) else 0.0
    fig.update_layout(
        title=title, height=420, bargap=0.35, hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text=left_title, secondary_y=False, showgrid=True,
                     gridcolor="rgba(0,0,0,0.08)",
                     range=[0, left_max * 1.25 if left_max > 0 else 1])
    fig.update_yaxes(title_text=right_title, secondary_y=True, showgrid=False,
                     range=[0, right_max * 1.25 if right_max > 0 else 1])
    return fig, agg





# ============== Sidebar: Upload & Global Controls ==============

st.sidebar.header("1) Upload File")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False)

if not uploaded:
    st.info("Upload file Excel kamu dulu. Pastikan ada kolom **Campaigns/Campaign** dan **Date**.")
    st.stop()

sheets = read_excel_all_sheets(uploaded)
sheet_names = list(sheets.keys())

st.sidebar.header("2) Global Options (Berlaku untuk Semua Chart)")
sheet_name = st.sidebar.selectbox("Pilih Sheet", sheet_names, index=0)

df = read_excel_sheet(uploaded, sheet_name)

# Pilih kolom Campaign & Date (global)
detected_campaign = find_campaign_column(df.columns) or df.columns[0]
campaign_col = st.sidebar.selectbox("Kolom Campaign", list(df.columns), index=list(df.columns).index(detected_campaign))

detected_date = find_date_column(df) or df.columns[0]
date_col = st.sidebar.selectbox("Kolom Date/Tanggal", list(df.columns), index=list(df.columns).index(detected_date))

# Opsi Top N untuk Pie
top_n = st.sidebar.slider("Top-N untuk Pie (sisanya → Others)", 3, 15, 8)

# === Global: pilih kolom Channel & Engagement (untuk Chart 4 & 5) ===
st.sidebar.header("3) Kolom Channel & Engagement")

chan_candidates = [c for c in df.columns if c.lower() in
                   ["channel","channels","platform","media","source","tipe","type"]]
channel_col = st.sidebar.selectbox(
    "Kolom Channel/Platform",
    options=list(df.columns),
    index=(list(df.columns).index(chan_candidates[0]) if chan_candidates else 0),
    key="global_channel_col"
)

num_cols_global = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
pref_names = {"engagement","total engagement","interaction","interactions","pr value","pr_value","prvalue"}
for c in df.columns:
    if c not in num_cols_global and c.lower() in pref_names:
        num_cols_global.append(c)

eng_default_idx = 0
if "Engagement" in num_cols_global: eng_default_idx = num_cols_global.index("Engagement")
elif "engagement" in num_cols_global: eng_default_idx = num_cols_global.index("engagement")

eng_col = st.sidebar.selectbox(
    "Kolom Engagement (default Chart 4 & 5)",
    options=(num_cols_global if num_cols_global else list(df.columns)),
    index=eng_default_idx if num_cols_global else 0,
    key="global_eng_col"
)


# ============== Chart 1: SOV + Metrics + 3 Pie ==============

# daftar nilai campaign (dipakai untuk checklist)
campaign_values_all = sorted(df[campaign_col].astype(str).dropna().unique().tolist())

# Filter (di bawah tabel) + tombol Reset & Pilih Semua (callback HARUS sebelum widget)
st.markdown("**Tampilkan Campaign**")
row_flt, row_btns = st.columns([5, 2])

def _reset_campaigns():
    # buang key supaya multiselect pakai default full pada rerun berikutnya
    if "campaign_filter_chart1" in st.session_state:
        del st.session_state["campaign_filter_chart1"]

def _select_all_campaigns():
    st.session_state["campaign_filter_chart1"] = campaign_values_all

with row_btns:
    b1, b2 = st.columns(2)
    with b1:
        st.button("Reset", help="Kembalikan ke default", on_click=_reset_campaigns)
    with b2:
        st.button("Pilih Semua", help="Select all campaigns", on_click=_select_all_campaigns)

with row_flt:
    selected_campaigns_chart1 = st.multiselect(
        "Pilih yang ingin ditampilkan (default: semua)",
        options=campaign_values_all,
        default=campaign_values_all,
        key="campaign_filter_chart1"
    )

if not selected_campaigns_chart1:
    selected_campaigns_chart1 = campaign_values_all

# df_work = hasil filter Chart 1 → dipakai semua chart di bawah
df_work = df[df[campaign_col].astype(str).isin(selected_campaigns_chart1)].copy()
if df_work.empty:
    df_work = df.copy()

# Hitung ulang SOV berdasar df_work
sov_work, total_post_work = prep_sov(df_work, campaign_col)

# --- Metrics (Total Post Terbaca, Total Engagement, Total Buzz) ---
eng_name  = pick_col(df, ["Engagement", "Total Engagement", "engagement", "interactions", "interaction"])
buzz_name = pick_col(df, ["Buzz", "buzz"])

tot_eng  = int(coerce_numeric(df_work[eng_name]).sum())  if eng_name  and eng_name in df_work.columns else 0
tot_buzz = int(coerce_numeric(df_work[buzz_name]).sum()) if buzz_name and buzz_name in df_work.columns else 0

m1, m2, m3 = st.columns(3)
with m1: st.metric("Total Post Terbaca", value=total_post_work)
with m2: st.metric("Total Engagement", value=f"{tot_eng:,}")
with m3: st.metric("Total Buzz", value=f"{tot_buzz:,}")

st.caption("Tabel Data")

# --- Tabel agregat lengkap (Total Post, Engagement, Buzz + SOV%) ---
grp_post = df_work.groupby(campaign_col).size().rename("Total Post")
tbl = grp_post.to_frame()

if eng_name and eng_name in df_work.columns:
    grp_eng = (df_work.assign(_eng=coerce_numeric(df_work[eng_name]))
               .groupby(campaign_col)["_eng"].sum(min_count=1).rename("Total Engagement"))
    tbl = tbl.join(grp_eng, how="left")
else:
    tbl["Total Engagement"] = 0

if buzz_name and buzz_name in df_work.columns:
    grp_buzz = (df_work.assign(_buzz=coerce_numeric(df_work[buzz_name]))
                .groupby(campaign_col)["_buzz"].sum(min_count=1).rename("Total Buzz"))
    tbl = tbl.join(grp_buzz, how="left")
else:
    tbl["Total Buzz"] = 0

tot_post_sum = int(tbl["Total Post"].sum()) if len(tbl) else 0
tot_eng_sum  = float(tbl["Total Engagement"].sum()) if len(tbl) else 0.0
tot_buzz_sum = float(tbl["Total Buzz"].sum()) if len(tbl) else 0.0

tbl["SOV Post%"]      = (tbl["Total Post"]       / tot_post_sum * 100).round(2) if tot_post_sum else 0.0
tbl["SOV Engagement%"] = (tbl["Total Engagement"] / tot_eng_sum  * 100).round(2) if tot_eng_sum  else 0.0
tbl["SOV Buzz%"]       = (tbl["Total Buzz"]       / tot_buzz_sum * 100).round(2) if tot_buzz_sum else 0.0

tbl = (tbl.reset_index()
          .rename(columns={campaign_col: "Campaign"})
          .sort_values(["Total Post", "Total Engagement", "Total Buzz"], ascending=False))

tbl = tbl[["Campaign", "Total Post", "SOV Post%", "Total Engagement", "SOV Engagement%", "Total Buzz", "SOV Buzz%"]]

st.dataframe(
    tbl.style.format({"SOV Post%": "{:.2f}", "SOV Engagement%": "{:.2f}", "SOV Buzz%": "{:.2f}"}),
    use_container_width=True, height=320
)

# --- 3 Pie berdampingan: Post / Engagement / Buzz ---
st.subheader("Share of Voice — Comparison")

def _pie(df_in, title):
    if df_in.empty:
        return go.Figure()
    fig = px.pie(df_in, names="Campaign", values="Value", hole=0.0, title=title)
    fig.update_traces(
        texttemplate="%{label}<br>%{percent}",
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        showlegend=False
    )
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
    return fig

top_post = topn_generic(tbl.rename(columns={"Total Post":"Value"}),            "Campaign", "Value", top_n)
top_eng   = topn_generic(tbl.rename(columns={"Total Engagement":"Value"}),      "Campaign", "Value", top_n)
top_buzz  = topn_generic(tbl.rename(columns={"Total Buzz":"Value"}),            "Campaign", "Value", top_n)

c_p1, c_p2, c_p3 = st.columns(3)
with c_p1: st.plotly_chart(_pie(top_post, "SOV by Total Post"),        use_container_width=True)
with c_p2: st.plotly_chart(_pie(top_eng,   "SOV by Total Engagement"),  use_container_width=True)
with c_p3: st.plotly_chart(_pie(top_buzz,  "SOV by Total Buzz"),        use_container_width=True)



# ============== Sidebar filters for Chart 4 & 5 ==============
st.sidebar.markdown("---")
st.sidebar.subheader("Chart 4 — Social Media Channels")
SOCIAL_ORDER = ["YouTube", "Tiktok", "Instagram", "Twitter", "Facebook", "Forum"]
selected_social_channels = st.sidebar.multiselect(
    "Pilih channel Social",
    options=SOCIAL_ORDER,
    default=SOCIAL_ORDER,
    key="social_channels_sel"
)

st.sidebar.subheader("Chart 5 — Mainstream Channels")
MAINSTREAM_ORDER = ["Online Media", "Printed", "TV"]
selected_mainstream_channels = st.sidebar.multiselect(
    "Pilih channel Mainstream",
    options=MAINSTREAM_ORDER,
    default=MAINSTREAM_ORDER,
    key="mainstream_channels_sel"
)


st.markdown("---")

# ============== Conversation Trends (Time Series) ==============
st.markdown("---")
st.subheader("Conversation Trends")

# Kandidat metrik
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
preferred_metric_names = ["engagement","likes","comments","shares","views","impressions","reach","er"]
for c in df.columns:
    if c not in numeric_cols and c.lower() in preferred_metric_names:
        numeric_cols.append(c)

LEFT_COUNT_ALIAS = "Total Post"
left_choices = [LEFT_COUNT_ALIAS] + numeric_cols
right_choices = numeric_cols.copy()

# Default right = Engagement jika ada
if "Engagement" in right_choices: default_right_idx = right_choices.index("Engagement")
elif "engagement" in right_choices: default_right_idx = right_choices.index("engagement")
else: default_right_idx = 0 if right_choices else None

c1, c2, c3 = st.columns([1.2, 1.2, 2])
with c1:
    left_metric_choice = st.selectbox("Left Axis (Bar)", left_choices, index=0, key="ts_left")
with c2:
    right_metric_choice = st.selectbox("Right Axis (Line)", right_choices, index=default_right_idx, key="ts_right")

# Date range
dates_parsed = pd.to_datetime(df_work[date_col], errors="coerce")
date_min_default, date_max_default = dates_parsed.min(), dates_parsed.max()
with c3:
    chosen_range = st.date_input(
        "Filter Date Range",
        value=(date_min_default.date(), date_max_default.date()),
        min_value=date_min_default.date(),
        max_value=date_max_default.date()
    )
    dmin, dmax = (chosen_range[0], chosen_range[1]) if isinstance(chosen_range, tuple) else (date_min_default.date(), date_max_default.date())

left_key  = "__count_post__" if left_metric_choice == LEFT_COUNT_ALIAS else left_metric_choice
right_key = right_metric_choice

# Normalisasi channel di df_work untuk filter grup
df_work_norm = df_work.copy()
df_work_norm["__chan_norm"] = df_work_norm[channel_col].map(normalize_channel)

SOCIAL_ORDER     = ["YouTube","Tiktok","Instagram","Twitter","Facebook","Forum"]
MAINSTREAM_ORDER = ["Online Media","Printed","TV"]

# Helper plot (reuse dari Chart 2 lama)
def _combo_ts(df_src, title):
    ts = build_timeseries(df_src, date_col, left_key, right_key, dmin, dmax)
    ts_vis = ts.copy().fillna(0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    BAR_COLOR, LINE_COLOR = "#f2b01e", "#1f77b4"
    fig.add_bar(x=ts_vis["Date"], y=ts_vis["Left"], name=left_metric_choice,
                marker_color=BAR_COLOR, opacity=0.95,
                text=ts_vis["Left"], texttemplate="%{text:,.0f}",
                textposition="inside", insidetextanchor="start",
                textfont=dict(size=10), cliponaxis=False, secondary_y=False)
    fig.add_scatter(x=ts_vis["Date"], y=ts_vis["Right"], name=right_metric_choice,
                    mode="lines+markers", line=dict(width=3, color=LINE_COLOR),
                    marker=dict(size=6), cliponaxis=False, secondary_y=True)
    # label float kanan
    if len(ts_vis):
        q2 = float(np.nanpercentile(ts_vis["Left"], 50))
        q3 = float(np.nanpercentile(ts_vis["Left"], 75))
    else:
        q2 = q3 = 0.0
    for i, (xi, y_r, y_l) in enumerate(zip(ts_vis["Date"], ts_vis["Right"], ts_vis["Left"])):
        yshift, xshift = _calc_line_label_shift(y_l, i, q2, q3)
        fig.add_annotation(x=xi, y=y_r, yref="y2", text=f"{int(y_r):,}",
                           showarrow=False, bgcolor="rgba(220,220,220,0.85)",
                           font=dict(size=10), borderpad=3, yshift=yshift, xshift=xshift)

    left_max  = float(np.nanmax(ts_vis["Left"]))  if len(ts_vis) else 0.0
    right_max = float(np.nanmax(ts_vis["Right"])) if len(ts_vis) else 0.0

    fig.update_layout(title=title, height=420, bargap=0.35, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_xaxes(title_text="Date", tickmode="linear", dtick="D1", tickformat="%d-%b",
                     tickfont=dict(size=10 if len(ts_vis) > 31 else 12),
                     showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text=left_metric_choice, range=[0, left_max*1.25 if left_max>0 else 1],
                     showgrid=True, gridcolor="rgba(0,0,0,0.08)", secondary_y=False)
    fig.update_yaxes(title_text=right_metric_choice, range=[0, right_max*1.25 if right_max>0 else 1],
                     showgrid=False, secondary_y=True)
    return fig

# SOCIAL
df_social = df_work_norm[df_work_norm["__chan_norm"].isin(SOCIAL_ORDER)]
st.plotly_chart(_combo_ts(df_social, "Social Conversation Trends"), use_container_width=True)

# MAINSTREAM
df_main = df_work_norm[df_work_norm["__chan_norm"].isin(MAINSTREAM_ORDER)]
st.plotly_chart(_combo_ts(df_main, "Mainstream Conversations Trends"), use_container_width=True)




# ============== Chart 3: Sentiment Breakdown (Stacked Horizontal) ==============
st.markdown("---")
st.subheader("Sentiment Breakdown")


# 1) Pilih kolom Sentiment + (opsional) filter Campaign
sentiment_candidates = [c for c in df.columns if c.lower() in [
    "sentiment", "new sentiment", "new_sentiment", "sentiment fix",
    "new sentiment level", "new_sentiment_level", "sentiment_level",
    "sentiment level"
]]
# Ambil otomatis kolom Sentiment; UI disembunyikan (sesuai instruksi)
sent_col = sentiment_candidates[0] if sentiment_candidates else df.columns[0]

work = df_work.copy()  # tergantung filter campaign atas; tidak ada filter tambahan
# (UI "Kolom Sentiment" & "Filter Campaign (opsional)" dikomentari sengaja)


# 2) Normalisasi nilai sentiment -> Positive / Neutral / Negative
def normalize_sentiment(v):
    s = str(v).strip().lower()
    if s in {"positive","positif","pos","+1","1","good","bagus"}:       return "Positive"
    if s in {"negative","negatif","neg","-1","bad","buruk"}:           return "Negative"
    if s in {"neutral","netral","neu","0"}:                            return "Neutral"
    try:
        f = float(s)
        if f > 0: return "Positive"
        if f < 0: return "Negative"
        return "Neutral"
    except:
        return "Neutral"

order  = ["Positive", "Neutral", "Negative"]
colors = {"Positive": "#43A047", "Neutral": "#BDBDBD", "Negative": "#E53935"}  # hijau/abu/merah
labels_bg = {"Positive": "rgba(67,160,71,0.85)", "Neutral": "rgba(189,189,189,0.85)", "Negative": "rgba(229,57,53,0.85)"}

work["__sent"] = work[sent_col].map(normalize_sentiment)
counts = work["__sent"].value_counts().reindex(order, fill_value=0)
total  = int(counts.sum())

if total == 0:
    st.info("Tidak ada data sentiment pada selection saat ini.")
else:
    perc = (counts / total * 100).round(2)

    # fungsi format persen dengan koma (ID)
    def fmt_pct(x):
        s = f"{x:.2f}".replace(".", ",")
        return f"{s}%"

    fig_s = go.Figure()
    # Tambahkan 3 segmen (stacked)
    for name in order:
        val = float(perc.get(name, 0.0))
        fig_s.add_bar(
            y=[" "], x=[val], name=name, orientation="h",
            marker_color=colors[name],
            hovertemplate=f"<b>{name}</b><br>{fmt_pct(val)}<extra></extra>",
            text=None
        )

    # Anotasi label di tengah tiap segmen
    cum = 0.0
    for name in order:
        val = float(perc.get(name, 0.0))
        if val <= 0:
            continue
        x_mid = cum + val/2.0
        fig_s.add_annotation(
            x=x_mid, y=0, xref="x", yref="y",
            text=f"{name}<br>{fmt_pct(val)}",
            showarrow=False, font=dict(size=16, color="white"),
            bgcolor=labels_bg[name], borderpad=4
        )
        cum += val

    fig_s.update_layout(
        barmode="stack",
        height=160,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
    )
    fig_s.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=False, title=None)
    fig_s.update_yaxes(visible=False)

    st.plotly_chart(fig_s, use_container_width=True)

    # Ringkasannya (opsional)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Post", total)
    with c2: st.metric("Positive", fmt_pct(float(perc.get("Positive", 0))))
    with c3: st.metric("Neutral",  fmt_pct(float(perc.get("Neutral", 0))))
    with c4: st.metric("Negative", fmt_pct(float(perc.get("Negative", 0))))





# Guard agar tidak crash jika kolom tidak ditemukan
if channel_col not in df.columns:
    st.error(f"Kolom channel '{channel_col}' tidak ditemukan di sheet {sheet_name}.")
    st.stop()
if eng_col not in df.columns:
    st.error(f"Kolom engagement '{eng_col}' tidak ditemukan di sheet {sheet_name}.")
    st.stop()


# ============== Chart 4: Social Media Channels (Bar • Line) ==============
st.markdown("---")
st.subheader("Social Media Channel Breakdown")


# opsi metrik seperti Chart 2
numeric_cols_ch = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
pref_names_ch = {"engagement","total engagement","interaction","interactions","views","impressions","reach","pr value","pr_value","prvalue"}
for c in df.columns:
    if c not in numeric_cols_ch and c.lower() in pref_names_ch:
        numeric_cols_ch.append(c)

LEFT_ALIAS_SOC = "Total Post"   # count post, penyebutan untuk social
left_choices4  = [LEFT_ALIAS_SOC] + numeric_cols_ch
right_choices4 = numeric_cols_ch

# default kanan = Engagement
right_default4 = right_choices4.index("Engagement") if "Engagement" in right_choices4 \
                 else (right_choices4.index("engagement") if "engagement" in right_choices4 else 0)

c41, c42 = st.columns(2)
with c41:
    left_choice4  = st.selectbox("Left Axis (Bar)", left_choices4, index=0, key="ch4_left")
with c42:
    right_choice4 = st.selectbox("Right Axis (Line)", right_choices4, index=right_default4, key="ch4_right")

left_key4   = "__count_post__" if left_choice4 == LEFT_ALIAS_SOC else left_choice4
left_title4 = LEFT_ALIAS_SOC if left_choice4 == LEFT_ALIAS_SOC else left_choice4
right_key4  = right_choice4
right_title4= right_choice4

# guard
if channel_col not in df.columns:
    st.error(f"Kolom channel '{channel_col}' tidak ditemukan di sheet {sheet_name}."); st.stop()

fig_social, agg_social = build_channel_combo_v2(
    df_in=df_work,
    channel_list=selected_social_channels,
    channel_col=channel_col,
    left_key=left_key4, left_title=left_title4,
    right_key=right_key4, right_title=right_title4,
    title="Social Media Channels"
)
st.plotly_chart(fig_social, use_container_width=True)


# ============== Chart 5: Mainstream Media Channels (Bar • Line) ==============
st.subheader("Mainstream Media Channel Breakdown")


numeric_cols_ch2 = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
pref_names_ch2 = {"engagement","interaction","interactions","views","impressions","reach","pr value","pr_value","prvalue","ad value"}
for c in df.columns:
    if c not in numeric_cols_ch2 and c.lower() in pref_names_ch2:
        numeric_cols_ch2.append(c)

LEFT_ALIAS_MS = "Volume Artikel"   # count post, penyebutan untuk mainstream
left_choices5  = [LEFT_ALIAS_MS] + numeric_cols_ch2
right_choices5 = numeric_cols_ch2

# default kanan = PR Value
def_idx_pr = 0
for nm in ["PR Value","pr value","pr_value","prvalue"]:
    if nm in right_choices5:
        def_idx_pr = right_choices5.index(nm); break

d51, d52 = st.columns(2)
with d51:
    left_choice5  = st.selectbox("Left Axis (Bar)", left_choices5, index=0, key="ch5_left")
with d52:
    right_choice5 = st.selectbox("Right Axis (Line)", right_choices5, index=def_idx_pr, key="ch5_right")

left_key5   = "__count_articles__" if left_choice5 == LEFT_ALIAS_MS else left_choice5
left_title5 = LEFT_ALIAS_MS if left_choice5 == LEFT_ALIAS_MS else left_choice5
right_key5  = right_choice5
right_title5= right_choice5

if channel_col not in df.columns:
    st.error(f"Kolom channel '{channel_col}' tidak ditemukan di sheet {sheet_name}."); st.stop()

fig_main, agg_main = build_channel_combo_v2(
    df_in=df_work,
    channel_list=selected_mainstream_channels,
    channel_col=channel_col,
    left_key=left_key5, left_title=left_title5,
    right_key=right_key5, right_title=right_title5,
    title="Mainstream Media Channels"
)
st.plotly_chart(fig_main, use_container_width=True)
