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
st.title("📈 Automation Sandbox — SOV & Combo Chart")

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
             .reset_index(name="Posts")
             .sort_values("Posts", ascending=False)
             .reset_index(drop=True))
    total = int(sov["Posts"].sum()) if len(sov) else 0
    sov["Share (%)"] = (sov["Posts"] / total * 100).round(2) if total else 0.0
    return sov, total

def topn_with_others(sov, campaign_col, top_n: int):
    if len(sov) <= top_n: return sov.copy()
    top = sov.head(top_n).copy()
    others_posts = int(sov["Posts"].iloc[top_n:].sum())
    if others_posts > 0:
        others_share = round(others_posts / sov["Posts"].sum() * 100, 2)
        top.loc[len(top)] = ["Others", others_posts, others_share]
    return top

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
        if key == "__count_posts__":
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

# ============== Chart 1: Pie SOV by Campaign ==============

sov, total_posts = prep_sov(df, campaign_col)
sov_display = topn_with_others(sov, campaign_col, top_n)

left_m1, right_m1 = st.columns([1, 2])
with left_m1:
    st.metric("Total Posts Terbaca", value=total_posts)
    st.caption(f"Sheet: **{sheet_name}** — Campaign: **{campaign_col}** — Date: **{date_col}**")

with right_m1:
    st.dataframe(sov.style.format({"Share (%)": "{:.2f}"}), use_container_width=True, height=320)

# === Pie Chart — SOV with leader lines & label two lines ===
st.subheader("Pie Chart — Share of Voice by Campaigns")

# hitung share untuk opsi 'pull' ringan pada slice kecil (bikin ruang label)
total_val = float(sov_display["Posts"].sum()) if len(sov_display) else 1.0
shares = (sov_display["Posts"] / total_val).fillna(0).tolist()
# tarik sedikit slice yg < 3% agar garis label tidak bertumpuk
pull = [0.04 if s < 0.03 else 0 for s in shares]

fig_pie = px.pie(
    sov_display,
    names=campaign_col,
    values="Posts",
    hole=0.0
)

# leader lines: text di luar + garis otomatis dari Plotly
fig_pie.update_traces(
    textposition="outside",                    # label di luar
    textinfo="label+percent",                  # tampilkan label & persen
    texttemplate="%{label}<br>%{percent}",     # dua baris: label + % 
    pull=pull,                                 # longgarkan slice kecil
    hovertemplate="<b>%{label}</b><br>Posts: %{value:,}<br>%{percent}<extra></extra>",
    showlegend=False
)

# beri margin agar label luar tak kepotong
fig_pie.update_layout(
    margin=dict(l=20, r=20, t=10, b=10)
)

st.plotly_chart(fig_pie, use_container_width=True)


# Download SOV
buf_sov = io.BytesIO()
with pd.ExcelWriter(buf_sov, engine="openpyxl") as writer:
    sov.to_excel(writer, sheet_name="SOV_All", index=False)
    sov_display.to_excel(writer, sheet_name=f"SOV_Top{top_n}_Others", index=False)
st.download_button(
    "📥 Download SOV (Excel)",
    data=buf_sov.getvalue(),
    file_name="sov_by_campaigns.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")

# ============== Chart 2: Combo (Bar Left / Line Right) by Date ==============

st.subheader("Combo Chart — By Date (Bar Left • Line Right)")
# Siapkan kandidat metrik numerik
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
# Tambahkan kolom-kolom umum yang mungkin berupa string angka
preferred_metric_names = ["engagement", "likes", "comments", "shares", "views", "impressions", "reach", "er"]
for c in df.columns:
    if c not in numeric_cols and c.lower() in preferred_metric_names:
        numeric_cols.append(c)

# Opsi metrik sumbu kiri/kanan
LEFT_COUNT_ALIAS = "Posts (count)"
left_choices = [LEFT_COUNT_ALIAS] + numeric_cols
right_choices = numeric_cols.copy()

# Default: kiri = count; kanan = Engagement jika ada
default_left_idx = 0
if "Engagement" in right_choices:
    default_right_idx = right_choices.index("Engagement")
elif "engagement" in right_choices:
    default_right_idx = right_choices.index("engagement")
else:
    default_right_idx = 0 if right_choices else None

c1, c2, c3 = st.columns([1.2, 1.2, 2])

with c1:
    left_metric_choice = st.selectbox("Left Axis (Bar)", left_choices, index=default_left_idx)
with c2:
    if right_choices:
        right_metric_choice = st.selectbox("Right Axis (Line)", right_choices, index=default_right_idx)
    else:
        st.error("Tidak ada kolom numerik terdeteksi untuk Right Axis.")
        st.stop()

# Date range filter
dates_parsed = pd.to_datetime(df[date_col], errors="coerce")
date_min_default = dates_parsed.min()
date_max_default = dates_parsed.max()
with c3:
    if pd.isna(date_min_default) or pd.isna(date_max_default):
        st.warning("Kolom tanggal belum valid. Pastikan kolom Date/Tanggal bisa diparse ke datetime.")
        st.stop()
    chosen_range = st.date_input(
        "Filter Date Range",
        value=(date_min_default.date(), date_max_default.date()),
        min_value=date_min_default.date(),
        max_value=date_max_default.date()
    )
    if isinstance(chosen_range, tuple) and len(chosen_range) == 2:
        dmin, dmax = chosen_range
    else:
        dmin, dmax = date_min_default.date(), date_max_default.date()

# Kunci agregasi
left_key = "__count_posts__" if left_metric_choice == LEFT_COUNT_ALIAS else left_metric_choice
right_key = right_metric_choice

ts = build_timeseries(df, date_col, left_key, right_key, dmin, dmax)

# === Plot combo (anti-niban label) ===
ts_vis = ts.copy()
ts_vis["Left"] = ts_vis["Left"].fillna(0)
ts_vis["Right"] = ts_vis["Right"].fillna(0)

fig = make_subplots(specs=[[{"secondary_y": True}]])

BAR_COLOR  = "#f2b01e"   # bar
LINE_COLOR = "#1f77b4"   # line

# BAR = Left axis, label di DALAM bar bagian bawah
fig.add_bar(
    x=ts_vis["Date"],
    y=ts_vis["Left"],
    name=left_metric_choice,
    marker_color=BAR_COLOR,
    opacity=0.95,
    text=ts_vis["Left"],
    texttemplate="%{text:,.0f}",
    textposition="inside",
    insidetextanchor="start",      # nempel di base (bawah)
    textfont=dict(size=10),
    cliponaxis=False,
    hovertemplate="<b>%{x}</b><br>" + left_metric_choice + ": %{y:,.0f}<extra></extra>",
    secondary_y=False,
)

# LINE = Right axis (tanpa text trace); label pakai annotation dengan shift dinamis
fig.add_scatter(
    x=ts_vis["Date"],
    y=ts_vis["Right"],
    name=right_metric_choice,
    mode="lines+markers",
    line=dict(width=3, color=LINE_COLOR),
    marker=dict(size=6),
    cliponaxis=False,
    hovertemplate="<b>%{x}</b><br>" + right_metric_choice + ": %{y:,.0f}<extra></extra>",
    secondary_y=True,
)

# hitung kuartil tinggi bar untuk patokan shift
if len(ts_vis):
    q2 = float(np.nanpercentile(ts_vis["Left"], 50))
    q3 = float(np.nanpercentile(ts_vis["Left"], 75))
else:
    q2 = q3 = 0.0

# Annotation untuk LINE (floating) + anti-niban
for i, (xi, y_right, y_left) in enumerate(zip(ts_vis["Date"], ts_vis["Right"], ts_vis["Left"])):
    yshift, xshift = _calc_line_label_shift(y_left, i, q2, q3)
    fig.add_annotation(
        x=xi, y=y_right, yref="y2",
        text=f"{int(y_right):,}",
        showarrow=False,
        bgcolor="rgba(220,220,220,0.85)",
        font=dict(size=10),
        borderpad=3,
        yshift=yshift,
        xshift=xshift
    )

# Layout & axes
left_max  = float(np.nanmax(ts_vis["Left"]))  if len(ts_vis) else 0.0
right_max = float(np.nanmax(ts_vis["Right"])) if len(ts_vis) else 0.0

# auto-kecilkan font tanggal agar semua tick tampil
n_days = len(ts_vis)
tick_size = 12 if n_days <= 14 else 10 if n_days <= 31 else 9 if n_days <= 62 else 8

fig.update_layout(
    height=420,
    bargap=0.35,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)

# tampilkan SEMUA tanggal harian, format 12-Oct
fig.update_xaxes(
    title_text="Date",
    tickmode="linear",
    dtick="D1",
    tickformat="%d-%b",
    tickfont=dict(size=tick_size),
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
)

fig.update_yaxes(
    title_text=f"{left_metric_choice}",
    range=[0, left_max * 1.25 if left_max > 0 else 1],
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    secondary_y=False
)
fig.update_yaxes(
    title_text=f"{right_metric_choice}",
    range=[0, right_max * 1.25 if right_max > 0 else 1],
    showgrid=False,
    secondary_y=True
)

st.plotly_chart(fig, use_container_width=True)

# Download time series
buf_ts = io.BytesIO()
ts_out = ts.rename(columns={"Left": f"Left ({left_metric_choice})", "Right": f"Right ({right_metric_choice})"})
with pd.ExcelWriter(buf_ts, engine="openpyxl") as writer:
    ts_out.to_excel(writer, sheet_name="TimeSeries", index=False)
st.download_button(
    "📥 Download Time Series (Excel)",
    data=buf_ts.getvalue(),
    file_name="timeseries_combo.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


# ============== Chart 3: Sentiment Breakdown (Stacked Horizontal) ==============
st.markdown("---")
st.subheader("Sentiment Breakdown — Stacked Bar")

# 1) Pilih kolom Sentiment + (opsional) filter Campaign
sentiment_candidates = [c for c in df.columns if c.lower() in [
    "sentiment", "new sentiment", "new_sentiment", "sentiment fix",
    "new sentiment level", "new_sentiment_level", "sentiment_level",
    "sentiment level"
]]
sent_col_default = sentiment_candidates[0] if sentiment_candidates else df.columns[0]
sent_col = st.selectbox("Kolom Sentiment", list(df.columns), index=list(df.columns).index(sent_col_default))

# (Opsional) filter campaign biar bisa lihat per-campaign
campaign_values = sorted([x for x in df[campaign_col].dropna().astype(str).unique()]) if campaign_col in df.columns else []
selected_campaigns = st.multiselect("Filter Campaign (opsional)", campaign_values)

work = df.copy()
if selected_campaigns:
    work = work[work[campaign_col].astype(str).isin(selected_campaigns)]

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
    with c1: st.metric("Total Posts", total)
    with c2: st.metric("Positive", fmt_pct(float(perc.get("Positive", 0))))
    with c3: st.metric("Neutral",  fmt_pct(float(perc.get("Neutral", 0))))
    with c4: st.metric("Negative", fmt_pct(float(perc.get("Negative", 0))))
