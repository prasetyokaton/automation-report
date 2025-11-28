# app.py - COMPLETE VERSION
# Dependencies: streamlit, pandas, plotly, openpyxl, wordcloud, matplotlib, openai
# Run: streamlit run app.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os, json, re, time
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from collections import Counter

# ============== LOGGING SETUP ==============
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SOV & Time Series Combo", layout="wide")
st.title("📈 Automation Report Insights")

load_dotenv(dotenv_path=".secretcontainer/.env", override=True)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if OPENAI_MODEL.strip().startswith("os.getenv"):
    st.error("OPENAI_MODEL di .env berisi ekspresi, bukan nama model. Contoh yang benar: OPENAI_MODEL=gpt-4o-mini")
    logger.error("Invalid OPENAI_MODEL configuration")

BANNED_CHARS = ["—", "–", "≈"]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    m = re.search(r"\d+", str(raw))
    try:
        return int(m.group(0)) if m else int(default)
    except Exception:
        return int(default)

OPENAI_MAX_OUT = _env_int("OPENAI_MAX_OUT", 700)

# ==== Helpers for narrative ====
def _word_trim(s: str, max_words=200) -> str:
    if not s: return s
    w = s.split()
    return " ".join(w[:max_words])

def _sanitize_output(s: str) -> str:
    if not s:
        return s
    for ch in BANNED_CHARS:
        s = s.replace(ch, " ")
    # Jangan hapus " - " lagi supaya bullet / penghubung aman
    s = s.replace("Others", "selebihnya").replace("others", "selebihnya")
    # Rapikan spasi berlebih
    return " ".join(s.split())

def _finalize_paragraph(s, max_words=200):
    s = _word_trim(s, max_words=max_words)
    if not s.endswith((".", "!", "?")):
        last = max(s.rfind("."), s.rfind("!"), s.rfind("?"))
        if last >= 50:
            s = s[:last+1]
    return _sanitize_output(s)

def _shorten_words(s, max_words=30):
    if pd.isna(s): return ""
    w = str(s).split()
    return " ".join(w[:max_words])

def _content_or_title(row, content_col, title_col, max_words=40):
    c = (str(row[content_col]).strip() if content_col and content_col in row and pd.notna(row[content_col]) else "")
    t = (str(row[title_col]).strip()   if title_col   and title_col   in row and pd.notna(row[title_col])   else "")
    txt = c if c else t
    return _shorten_words(txt, max_words=max_words)

def _sov_share_lookup(tbl_df, name: str, post=True):
    col = "SOV Post%" if post else "SOV Engagement%"
    try:
        row = tbl_df.loc[tbl_df["Campaign"].astype(str)==str(name), col]
        if len(row): 
            return float(row.iloc[0])
    except:
        pass
    return None

# ============== Helpers ==============

@st.cache_data(show_spinner=False)
def read_excel_all_sheets(file):
    logger.info(f"Reading Excel file: {getattr(file, 'name', 'uploaded')}")
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
    candidates_by_name = ["date", "created", "created at", "created_at",
                          "post date", "post_date", "published", "timestamp",
                          "time", "tanggal"]
    by_name, norm = [], {c.lower().strip(): c for c in df.columns}
    for nm in candidates_by_name:
        if nm in norm: by_name.append(norm[nm])
    by_dtype = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    seen, ordered = set(), []
    for c in by_name + by_dtype:
        if c not in seen:
            ordered.append(c); seen.add(c)
    return ordered[0] if ordered else None

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")

def prep_sov(df, campaign_col):
    ser = (df[campaign_col].astype(str).str.strip()
           .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).dropna())
    sov = (ser.to_frame(name=campaign_col).groupby(campaign_col, dropna=True)
             .size().reset_index(name="Post")
             .sort_values("Post", ascending=False).reset_index(drop=True))
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
    lowmap = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lowmap:
            return lowmap[a.lower()]
    return None

def topn_generic(df, label_col, value_col, top_n: int):
    if df.empty or value_col not in df.columns: return df.copy()
    data = df[[label_col, value_col]].sort_values(value_col, ascending=False)
    if len(data) <= top_n: return data.reset_index(drop=True)
    top = data.head(top_n).copy()
    others_val = data[value_col].iloc[top_n:].sum()
    if others_val > 0: top.loc[len(top)] = ["Others", others_val]
    return top.reset_index(drop=True)

def build_timeseries(df, date_col, left_metric_key, right_metric_key, date_min=None, date_max=None):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    base = df.copy()
    base["__date"] = dates.dt.date
    base = base.dropna(subset=["__date"])
    if date_min: base = base[base["__date"] >= date_min]
    if date_max: base = base[base["__date"] <= date_max]

    def agg_metric(frame, key):
        if key in {"__count_post__", "__count_articles__"}:
            return frame.groupby("__date").size().rename("value")
        vals = coerce_numeric(frame[key])
        tmp = frame.assign(_val=vals)
        return tmp.groupby("__date")["_val"].sum(min_count=1).rename("value")

    left_series  = agg_metric(base, left_metric_key)
    right_series = agg_metric(base, right_metric_key)

    ts = pd.DataFrame({"Date": left_series.index, "Left": left_series.values})
    ts = pd.merge(ts, right_series.rename_axis("Date").reset_index(), on="Date", how="outer")
    ts.rename(columns={"value": "Right"}, inplace=True)
    ts = ts.sort_values("Date").reset_index(drop=True)
    return ts

def _calc_line_label_shift(bar_val, idx, q2, q3):
    if pd.isna(bar_val) or bar_val <= 0: base_y = 10
    elif bar_val >= q3: base_y = 28
    elif bar_val >= q2: base_y = 20
    else: base_y = 14
    xshift = 8 if (idx % 2 == 0) else -8
    return base_y, xshift

def normalize_channel(val: str):
    if pd.isna(val): return None
    s = str(val).strip().lower()
    if s in {"youtube","you tube","yt"}: return "YouTube"
    if s in {"tiktok","tik tok","tik-tok"}: return "Tiktok"
    if s in {"instagram","ig"}: return "Instagram"
    if s in {"twitter","x","tw"}: return "Twitter"
    if s in {"facebook","fb"}: return "Facebook"
    if s in {"forum","kaskus","reddit"}: return "Forum"
    if s in {"online media","online","portal","news","website"}: return "Online Media"
    if s in {"printedmedia","printed","print","koran","majalah","newspaper","tabloid"}: return "Printed"
    if s in {"tv","television","televisi"}: return "TV"
    return str(val).strip().title()

def fmt_id_int(x):
    try: return f"{int(round(float(x))):,}".replace(",", ".")
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
    if not channel_list:
        channel_list = ["YouTube","Tiktok","Instagram","Twitter","Facebook","Forum"] \
            if title.lower().startswith("social") else ["Online Media","Printed","TV"]

    base = df_in.copy()
    base["__chan"] = base[channel_col].map(normalize_channel)

    post = base.groupby("__chan").size().rename("__post__")

    if left_key in {"__count_post__","__count_articles__"}:
        left_ser = post.rename(left_title)
    else:
        left_ser = _agg_sum_by_channel(base, left_key).rename(left_title)
    right_ser = _agg_sum_by_channel(base, right_key).rename(right_title)

    idx = pd.Index(channel_list, name="__chan")
    agg = (pd.concat([post, left_ser, right_ser], axis=1).reindex(idx).fillna(0.0).reset_index())
    agg.rename(columns={"__post__": "Post"}, inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    BAR_COLOR, LINE_COLOR = "#f2b01e", "#1f77b4"

    fig.add_bar(
        x=agg["__chan"], y=agg[left_title], name=left_title,
        marker_color=BAR_COLOR, opacity=0.95,
        text=[fmt_id_int(v) for v in agg[left_title]], texttemplate="%{text}",
        textposition="inside", insidetextanchor="start",
        textfont=dict(size=10),
        hovertemplate=f"<b>%{{x}}</b><br>{left_title}: %{{y:,}}<extra></extra>",
        secondary_y=False
    )
    fig.add_scatter(
        x=agg["__chan"], y=agg[right_title], name=right_title,
        mode="lines+markers", line=dict(width=3, color=LINE_COLOR), marker=dict(size=7),
        hovertemplate=f"<b>%{{x}}</b><br>{right_title}: %{{y:,}}<extra></extra>",
        secondary_y=True
    )
    for i, (xi, yv) in enumerate(zip(agg["__chan"], agg[right_title])):
        fig.add_annotation(
            x=xi, y=yv, yref="y2", text=fmt_id_int(yv),
            showarrow=False, bgcolor="rgba(220,220,220,0.85)",
            font=dict(size=10), borderpad=3, yshift=6,
            xshift=(7 if i % 2 == 0 else -7)
        )

    fig.update_layout(
        title=title, height=420, bargap=0.35, hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="center", x=0.5),
    )
    
    left_max  = float(agg[left_title].max())  if len(agg) else 0.0
    right_max = float(agg[right_title].max()) if len(agg) else 0.0

    fig.update_xaxes(
        title_text=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickangle=0
    )
    fig.update_yaxes(
        title_text=left_title, secondary_y=False, showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        range=[0, left_max * 1.25 if left_max > 0 else 1], autorange=False
    )
    fig.update_yaxes(
        title_text=right_title, secondary_y=True, showgrid=False,
        range=[0, right_max * 1.25 if right_max > 0 else 1], autorange=False
    )

    return fig, agg

# ============== AI HELPERS ==============

def _get_openai_client():
    try:
        return OpenAI()
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}")
        st.error(f"Gagal init OpenAI client: {e}")
        return None

def _is_reasoning_model(name: str) -> bool:
    s = (name or "").lower()
    return s.startswith("gpt-5") or "-r" in s

def responses_create_safe(client, **kwargs):
    try:
        return client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e)
        strip_keys = []
        if "Unsupported parameter: 'temperature'" in msg:
            strip_keys.append("temperature")
        if "Unsupported parameter: 'top_p'" in msg:
            strip_keys.append("top_p")
        if "Unsupported parameter: 'max_output_tokens'" in msg:
            strip_keys.append("max_output_tokens")

        if strip_keys:
            for k in strip_keys:
                kwargs.pop(k, None)
            return client.responses.create(**kwargs)
        raise

def _safe_output_text(resp):
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt.strip()
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if t:
                    parts.append(t)
        if parts:
            return "\n".join(parts).strip()
    except:
        pass
    if hasattr(resp, "choices") and resp.choices:
        msg = getattr(resp.choices[0], "message", None)
        if msg and getattr(msg, "content", None):
            return msg.content.strip()
    return ""

def generate_single_narrative(client, prompt, max_words=200, max_retries=2):
    """Generate narrative with retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Generating narrative (attempt {attempt + 1}/{max_retries})")
            
            req = dict(
                model=OPENAI_MODEL, 
                input=prompt, 
                max_output_tokens=max(380, OPENAI_MAX_OUT), 
                temperature=0.2
            )
            
            if _is_reasoning_model(OPENAI_MODEL):
                req["reasoning"] = {"effort": "low"}
            
            resp = responses_create_safe(client, **req)
            narrative_raw = _safe_output_text(resp)

            if (not narrative_raw) or (getattr(resp, "status", "") == "incomplete"):
                req["max_output_tokens"] = max(req.get("max_output_tokens", 0), 1200)
                if _is_reasoning_model(OPENAI_MODEL):
                    req["reasoning"] = {"effort": "low"}
                resp = responses_create_safe(client, **req)
                narrative_raw = _safe_output_text(resp)

            narrative = _finalize_paragraph(narrative_raw, max_words=max_words)
            if narrative:
                logger.info("Narrative generated successfully")
                return narrative
            else:
                raise ValueError("Model returned empty text")
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)  # Wait 1 second before retry
    
    return None

# ============== SENTIMENT HELPERS ==============

def normalize_sentiment(v):
    s = str(v).strip().lower()
    if s in {"positive","positif","pos","+1","1","good","bagus"}: return "Positive"
    if s in {"negative","negatif","neg","-1","bad","buruk"}: return "Negative"
    if s in {"neutral","netral","neu","0"}: return "Neutral"
    try:
        f = float(s)
        if f > 0: return "Positive"
        if f < 0: return "Negative"
        return "Neutral"
    except: return "Neutral"

def sentiment_bar(df_src, title):
    """Render sentiment bar chart and return counts for table"""
    if df_src.empty:
        st.info(f"Tidak ada data untuk {title}.")
        return None
    sent_col = pick_col(df_src, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix","sentiment level","new sentiment level","new_sentiment_level"])
    if not sent_col:
        st.info(f"Kolom Sentiment tidak ditemukan untuk {title}.")
        return None
    work = df_src.copy()
    work["__sent"] = work[sent_col].map(normalize_sentiment)
    order  = ["Positive", "Neutral", "Negative"]
    colors = {"Positive": "#43A047", "Neutral": "#BDBDBD", "Negative": "#E53935"}
    labels_bg = {"Positive": "rgba(67,160,71,0.85)", "Neutral": "rgba(189,189,189,0.85)", "Negative": "rgba(229,57,53,0.85)"}
    counts = work["__sent"].value_counts().reindex(order, fill_value=0)
    total  = int(counts.sum())
    if total == 0:
        st.info(f"Tidak ada data sentiment pada {title}.")
        return None
    perc = (counts / total * 100).round(2)
    def fmt_pct(x): return f"{str(f'{x:.2f}').replace('.',',')}%"
    fig_s = go.Figure()
    for name in order:
        val = float(perc.get(name, 0.0))
        fig_s.add_bar(y=[" "], x=[val], name=name, orientation="h",
                      marker_color=colors[name],
                      hovertemplate=f"<b>{name}</b><br>{fmt_pct(val)}<extra></extra>", text=None)
    cum = 0.0
    for name in order:
        val = float(perc.get(name, 0.0))
        if val <= 0: continue
        x_mid = cum + val/2.0
        fig_s.add_annotation(x=x_mid, y=0, xref="x", yref="y",
                             text=f"{name}<br>{fmt_pct(val)}",
                             showarrow=False, font=dict(size=16, color="white"),
                             bgcolor=labels_bg[name], borderpad=4)
        cum += val
    fig_s.update_layout(barmode="stack", height=160, margin=dict(l=10,r=10,t=10,b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5))
    fig_s.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=False, title=None)
    fig_s.update_yaxes(visible=False)
    st.plotly_chart(fig_s, use_container_width=True)
    
    # Return counts for table display
    return counts

# ============== TOPIC CHART HELPERS ==============

def _fmt_int(v):
    try: return f"{int(round(float(v))):,}"
    except: return str(v)

def _topic_bar_plot(df_src, topic_col, title, color_hex, metric_kind, value_col=None, top_n=None):
    if topic_col not in df_src.columns or df_src.empty:
        st.info("Data Not Found")
        return

    if metric_kind == "post":
        agg = df_src.groupby(topic_col).size().rename("Value").reset_index()
    else:
        vals = coerce_numeric(df_src[value_col]) if value_col and value_col in df_src.columns else None
        if vals is None: 
            st.info("Data Not Found")
            return
        agg = df_src.assign(_v=vals).groupby(topic_col)["_v"].sum(min_count=1).rename("Value").reset_index()

    agg = agg.sort_values("Value", ascending=False)
    if top_n and top_n > 0:
        agg = agg.head(top_n)
    if agg.empty: 
        st.info("Data Not Found")
        return

    fig = go.Figure()
    fig.add_bar(
        x=agg["Value"], 
        y=agg[topic_col],
        orientation="h", 
        marker_color=color_hex, 
        text=None
    )

    vmax = float(agg["Value"].max())
    for y, v in zip(agg[topic_col], agg["Value"]):
        v_float = float(v)
        fig.add_annotation(
            x=v_float + 0.02 * (vmax if vmax > 0 else 1),
            y=y, text=_fmt_int(v), showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(color="white", size=18),
            bgcolor=color_hex, bordercolor=color_hex, borderwidth=2, borderpad=3
        )

    fig.update_layout(
        title=title, height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.25, hovermode=False
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", rangemode="tozero")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), showgrid=False)
    st.plotly_chart(fig, use_container_width=True)



# ============== MAIN APP ==============

logger.info("=" * 80)
logger.info("Application started")

st.sidebar.header("1) Upload File")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False)

if not uploaded:
    st.info("Upload file Excel kamu dulu. Pastikan ada kolom **Campaigns/Campaign** dan **Date**.")
    st.stop()

sheets = read_excel_all_sheets(uploaded)
sheet_names = list(sheets.keys())

st.sidebar.header("2) Pilih Sheet")
sheet_name = st.sidebar.selectbox("Sheet", sheet_names, index=0)
df = sheets[sheet_name]

logger.info(f"Sheet selected: {sheet_name}, Shape: {df.shape}")

# Session state
ds_sig = f"{getattr(uploaded, 'name', 'uploaded')}::{sheet_name}"
if "dataset_sig" not in st.session_state or st.session_state["dataset_sig"] != ds_sig:
    st.session_state["dataset_sig"] = ds_sig
    # Initialize all narrative keys
    for k in ["narr_sov", "narr_trend_social", "narr_trend_main", 
              "narr_sent_social", "narr_sent_main",
              "narr_channel_social", "narr_channel_main",
              "narr_topic_soc_pos", "narr_topic_soc_neu", "narr_topic_soc_neg",
              "narr_topic_main_pos", "narr_topic_main_neu", "narr_topic_main_neg",
              "narr_insights", "narr_recommendations"]:
        st.session_state[k] = ""
else:
    for k in ["narr_sov", "narr_trend_social", "narr_trend_main",
              "narr_sent_social", "narr_sent_main",
              "narr_channel_social", "narr_channel_main",
              "narr_topic_soc_pos", "narr_topic_soc_neu", "narr_topic_soc_neg",
              "narr_topic_main_pos", "narr_topic_main_neu", "narr_topic_main_neg",
              "narr_insights", "narr_recommendations"]:
        if k not in st.session_state:
            st.session_state[k] = ""

# Auto-detect columns
campaign_col = find_campaign_column(df.columns) or df.columns[0]
date_col     = find_date_column(df) or df.columns[0]
channel_col  = pick_col(df, ["Channel","Channels","Platform","Media","Source","Tipe","Type"]) or df.columns[0]
title_col    = pick_col(df, ["Title", "Judul", "Post Title", "title"])
content_col  = pick_col(df, ["Content", "Konten", "Caption", "Text", "Isi", "content", "caption", "text"])
topic_col    = pick_col(df, ["Topic","topic","Topik"])
sent_col     = pick_col(df, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix",
                              "sentiment level","new sentiment level","new_sentiment_level"])
media_col    = pick_col(df, ["Media Name", "media name", "Media", "Source", "Publisher", "Outlet", "media", "source"])

# Campaign filters
campaign_values_all = sorted(df[campaign_col].astype(str).dropna().unique().tolist())
selected_campaigns_chart1 = st.sidebar.multiselect(
    "Filter Campaign", 
    options=campaign_values_all, 
    default=campaign_values_all, 
    key="campaign_filter_chart1"
)
if not selected_campaigns_chart1:
    selected_campaigns_chart1 = campaign_values_all

# Channel filters
st.sidebar.markdown("---")
st.sidebar.subheader("Channel Filters")
SOCIAL_ORDER = ["Instagram", "Tiktok", "Twitter", "Facebook", "YouTube", "Forum"]
selected_social_channels = st.sidebar.multiselect(
    "Social Channels", 
    options=SOCIAL_ORDER, 
    default=SOCIAL_ORDER, 
    key="social_channels_sel"
)

MAINSTREAM_ORDER = ["Online Media", "Printed", "TV"]
selected_mainstream_channels = st.sidebar.multiselect(
    "Mainstream Channels", 
    options=MAINSTREAM_ORDER, 
    default=MAINSTREAM_ORDER, 
    key="mainstream_channels_sel"
)

# Language selector
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Narrative Settings")
narrative_lang = st.sidebar.selectbox(
    "Language",
    options=["ID", "EN"],
    index=0,
    help="Select language for AI-generated narratives"
)

# Reset button
def _reset_all():
    st.session_state["campaign_filter_chart1"]  = campaign_values_all
    st.session_state["social_channels_sel"]     = SOCIAL_ORDER
    st.session_state["mainstream_channels_sel"] = MAINSTREAM_ORDER
    
    num_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    fallback_right = next((nm for nm in ["Engagement","engagement"] if nm in df.columns),
                          (num_candidates[0] if num_candidates else None))
    pr_fallback = pick_col(df, ["PR Value","pr value","pr_value","prvalue"])
    
    st.session_state["ts_left"] = "Volume Post"
    if fallback_right: st.session_state["ts_right"] = fallback_right
    st.session_state["ts_ms_left"] = "Total Artikel"
    if pr_fallback: st.session_state["ts_ms_right"] = pr_fallback
    st.session_state["ch4_left"] = "Volume Post"
    if fallback_right: st.session_state["ch4_right"] = fallback_right
    st.session_state["ch5_left"] = "Volume Artikel"
    st.session_state["ch5_right"] = pr_fallback or (fallback_right if fallback_right else (num_candidates[0] if num_candidates else None))

st.sidebar.button("🔄 Reset Filters", use_container_width=True, on_click=_reset_all)

# ============== AI NARRATIVE GENERATION (SIDEBAR) ==============
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Narratives")
st.sidebar.caption(f"⚠️ Will generate 15 narratives in {narrative_lang} (~90-120 seconds)")

if st.sidebar.button("✨ Generate All AI Narratives", use_container_width=True, type="primary"):
    client = _get_openai_client()
    if not client:
        st.sidebar.error("OpenAI client initialization failed")
    else:
        # Language instruction
        lang_instruction = "Tulis dalam Bahasa Indonesia yang natural dan mudah dipahami." if narrative_lang == "ID" else "Write in clear, professional English."
        
        # Prepare filtered data
        df_work = df[df[campaign_col].astype(str).isin(selected_campaigns_chart1)].copy()
        if df_work.empty:
            df_work = df.copy()
        
        df_work_norm = df_work.copy()
        df_work_norm["__chan_norm"] = df_work_norm[channel_col].map(normalize_channel)
        
        df_social_metrics = df_work_norm[df_work_norm["__chan_norm"].isin(selected_social_channels)]
        df_main_metrics   = df_work_norm[df_work_norm["__chan_norm"].isin(selected_mainstream_channels)]
        
        eng_name  = pick_col(df, ["Engagement", "Total Engagement", "engagement","interactions","interaction"])
        buzz_name = pick_col(df, ["Buzz","buzz","PR Value","pr value","pr_value","prvalue"])
        pr_col    = pick_col(df, ["PR Value","pr value","pr_value","prvalue"])
        
        # Progress tracking
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        narratives_to_generate = [
            ("narr_sov", "SOV Comparison", 200),
            ("narr_trend_social", "Social Trends", 200),
            ("narr_trend_main", "Mainstream Trends", 200),
            ("narr_sent_social", "Sentiment Social", 150),
            ("narr_sent_main", "Sentiment Mainstream", 150),
            ("narr_channel_social", "Social Channels", 200),
            ("narr_channel_main", "Mainstream Channels", 200),
            ("narr_topic_soc_pos", "Topic Social Positive", 120),
            ("narr_topic_soc_neu", "Topic Social Neutral", 120),
            ("narr_topic_soc_neg", "Topic Social Negative", 120),
            ("narr_topic_main_pos", "Topic Mainstream Positive", 120),
            ("narr_topic_main_neu", "Topic Mainstream Neutral", 120),
            ("narr_topic_main_neg", "Topic Mainstream Negative", 120),
            ("narr_insights", "Insights Summary", 500),
            ("narr_recommendations", "Recommendations", 500),
        ]
        
        total = len(narratives_to_generate)
        success_count = 0
        failed = []
        
        for idx, (key, name, max_words) in enumerate(narratives_to_generate):
            status_text.text(f"Generating {idx+1}/{total}: {name}...")
            progress_bar.progress((idx) / total)
            
            # Generate prompt based on key
            prompt = ""
            
            if key == "narr_sov":
                # SOV prompt (existing logic)
                grp_post = df_work.groupby(campaign_col).size().rename("Total Post")
                tbl = grp_post.to_frame()
                if eng_name and eng_name in df_work.columns:
                    grp_eng = (df_work.assign(_eng=coerce_numeric(df_work[eng_name]))
                               .groupby(campaign_col)["_eng"].sum(min_count=1).rename("Total Engagement"))
                    tbl = tbl.join(grp_eng, how="left")
                else:
                    tbl["Total Engagement"] = 0
                
                tot_post_sum = int(tbl["Total Post"].sum()) if len(tbl) else 0
                tot_eng_sum  = float(tbl["Total Engagement"].sum()) if len(tbl) else 0.0
                
                tbl["SOV Post%"]       = (tbl["Total Post"] / tot_post_sum * 100).round(2) if tot_post_sum else 0.0
                tbl["SOV Engagement%"] = (tbl["Total Engagement"] / tot_eng_sum * 100).round(2) if tot_eng_sum else 0.0
                
                tbl = (tbl.reset_index()
                          .rename(columns={campaign_col: "Campaign"})
                          .sort_values(["Total Post", "Total Engagement"], ascending=False))
                
                top_n = 8
                top_post = topn_generic(tbl.rename(columns={"Total Post":"Value"}), "Campaign", "Value", top_n)
                top_eng  = topn_generic(tbl.rename(columns={"Total Engagement":"Value"}), "Campaign", "Value", top_n)
                
                tp = [r for r in top_post.to_dict("records") if r.get("Campaign") != "Others"]
                te = [r for r in top_eng.to_dict("records")   if r.get("Campaign") != "Others"]
                
                keep_names = []
                for r in tp[:3]:
                    n = str(r["Campaign"])
                    if n not in keep_names: keep_names.append(n)
                for r in te[:3]:
                    n = str(r["Campaign"])
                    if n not in keep_names: keep_names.append(n)
                keep_names = keep_names[:3] or ([str(tp[0]["Campaign"])] if tp else [])
                
                ctx = []
                for name_camp in keep_names:
                    sub = df_work[df_work[campaign_col].astype(str) == str(name_camp)].copy()
                    eg = []
                    if eng_name and eng_name in sub.columns:
                        sub["_eng"] = coerce_numeric(sub[eng_name])
                        for _, r in sub.sort_values("_eng", ascending=False).head(3).iterrows():
                            p = _content_or_title(r, content_col, title_col, 28)
                            if p: eg.append(p)
                    ctx.append({
                        "campaign": name_camp,
                        "post_share_pct": _sov_share_lookup(tbl, name_camp, post=True),
                        "eng_share_pct":  _sov_share_lookup(tbl, name_camp, post=False),
                        "content_examples": eg[:3]
                    })
                
                prompt = f"""
{lang_instruction}

Anda adalah analis insight senior. Tulis satu paragraf maksimum {max_words} kata.
Fokus merangkum isi konten, bukan nama topik. Hindari titik dua, titik koma, dash, dan simbol aneh.
Jangan membuat bullet. Jangan gunakan kata other. Tutup paragraf dengan kalimat yang selesai.

Tugas:
1) Sebut kampanye dengan porsi SOV Post terbesar dan ringkas isi konten paling sering muncul dari contoh.
2) Sebut kampanye dengan porsi SOV Engagement terbesar dan jelaskan singkat mengapa interaksinya tinggi berdasarkan gaya/isi konten.
3) Sentuh kampanye terbesar berikutnya bila relevan.
4) Akhiri satu kalimat yang merangkum arah percakapan lintas kampanye.

Data:
{json.dumps(ctx, ensure_ascii=False)}
""".strip()
            
            elif key == "narr_trend_social":
                prompt = f"""
{lang_instruction}

Anda adalah analis insight senior. Tulis satu paragraf maksimum {max_words} kata.
Fokus pada tren percakapan social media. Hindari titik dua, titik koma, dash, dan simbol aneh.

Tugas:
1) Jelaskan tren volume post dan engagement pada periode yang difilter.
2) Identifikasi peak/puncak aktivitas jika ada.
3) Tutup dengan takeaway utama dari tren ini.

Data: Gunakan data social media yang sudah difilter berdasarkan campaign dan channel yang dipilih user.
""".strip()
            
            elif key == "narr_trend_main":
                prompt = f"""
{lang_instruction}

Anda adalah analis insight senior. Tulis satu paragraf maksimum {max_words} kata.
Fokus pada tren pemberitaan mainstream media. Hindari titik dua, titik koma, dash, dan simbol aneh.

Tugas:
1) Jelaskan tren volume artikel dan PR value pada periode yang difilter.
2) Identifikasi peak/puncak coverage jika ada.
3) Tutup dengan takeaway utama dari tren media mainstream ini.

Data: Gunakan data mainstream yang sudah difilter berdasarkan campaign dan channel yang dipilih user.
""".strip()
            
            elif key == "narr_sent_social":
                sent_col = pick_col(df_social_metrics, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix"])
                if sent_col:
                    work = df_social_metrics.copy()
                    work["__sent"] = work[sent_col].map(normalize_sentiment)
                    counts = work["__sent"].value_counts()
                    prompt = f"""
{lang_instruction}

Anda adalah analis insight senior. Tulis satu paragraf maksimum {max_words} kata.
Fokus pada distribusi sentiment social media. Hindari titik dua, titik koma, dash, dan simbol aneh.

Tugas:
1) Jelaskan distribusi sentiment (Positive/Neutral/Negative) pada data social media.
2) Sorot sentiment yang paling dominan dan berikan insight singkat.
3) Tutup dengan implicasi dari distribusi sentiment ini.

Data sentiment:
{counts.to_dict()}
""".strip()
                else:
                    prompt = "Data sentiment tidak tersedia untuk social media."
            
            elif key == "narr_sent_main":
                sent_col = pick_col(df_main_metrics, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix"])
                if sent_col:
                    work = df_main_metrics.copy()
                    work["__sent"] = work[sent_col].map(normalize_sentiment)
                    counts = work["__sent"].value_counts()
                    prompt = f"""
{lang_instruction}

Anda adalah analis insight senior. Tulis satu paragraf maksimum {max_words} kata.
Fokus pada distribusi sentiment mainstream media. Hindari titik dua, titik koma, dash, dan simbol aneh.

Tugas:
1) Jelaskan distribusi sentiment (Positive/Neutral/Negative) pada pemberitaan mainstream.
2) Sorot tone pemberitaan yang paling dominan.
3) Tutup dengan implicasi dari tone media mainstream ini.

Data sentiment:
{counts.to_dict()}
""".strip()
                else:
                    prompt = "Data sentiment tidak tersedia untuk mainstream media."
            
            elif key == "narr_channel_social":
                channels_for_chart = [c for c in SOCIAL_ORDER if c in selected_social_channels]
                df_social_section = df_work_norm[df_work_norm["__chan_norm"].isin(channels_for_chart)].copy()
                
                post_counts = df_social_section["__chan_norm"].value_counts().to_dict()
                eng_sum = {}
                if eng_name and eng_name in df_social_section.columns:
                    _vals_eng = coerce_numeric(df_social_section[eng_name])
                    eng_sum = df_social_section.assign(_v=_vals_eng).groupby("__chan_norm")["_v"].sum().to_dict()
                
                prompt = f"""
{lang_instruction}

Anda adalah analis insight senior. Tulis satu paragraf maksimum {max_words} kata.
Fokus pada perbandingan performa channel social media. Hindari titik dua, titik koma, dash, dan simbol aneh.

Tugas:
1) Identifikasi channel dengan volume post terbanyak dan engagement tertinggi.
2) Berikan insight mengapa channel tertentu perform lebih baik.
3) Tutup dengan rekomendasi channel focus.

Data:
Post per channel: {post_counts}
Engagement per channel: {eng_sum}
""".strip()
            
            elif key == "narr_channel_main":
                channels_for_main = [c for c in MAINSTREAM_ORDER if c in selected_mainstream_channels]
                df_main_section = df_work_norm[df_work_norm["__chan_norm"].isin(channels_for_main)].copy()
                
                art_counts = df_main_section["__chan_norm"].value_counts().to_dict()
                pr_sum = {}
                if pr_col and pr_col in df_main_section.columns:
                    _vals_pr = coerce_numeric(df_main_section[pr_col])
                    pr_sum = df_main_section.assign(_v=_vals_pr).groupby("__chan_norm")["_v"].sum().to_dict()
                
                prompt = f"""
{lang_instruction}

Anda adalah analis insight senior. Tulis satu paragraf maksimum {max_words} kata.
Fokus pada distribusi coverage mainstream media. Hindari titik dua, titik koma, dash, dan simbol aneh.

Tugas:
1) Identifikasi channel dengan artikel terbanyak dan PR value tertinggi.
2) Berikan insight tentang distribusi coverage.
3) Tutup dengan strategi media relations.

Data:
Artikel per channel: {art_counts}
PR Value per channel: {pr_sum}
""".strip()
            
            elif key.startswith("narr_topic_"):
                # Topic narratives
                is_social = "soc" in key
                sentiment_type = "Positive" if "pos" in key else ("Neutral" if "neu" in key else "Negative")
                
                topic_col = pick_col(df, ["Topic","topic","Topik"])
                sent_col  = pick_col(df, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix"])
                
                if topic_col and sent_col:
                    tmp_topics = df_work_norm.copy()
                    tmp_topics["__sent"] = tmp_topics[sent_col].map(normalize_sentiment)
                    
                    if is_social:
                        df_subset = tmp_topics[tmp_topics["__chan_norm"].isin(selected_social_channels)]
                    else:
                        df_subset = tmp_topics[tmp_topics["__chan_norm"].isin(selected_mainstream_channels)]
                    
                    df_subset = df_subset[df_subset["__sent"] == sentiment_type]
                    
                    top_topics_post = df_subset.groupby(topic_col).size().sort_values(ascending=False).head(5).to_dict()
                    
                    metric_col = eng_name if is_social else (buzz_name or pr_col)
                    top_topics_metric = {}
                    if metric_col and metric_col in df_subset.columns:
                        vals = coerce_numeric(df_subset[metric_col])
                        top_topics_metric = df_subset.assign(_v=vals).groupby(topic_col)["_v"].sum().sort_values(ascending=False).head(5).to_dict()
                    
                    media_type = "social media" if is_social else "mainstream media"
                    prompt = f"""
{lang_instruction}

Anda adalah analis insight senior. Tulis satu paragraf maksimum {max_words} kata.
Fokus pada topik dengan sentiment {sentiment_type} di {media_type}. Hindari titik dua, titik koma, dash, dan simbol aneh.

Tugas:
1) Sebutkan 3 topik teratas dengan sentiment {sentiment_type}.
2) Jelaskan mengapa topik-topik ini mendapat sentiment tersebut.
3) Berikan insight singkat tentang implikasi dari topik-topik ini.

Data:
Top topics by post: {top_topics_post}
Top topics by metric: {top_topics_metric}
""".strip()
                else:
                    prompt = f"Data topic atau sentiment tidak tersedia untuk {sentiment_type}."
            
            elif key == "narr_insights":
                # ==== Insights Summary ====
                # Gather key metrics
                tot_post_social = len(df_social_metrics)
                tot_eng_social = int(coerce_numeric(df_social_metrics[eng_name]).sum()) if eng_name and eng_name in df_social_metrics.columns else 0
                tot_art_main = len(df_main_metrics)
                tot_pr_main = int(coerce_numeric(df_main_metrics[pr_col]).sum()) if pr_col and pr_col in df_main_metrics.columns else 0

                # Top campaigns by post
                grp_post = df_work.groupby(campaign_col).size().sort_values(ascending=False).head(3)
                top_campaigns = grp_post.to_dict()

                # Sentiment distribution
                sent_col_global = pick_col(df, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix"])
                if sent_col_global and sent_col_global in df_work.columns:
                    df_sent = df_work.copy()
                    df_sent["__sent"] = df_sent[sent_col_global].map(normalize_sentiment)
                    sent_dist = df_sent["__sent"].value_counts().to_dict()
                else:
                    sent_dist = {}

                # Contoh konten per top campaign biar narasi lebih nyambung ke isi
                content_examples = []
                for camp_name in top_campaigns.keys():
                    sub = df_work[df_work[campaign_col].astype(str) == str(camp_name)].copy()
                    ex_list = []
                    for _, r in sub.head(8).iterrows():
                        txt = _content_or_title(r, content_col, title_col, max_words=25)
                        if txt:
                            ex_list.append(txt)
                        if len(ex_list) >= 4:
                            break
                    content_examples.append({"campaign": str(camp_name), "examples": ex_list})

                data_ctx = {
                    "total_social_post": tot_post_social,
                    "total_social_engagement": tot_eng_social,
                    "total_mainstream_articles": tot_art_main,
                    "total_pr_value": tot_pr_main,
                    "top_campaigns": top_campaigns,
                    "sentiment_distribution": sent_dist,
                    "content_examples": content_examples,
                }

                if narrative_lang == "ID":
                    prompt = f"""
            Anda adalah senior data analyst untuk brand FMCG. Tulis insight summary maksimum {max_words} kata.

            Gunakan struktur berikut dalam format paragraf dengan heading yang jelas:

            1. OVERVIEW
            - Gambarkan secara ringkas performa campaign di social dan mainstream (volume, engagement, PR value).

            2. KEY FINDINGS
            - Sebutkan 3–4 temuan paling penting, misalnya:
                * Kampanye mana yang paling dominan dan apa pola isi kontennya.
                * Seperti apa efisiensi engagement per post.
                * Pola tema konten berdasarkan contoh di data (jangan hanya ulangi angka).

            3. SENTIMENT ANALYSIS
            - Jelaskan distribusi sentiment dan artinya bagi brand (apakah aman, perlu waspada, atau butuh perbaikan).

            4. COMPARISON
            - Bandingkan peran social vs mainstream: social sebagai mesin volume dan engagement, mainstream sebagai penguat kredibilitas.
            - Tutup dengan satu kalimat tentang implikasi strategis kombinasi keduanya.

            Gunakan bahasa Indonesia yang natural, jelas, dan mudah dipahami. Jangan pakai bullet yang terlalu teknis, cukup heading dan kalimat pendek. Hindari simbol aneh.

            Data ringkas (jangan ditiru mentah, gunakan hanya sebagai referensi analisis):

            {json.dumps(data_ctx, ensure_ascii=False, indent=2)}
            """.strip()
                else:
                    # English version
                    prompt = f"""
            You are a senior data analyst for an FMCG brand. Write an insight summary in clear professional English, maximum {max_words} words.

            Use the following structure with clear headings:

            1. OVERVIEW
            - Briefly describe the overall campaign performance across social and mainstream (volume, engagement, PR value).

            2. KEY FINDINGS
            - Highlight 3–4 key findings, such as:
                * Which campaigns dominate and what content patterns they use.
                * How efficient engagement per post looks.
                * The main themes of content based on the examples in the data (do not just repeat numbers).

            3. SENTIMENT ANALYSIS
            - Explain the sentiment distribution and what it implies for the brand (safe, needs attention, or requires corrective action).

            4. COMPARISON
            - Compare the roles of social vs mainstream: social as the volume and engagement engine, mainstream as credibility builder.
            - Close with one sentence on the strategic implication of combining both.

            Write in paragraphs with headings, no bullet list syntax. Use the data only as analytical context, not as a dump.

            Data summary (for your reference only):

            {json.dumps(data_ctx, ensure_ascii=False, indent=2)}
            """.strip()

            
            elif key == "narr_recommendations":
                # ==== Strategic Recommendations ====
                current_date = datetime.now().strftime("%B %Y")

                # Top & bottom campaigns berdasar volume
                grp_post_all = df_work.groupby(campaign_col).size().sort_values(ascending=False)
                top_campaign_name = grp_post_all.index[0] if len(grp_post_all) > 0 else "N/A"
                bottom_campaign_name = grp_post_all.index[-1] if len(grp_post_all) > 1 else "N/A"

                # Channel distribution
                channel_dist = df_work_norm["__chan_norm"].value_counts().to_dict()

                # Negative sentiment count
                sent_col_global = pick_col(df, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix"])
                negative_count = 0
                if sent_col_global and sent_col_global in df_work.columns:
                    df_sent = df_work.copy()
                    df_sent["__sent"] = df_sent[sent_col_global].map(normalize_sentiment)
                    negative_count = int((df_sent["__sent"] == "Negative").sum())

                # Contoh konten dari top & bottom campaign untuk arah rekomendasi
                def _sample_contents_for_campaign(cname, max_rows=10, max_examples=5):
                    sub = df_work[df_work[campaign_col].astype(str) == str(cname)].copy()
                    examples = []
                    for _, r in sub.head(max_rows).iterrows():
                        txt = _content_or_title(r, content_col, title_col, max_words=25)
                        if txt:
                            examples.append(txt)
                        if len(examples) >= max_examples:
                            break
                    return examples

                top_camp_examples = _sample_contents_for_campaign(top_campaign_name)
                bottom_camp_examples = _sample_contents_for_campaign(bottom_campaign_name)

                recomm_ctx = {
                    "current_period": current_date,
                    "top_campaign": top_campaign_name,
                    "needs_improvement_campaign": bottom_campaign_name,
                    "channel_distribution": channel_dist,
                    "negative_sentiment_count": negative_count,
                    "top_campaign_examples": top_camp_examples,
                    "bottom_campaign_examples": bottom_camp_examples,
                }

                if narrative_lang == "ID":
                    prompt = f"""
            Anda adalah strategic communications consultant untuk brand FMCG. Tulis rekomendasi strategis maksimum {max_words} kata.

            Gunakan struktur berikut, dalam bentuk paragraf dengan sub-judul yang jelas:

            1. CONTENT STRATEGY
            - Rekomendasikan 3–4 arah konten utama untuk campaign ini.
            - Kaitkan dengan pola konten yang muncul pada top campaign dan kebutuhan untuk mengangkat campaign yang lebih lemah.
            - Sentuh juga jenis konten yang paling cocok berdasarkan contoh isi (storytelling keluarga, edukasi nutrisi, promo, dsb).

            2. CHANNEL OPTIMIZATION
            - Jelaskan kanal mana yang perlu diprioritaskan, dipertahankan, atau dioptimalkan ulang berdasarkan distribusi channel.
            - Bedakan peran kanal awareness vs kanal yang paling dekat dengan konversi.

            3. SENTIMENT MANAGEMENT
            - Beri saran bagaimana mengelola dan menurunkan sentimen negatif.
            - Sertakan ide quick wins yang realistis untuk memperkuat konten positif dan meng-address keluhan utama.

            4. CAMPAIGN FOCUS & BUSINESS IMPACT
            - Jelaskan fokus campaign ke depan: apa yang harus diperkuat di top campaign dan apa yang perlu diperbaiki di campaign yang lemah.
            - Kaitkan dengan tujuan bisnis: konversi, repeat purchase, dan penguatan brand trust.

            Gunakan bahasa Indonesia yang to the point, actionable, dan mudah dipresentasikan ke klien. Jangan membuat bullet list teknis, cukup paragraf terpisah per sub-judul.

            Konteks data (untuk referensi analisis, jangan ditulis apa adanya):

            {json.dumps(recomm_ctx, ensure_ascii=False, indent=2)}
            """.strip()
                else:
                    prompt = f"""
            You are a strategic communications consultant for an FMCG brand. Write strategic recommendations in clear professional English, maximum {max_words} words.

            Follow this structure, in paragraphs with clear sub-headings:

            1. CONTENT STRATEGY
            - Recommend 3–4 key content directions for this campaign.
            - Link them to what is working in the top-performing campaign and what is missing in the weaker one.
            - Reflect the dominant content patterns in the examples (family stories, nutrition education, promotions, etc.).

            2. CHANNEL OPTIMIZATION
            - Suggest which channels should be prioritized, maintained, or rebalanced based on the channel distribution.
            - Differentiate between awareness channels and channels that are closer to conversion.

            3. SENTIMENT MANAGEMENT
            - Provide ideas to handle and reduce negative sentiment.
            - Include realistic quick wins to amplify positive narratives and address the main complaints.

            4. CAMPAIGN FOCUS & BUSINESS IMPACT
            - Explain where to focus the next wave of campaigns: how to strengthen the top campaign and repair the weaker one.
            - Tie the recommendations back to business goals such as conversion, repeat purchase, and brand trust.

            Write in paragraphs with sub-headings, no bullet list syntax. Make it easy to present directly to a client.

            Data context (for your reference only, do not copy verbatim):

            {json.dumps(recomm_ctx, ensure_ascii=False, indent=2)}
            """.strip()

            
            # Generate narrative
            narrative = generate_single_narrative(client, prompt, max_words=max_words)
            
            if narrative:
                st.session_state[key] = narrative
                success_count += 1
                logger.info(f"Successfully generated: {name}")
            else:
                failed.append(name)
                logger.error(f"Failed to generate: {name}")
            
            time.sleep(0.5)  # Small delay between requests
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        # Show summary
        if success_count == total:
            st.sidebar.success(f"✅ All {total} narratives generated!")
        else:
            st.sidebar.warning(f"⚠️ {success_count}/{total} narratives generated")
            if failed:
                st.sidebar.error(f"Failed: {', '.join(failed)}")
        
        st.rerun()

if st.sidebar.button("🗑️ Clear All Narratives", use_container_width=True):
    for k in ["narr_sov", "narr_trend_social", "narr_trend_main",
              "narr_sent_social", "narr_sent_main",
              "narr_channel_social", "narr_channel_main",
              "narr_topic_soc_pos", "narr_topic_soc_neu", "narr_topic_soc_neg",
              "narr_topic_main_pos", "narr_topic_main_neu", "narr_topic_main_neg",
              "narr_insights", "narr_recommendations"]:
        st.session_state[k] = ""
    st.sidebar.success("All narratives cleared!")
    st.rerun()

# Working dataframe
df_work = df[df[campaign_col].astype(str).isin(selected_campaigns_chart1)].copy()
if df_work.empty:
    df_work = df.copy()

df_work_norm = df_work.copy()
df_work_norm["__chan_norm"] = df_work_norm[channel_col].map(normalize_channel)

df_social_metrics = df_work_norm[df_work_norm["__chan_norm"].isin(SOCIAL_ORDER)]
df_main_metrics   = df_work_norm[df_work_norm["__chan_norm"].isin(MAINSTREAM_ORDER)]

eng_name  = pick_col(df, ["Engagement", "Total Engagement", "engagement","interactions","interaction"])
buzz_name = pick_col(df, ["Buzz","buzz","PR Value","pr value","pr_value","prvalue"])
pr_col    = pick_col(df, ["PR Value","pr value","pr_value","prvalue"])

# ============== Chart 1: SOV + Metrics ==============

tot_post_social = int(len(df_social_metrics))
tot_eng_social   = int(coerce_numeric(df_social_metrics[eng_name]).sum()) if eng_name and eng_name in df_social_metrics.columns else 0
tot_buzz_social  = int(coerce_numeric(df_social_metrics[buzz_name]).sum()) if buzz_name and buzz_name in df_social_metrics.columns else 0

st.markdown("### Data Social")
cS1, cS2, cS3 = st.columns(3)
with cS1: st.metric("Total Post", f"{tot_post_social:,}")
with cS2: st.metric("Total Engagement", f"{tot_eng_social:,}")
with cS3: st.metric("Total Buzz/PR Value", f"{tot_buzz_social:,}")

tot_art_main = int(len(df_main_metrics))
tot_pr_main  = int(coerce_numeric(df_main_metrics[pr_col]).sum()) if pr_col and pr_col in df_main_metrics.columns else 0

st.markdown("### Data Mainstream")
cM1, cM2 = st.columns(2)
with cM1: st.metric("Total Artikel", f"{tot_art_main:,}")
with cM2: st.metric("Total PR Value", f"{tot_pr_main:,}")

st.caption("Tabel Data")
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

tbl["SOV Post%"]       = (tbl["Total Post"]       / tot_post_sum * 100).round(2) if tot_post_sum else 0.0
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

st.subheader("Share of Voice — Comparison")

def _pie(df_in, title):
    if df_in.empty: return go.Figure()
    fig = px.pie(df_in, names="Campaign", values="Value", hole=0.0, title=title)
    fig.update_traces(
        texttemplate="%{label}<br>%{percent}", 
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>", 
        showlegend=False
    )
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
    return fig

top_n = 8
top_post = topn_generic(tbl.rename(columns={"Total Post":"Value"}),       "Campaign", "Value", top_n)
top_eng  = topn_generic(tbl.rename(columns={"Total Engagement":"Value"}), "Campaign", "Value", top_n)
top_buzz = topn_generic(tbl.rename(columns={"Total Buzz":"Value"}),       "Campaign", "Value", top_n)

c_p1, c_p2, c_p3 = st.columns(3)
with c_p1: st.plotly_chart(_pie(top_post, "SOV by Total Post"),       use_container_width=True)
with c_p2: st.plotly_chart(_pie(top_eng,   "SOV by Total Engagement"), use_container_width=True)
with c_p3: st.plotly_chart(_pie(top_buzz,  "SOV by Total Buzz"),       use_container_width=True)

# AI Narrative for SOV
if st.session_state["narr_sov"]:
    st.markdown("---")
    st.caption("🤖 AI Narrative - Share of Voice")
    st.markdown(st.session_state["narr_sov"])

# ============== Conversation Trends ==============
st.markdown("---")
st.subheader("Conversation Trends")

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
preferred_metric_names = ["engagement","likes","comments","shares","views","impressions","reach","er","pr value","pr_value","prvalue","buzz"]
for c in df.columns:
    if c not in numeric_cols and c.lower() in preferred_metric_names:
        numeric_cols.append(c)

if not numeric_cols:
    st.warning("Tidak ada kolom numerik terdeteksi untuk time-series.")
else:
    dates_parsed = pd.to_datetime(df_work[date_col], errors="coerce")
    date_min_default, date_max_default = dates_parsed.min(), dates_parsed.max()
    dmin, dmax = None, None
    
    if pd.notna(date_min_default) and pd.notna(date_max_default):
        chosen_range = st.date_input(
            "Filter Date Range",
            value=(date_min_default.date(), date_max_default.date()),
            min_value=date_min_default.date(),
            max_value=date_max_default.date()
        )
        dmin, dmax = (chosen_range[0], chosen_range[1]) if isinstance(chosen_range, tuple) else (date_min_default.date(), date_max_default.date())

    def _combo_ts(df_src, left_key, left_title, right_key, right_title, title):
        ts = build_timeseries(df_src, date_col, left_key, right_key, dmin, dmax)
        ts_vis = ts.copy().fillna(0)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        BAR_COLOR, LINE_COLOR = "#f2b01e", "#1f77b4"
        
        fig.add_bar(
            x=ts_vis["Date"], y=ts_vis["Left"], name=left_title,
            marker_color=BAR_COLOR, opacity=0.95,
            text=ts_vis["Left"], texttemplate="%{text:,.0f}",
            textposition="inside", insidetextanchor="start",
            textfont=dict(size=10), cliponaxis=False, secondary_y=False
        )
        fig.add_scatter(
            x=ts_vis["Date"], y=ts_vis["Right"], name=right_title,
            mode="lines+markers", line=dict(width=3, color=LINE_COLOR),
            marker=dict(size=6), cliponaxis=False, secondary_y=True
        )
        
        if len(ts_vis):
            q2 = float(np.nanpercentile(ts_vis["Left"], 50))
            q3 = float(np.nanpercentile(ts_vis["Left"], 75))
        else:
            q2 = q3 = 0.0
            
        for i, (xi, y_r, y_l) in enumerate(zip(ts_vis["Date"], ts_vis["Right"], ts_vis["Left"])):
            yshift, xshift = _calc_line_label_shift(y_l, i, q2, q3)
            fig.add_annotation(
                x=xi, y=y_r, yref="y2", text=f"{int(y_r):,}",
                showarrow=False, bgcolor="rgba(220,220,220,0.85)",
                font=dict(size=10), borderpad=3, yshift=yshift, xshift=xshift
            )
        
        fig.update_layout(
            title=title, height=420, bargap=0.35, hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        left_max  = float(np.nanmax(ts_vis["Left"]))  if len(ts_vis) else 0.0
        right_max = float(np.nanmax(ts_vis["Right"])) if len(ts_vis) else 0.0

        n_days = int(pd.Series(ts_vis["Date"]).nunique()) if len(ts_vis) else 0
        if   n_days <= 14: tick_size = 12
        elif n_days <= 21: tick_size = 10
        elif n_days <= 31: tick_size = 9
        elif n_days <= 45: tick_size = 8
        else:              tick_size = 7

        fig.update_xaxes(
            type="date", title_text="Date",
            tickmode="linear", dtick="D1", tickformat="%d-%b",
            ticklabelmode="instant", tickangle=0,
            tickfont=dict(size=tick_size), automargin=True,
            showgrid=True, gridcolor="rgba(0,0,0,0.08)"
        )

        fig.update_yaxes(
            title_text=left_title, secondary_y=False, showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            range=[0, left_max * 1.25 if left_max > 0 else 1], autorange=False
        )
        fig.update_yaxes(
            title_text=right_title, secondary_y=True, showgrid=False,
            range=[0, right_max * 1.25 if right_max > 0 else 1], autorange=False
        )

        return fig

    LEFT_COUNT_ALIAS = "Volume Post"
    left_choices_social  = [LEFT_COUNT_ALIAS] + numeric_cols
    right_choices_social = numeric_cols.copy()
    default_right_social = right_choices_social.index("Engagement") \
        if "Engagement" in right_choices_social else (
        right_choices_social.index("engagement") if "engagement" in right_choices_social else 0)

    c1, c2 = st.columns(2)
    with c1:
        left_metric_choice = st.selectbox("Social — Left Axis (Bar)", left_choices_social, index=0, key="ts_left")
    with c2:
        right_metric_choice = st.selectbox("Social — Right Axis (Line)", right_choices_social, index=default_right_social, key="ts_right")

    left_key_social  = "__count_post__" if left_metric_choice == LEFT_COUNT_ALIAS else left_metric_choice
    right_key_social = right_metric_choice

    df_social = df_work_norm[df_work_norm["__chan_norm"].isin([c for c in selected_social_channels if c in SOCIAL_ORDER])]
    st.plotly_chart(
        _combo_ts(df_social, left_key_social, left_metric_choice, right_key_social, right_metric_choice, "Social Conversation Trends"),
        use_container_width=True
    )
    
    if st.session_state["narr_trend_social"]:
        st.caption("🤖 AI Narrative - Social Trends")
        st.markdown(st.session_state["narr_trend_social"])

    LEFT_ALIAS_MS_TS = "Total Artikel"
    right_choices_ms = numeric_cols.copy()
    def_idx_pr = 0
    for nm in ["PR Value","pr value","pr_value","prvalue"]:
        if nm in right_choices_ms:
            def_idx_pr = right_choices_ms.index(nm)
            break

    d1, d2 = st.columns(2)
    with d1:
        left_metric_choice_ms = st.selectbox("Mainstream — Left Axis (Bar)", [LEFT_ALIAS_MS_TS] + numeric_cols, index=0, key="ts_ms_left")
    with d2:
        right_metric_choice_ms = st.selectbox("Mainstream — Right Axis (Line)", right_choices_ms, index=def_idx_pr, key="ts_ms_right")

    left_key_ms  = "__count_articles__" if left_metric_choice_ms == LEFT_ALIAS_MS_TS else left_metric_choice_ms
    right_key_ms = right_metric_choice_ms

    df_main = df_work_norm[df_work_norm["__chan_norm"].isin([c for c in selected_mainstream_channels if c in MAINSTREAM_ORDER])]
    st.plotly_chart(
        _combo_ts(df_main, left_key_ms, left_metric_choice_ms, right_key_ms, right_metric_choice_ms, "Mainstream Conversations Trends"),
        use_container_width=True
    )
    
    if st.session_state["narr_trend_main"]:
        st.caption("🤖 AI Narrative - Mainstream Trends")
        st.markdown(st.session_state["narr_trend_main"])

# ============== Sentiment Breakdown ==============
st.markdown("---")
st.subheader("Sentiment Breakdown")

c_sb1, c_sb2 = st.columns(2)

with c_sb1:
    st.markdown("<div style='font-weight:600; font-size:16px; margin:4px 0 6px;'>Data Social</div>", unsafe_allow_html=True)
    counts_social = sentiment_bar(df_social_metrics, "Social Media")
    
    # Sentiment Table for Social
    if counts_social is not None:
        st.caption("Distribusi Sentiment")
        sent_table_social = pd.DataFrame({
            "Sentimen": ["Positif", "Neutral", "Negatif"],
            "Jumlah Post": [
                int(counts_social.get("Positive", 0)),
                int(counts_social.get("Neutral", 0)),
                int(counts_social.get("Negative", 0))
            ]
        })
        st.dataframe(sent_table_social, use_container_width=True, hide_index=True)

with c_sb2:
    st.markdown("<div style='font-weight:600; font-size:16px; margin:4px 0 6px;'>Data Mainstream</div>", unsafe_allow_html=True)
    counts_main = sentiment_bar(df_main_metrics, "Mainstream Media")
    
    # Sentiment Table for Mainstream
    if counts_main is not None:
        st.caption("Distribusi Sentiment")
        sent_table_main = pd.DataFrame({
            "Sentimen": ["Positif", "Neutral", "Negatif"],
            "Jumlah Artikel": [
                int(counts_main.get("Positive", 0)),
                int(counts_main.get("Neutral", 0)),
                int(counts_main.get("Negative", 0))
            ]
        })
        st.dataframe(sent_table_main, use_container_width=True, hide_index=True)

# AI Narratives for Sentiment (below both charts)
if st.session_state["narr_sent_social"] or st.session_state["narr_sent_main"]:
    cs1, cs2 = st.columns(2)
    with cs1:
        if st.session_state["narr_sent_social"]:
            st.caption("🤖 AI Narrative - Sentiment Social")
            st.markdown(st.session_state["narr_sent_social"])
    with cs2:
        if st.session_state["narr_sent_main"]:
            st.caption("🤖 AI Narrative - Sentiment Mainstream")
            st.markdown(st.session_state["narr_sent_main"])

# ============== Channel Breakdowns ==============
st.markdown("---")
st.subheader("Social Media Channel Breakdown")

numeric_cols_ch = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
pref_names_ch = {"engagement","total engagement","interaction","interactions","views","impressions","reach","pr value","pr_value","prvalue"}
for c in df.columns:
    if c not in numeric_cols_ch and c.lower() in pref_names_ch:
        numeric_cols_ch.append(c)

LEFT_ALIAS_SOC = "Volume Post"
left_choices4  = [LEFT_ALIAS_SOC] + numeric_cols_ch
right_choices4 = numeric_cols_ch
right_default4 = right_choices4.index("Engagement") if "Engagement" in right_choices4 else (
    right_choices4.index("engagement") if "engagement" in right_choices4 else 0
)

CHART4_ORDER = ["Instagram", "Tiktok", "Twitter", "Facebook", "YouTube", "Forum"]
channels_for_chart = [c for c in CHART4_ORDER if c in selected_social_channels]

df_social_section = df_work_norm[df_work_norm["__chan_norm"].isin(channels_for_chart)].copy()

post_counts = (
    df_social_section["__chan_norm"].value_counts()
    .reindex(channels_for_chart, fill_value=0)
    .reset_index()
)
post_counts.columns = ["Channel", "Value"]

if eng_name and eng_name in df_social_section.columns:
    _vals_eng = coerce_numeric(df_social_section[eng_name])
    eng_sum = (
        df_social_section.assign(_v=_vals_eng)
        .groupby("__chan_norm")["_v"].sum(min_count=1)
        .reindex(channels_for_chart, fill_value=0)
        .reset_index()
    )
    eng_sum.columns = ["Channel", "Value"]
else:
    eng_sum = pd.DataFrame(columns=["Channel", "Value"])

def _pie_channels(df_in, title):
    if df_in.empty or (pd.to_numeric(df_in["Value"], errors="coerce").fillna(0).sum() <= 0):
        fig = go.Figure()
        fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
        return fig
    fig = px.pie(df_in, names="Channel", values="Value", hole=0.0, title=title)
    fig.update_traces(
        texttemplate="%{label}<br>%{percent}",
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        showlegend=False
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=380)
    return fig

pie_l, pie_r = st.columns(2)
with pie_l:
    st.plotly_chart(_pie_channels(post_counts, "SOV by Post (Channel)"), use_container_width=True)
with pie_r:
    if not eng_sum.empty:
        st.plotly_chart(_pie_channels(eng_sum, "SOV by Engagement (Channel)"), use_container_width=True)
    else:
        st.info("Kolom Engagement tidak ditemukan untuk pie.")

# Channel Table for Social
st.caption("Distribusi Channel Social Media")
channel_colors = {
    "Instagram": "🟣",
    "Tiktok": "⚫",
    "Twitter": "🔵",
    "Facebook": "🔵",
    "YouTube": "🔴",
    "Forum": "🟢"
}

channel_table_data = []
for ch in channels_for_chart:
    post_val = int(post_counts[post_counts["Channel"] == ch]["Value"].iloc[0]) if len(post_counts[post_counts["Channel"] == ch]) > 0 else 0
    eng_val = 0
    if not eng_sum.empty and len(eng_sum[eng_sum["Channel"] == ch]) > 0:
        eng_val = int(eng_sum[eng_sum["Channel"] == ch]["Value"].iloc[0])
    
    channel_table_data.append({
        "Channel": f"{channel_colors.get(ch, '•')} {ch}",
        "Volume Post": post_val,
        "Engagement": eng_val
    })

channel_table_social = pd.DataFrame(channel_table_data)
st.dataframe(channel_table_social, use_container_width=True, hide_index=True)

c41, c42 = st.columns(2)
with c41:
    left_choice4  = st.selectbox("Left Axis (Bar)", left_choices4, index=0, key="ch4_left")
with c42:
    right_choice4 = st.selectbox("Right Axis (Line)", right_choices4, index=right_default4, key="ch4_right")

left_key4   = "__count_post__" if left_choice4 == LEFT_ALIAS_SOC else left_choice4
left_title4 = LEFT_ALIAS_SOC if left_choice4 == LEFT_ALIAS_SOC else left_choice4
right_key4  = right_choice4
right_title4= right_choice4

fig_social_combo, _ = build_channel_combo_v2(
    df_in=df_social_section,
    channel_list=channels_for_chart,
    channel_col=channel_col,
    left_key=left_key4, left_title=left_title4,
    right_key=right_key4, right_title=right_title4,
    title="Social Media Channels"
)
st.plotly_chart(fig_social_combo, use_container_width=True)

if st.session_state["narr_channel_social"]:
    st.caption("🤖 AI Narrative - Social Channels")
    st.markdown(st.session_state["narr_channel_social"])

# Mainstream
st.subheader("Mainstream Media Channel Breakdown")

numeric_cols_ch2 = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
pref_names_ch2 = {"engagement","interaction","interactions","views","impressions","reach",
                  "pr value","pr_value","prvalue","ad value","buzz"}
for c in df.columns:
    if c not in numeric_cols_ch2 and c.lower() in pref_names_ch2:
        numeric_cols_ch2.append(c)

LEFT_ALIAS_MS = "Volume Artikel"
left_choices5  = [LEFT_ALIAS_MS] + numeric_cols_ch2
right_choices5 = numeric_cols_ch2

def_idx_pr = 0
for nm in ["PR Value","pr value","pr_value","prvalue","ad value"]:
    if nm in right_choices5:
        def_idx_pr = right_choices5.index(nm)
        break

channels_for_main = [c for c in MAINSTREAM_ORDER if c in selected_mainstream_channels]
df_main_section = df_work_norm[df_work_norm["__chan_norm"].isin(channels_for_main)].copy()

art_counts = (
    df_main_section["__chan_norm"].value_counts()
      .reindex(channels_for_main, fill_value=0)
      .reset_index()
)
art_counts.columns = ["Channel", "Value"]

title_value_pie = "SOV by PR Value (Channel)"
if pr_col and pr_col in df_main_section.columns:
    _vals_val = coerce_numeric(df_main_section[pr_col])
elif buzz_name and buzz_name in df_main_section.columns:
    _vals_val = coerce_numeric(df_main_section[buzz_name])
    title_value_pie = "SOV by Buzz/Value (Channel)"
else:
    fb = pick_col(df, ["Engagement","Total Engagement","engagement","interactions","interaction"])
    if fb and fb in df_main_section.columns:
        _vals_val = coerce_numeric(df_main_section[fb])
        title_value_pie = f"SOV by {fb} (Channel)"
    else:
        _vals_val = None

if _vals_val is not None:
    val_sum_main = (
        df_main_section.assign(_v=_vals_val)
          .groupby("__chan_norm")["_v"].sum(min_count=1)
          .reindex(channels_for_main, fill_value=0)
          .reset_index()
    )
    val_sum_main.columns = ["Channel","Value"]
else:
    val_sum_main = pd.DataFrame(columns=["Channel","Value"])

pie_l, pie_r = st.columns(2)
with pie_l:
    st.plotly_chart(_pie_channels(art_counts, "SOV by Articles (Channel)"), use_container_width=True)
with pie_r:
    if not val_sum_main.empty:
        st.plotly_chart(_pie_channels(val_sum_main, title_value_pie), use_container_width=True)
    else:
        st.info("Nilai PR/Buzz tidak ditemukan untuk pie.")

# Channel Table for Mainstream
st.caption("Distribusi Channel Mainstream Media")
channel_colors_main = {
    "Online Media": "🔵",
    "Printed": "📰",
    "TV": "📺"
}

channel_table_data_main = []
for ch in channels_for_main:
    art_val = int(art_counts[art_counts["Channel"] == ch]["Value"].iloc[0]) if len(art_counts[art_counts["Channel"] == ch]) > 0 else 0
    pr_val = 0
    if not val_sum_main.empty and len(val_sum_main[val_sum_main["Channel"] == ch]) > 0:
        pr_val = int(val_sum_main[val_sum_main["Channel"] == ch]["Value"].iloc[0])
    
    channel_table_data_main.append({
        "Channel": f"{channel_colors_main.get(ch, '•')} {ch}",
        "Artikel": art_val,
        "PR Value": pr_val
    })

channel_table_main = pd.DataFrame(channel_table_data_main)
st.dataframe(channel_table_main, use_container_width=True, hide_index=True)

d51, d52 = st.columns(2)
with d51:
    left_choice5  = st.selectbox("Left Axis (Bar)", left_choices5, index=0, key="ch5_left")
with d52:
    right_choice5 = st.selectbox("Right Axis (Line)", right_choices5, index=def_idx_pr, key="ch5_right")

left_key5   = "__count_articles__" if left_choice5 == LEFT_ALIAS_MS else left_choice5
left_title5 = LEFT_ALIAS_MS if left_choice5 == LEFT_ALIAS_MS else left_choice5
right_key5  = right_choice5
right_title5= right_choice5

fig_main_combo, _ = build_channel_combo_v2(
    df_in=df_main_section,
    channel_list=channels_for_main,
    channel_col=channel_col,
    left_key=left_key5, left_title=left_title5,
    right_key=right_key5, right_title=right_title5,
    title="Mainstream Media Channels"
)
st.plotly_chart(fig_main_combo, use_container_width=True)

if st.session_state["narr_channel_main"]:
    st.caption("🤖 AI Narrative - Mainstream Channels")
    st.markdown(st.session_state["narr_channel_main"])

# ============== TOP 10 MEDIA (MAINSTREAM) ==============
st.markdown("---")
st.subheader("Top 10 Media (Mainstream)")

if media_col and media_col in df_main_metrics.columns:
    media_counts = df_main_metrics.groupby(media_col).size().sort_values(ascending=False)
    top10_media = media_counts.head(10)
    
    if not top10_media.empty:
        # Chart
        fig_media = go.Figure()
        fig_media.add_bar(
            y=top10_media.index,
            x=top10_media.values,
            orientation='h',
            marker_color='#1f77b4',
            text=top10_media.values,
            texttemplate='%{text}',
            textposition='outside'
        )
        fig_media.update_layout(
            title="Top 10 Media by Article Count",
            height=400,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Jumlah Artikel"),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_media, use_container_width=True)
        
        # Table with topics
        st.caption("Detail Media")
        
        table_data = []
        for media in top10_media.index:
            count = int(top10_media[media])
            
            media_df = df_main_metrics[df_main_metrics[media_col] == media]
            
            # Get top 3 topics for this media
            if topic_col and topic_col in media_df.columns:
                top_topics = media_df[topic_col].value_counts().head(3).index.tolist()
                topics_str = ", ".join([str(t) for t in top_topics])
            else:
                topics_str = "-"
            
            table_data.append({
                "MEDIA": media,
                "JUMLAH": count,
                "TOPIC": topics_str
            })
        
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada data media untuk ditampilkan.")
else:
    st.info("Kolom Media tidak ditemukan untuk mainstream data.")

# ============== Topic Charts ==============
st.markdown("---")
st.subheader("Topic Charts")

topic_col = pick_col(df, ["Topic","topic","Topik"])
sent_col  = pick_col(df, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix",
                          "sentiment level","new sentiment level","new_sentiment_level"])

if not topic_col or topic_col not in df_work.columns:
    st.info("Data Not Found (kolom 'Topic' tidak tersedia).")
elif not sent_col:
    st.info("Kolom Sentiment tidak ditemukan — Topic charts tidak bisa dibuat.")
else:
    tmp_topics = df_work_norm.copy()
    tmp_topics["__sent"] = tmp_topics[sent_col].map(normalize_sentiment)

    total_topics = int(tmp_topics[topic_col].dropna().nunique()) if topic_col in tmp_topics.columns else 0
    slider_max   = min(20, total_topics or 20)
    default_n    = min(10, slider_max)

    topic_topn = st.slider(
        "Top N Topic (0 = semua)",
        min_value=0, max_value=slider_max, value=default_n, step=1,
        help="Geser ke 0 untuk menampilkan semua topic di semua chart."
    )

    ENG_COL  = pick_col(df, ["Engagement", "Total Engagement", "engagement","interactions","interaction"])
    BUZZ_COL = pick_col(df, ["Buzz","buzz","PR Value","pr value","pr_value","prvalue"])

    def _render_topic_grid(df_subset, section_title, top_n):
        st.markdown(f"### {section_title}")
        
        is_social = section_title == "Social"
        
        for sent_label, color_hex in [("Positive", "#43A047"),
                                    ("Neutral",  "#BDBDBD"),
                                    ("Negative", "#E53935")]:
            st.markdown(f"**{sent_label}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                _topic_bar_plot(
                    df_subset[df_subset["__sent"]==sent_label],
                    topic_col, f"{sent_label} Isu by Post", color_hex,
                    metric_kind="post", top_n=top_n
                )
            with c2:
                if ENG_COL:
                    _topic_bar_plot(
                        df_subset[df_subset["__sent"]==sent_label],
                        topic_col, f"{sent_label} Isu by Engagement", color_hex,
                        metric_kind="eng", value_col=ENG_COL, top_n=top_n
                    )
                else:
                    st.info("Data Not Found (Engagement)")
            with c3:
                if BUZZ_COL:
                    _topic_bar_plot(
                        df_subset[df_subset["__sent"]==sent_label],
                        topic_col, f"{sent_label} Isu by Buzz", color_hex,
                        metric_kind="buzz", value_col=BUZZ_COL, top_n=top_n
                    )
                else:
                    st.info("Data Not Found (Buzz)")
            
            # AI Narrative for this sentiment
            if is_social:
                key_map = {"Positive": "narr_topic_soc_pos", "Neutral": "narr_topic_soc_neu", "Negative": "narr_topic_soc_neg"}
            else:
                key_map = {"Positive": "narr_topic_main_pos", "Neutral": "narr_topic_main_neu", "Negative": "narr_topic_main_neg"}
            
            narr_key = key_map[sent_label]
            if st.session_state[narr_key]:
                st.caption(f"🤖 AI Narrative - {sent_label}")
                st.markdown(st.session_state[narr_key])

    df_topic_social = tmp_topics[tmp_topics["__chan_norm"].isin(selected_social_channels)]
    _render_topic_grid(df_topic_social, "Social", top_n=topic_topn)

    df_topic_main = tmp_topics[tmp_topics["__chan_norm"].isin(selected_mainstream_channels)]
    _render_topic_grid(df_topic_main, "Mainstream", top_n=topic_topn)

# ============== INSIGHTS & RECOMMENDATIONS ==============
st.markdown("---")
st.subheader("📊 Insights")
if st.session_state["narr_insights"]:
    st.markdown(st.session_state["narr_insights"])
else:
    st.info("_Generate insights using the button in the sidebar._")

st.markdown("---")
st.subheader("💡 Recommendations")
if st.session_state["narr_recommendations"]:
    st.markdown(st.session_state["narr_recommendations"])
else:
    st.info("_Generate recommendations using the button in the sidebar._")

logger.info("Application rendering completed")

#by katon