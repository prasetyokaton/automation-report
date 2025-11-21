# app.py
# Dependencies: streamlit, pandas, plotly, openpyxl
# Run: streamlit run app.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os, json, re
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI


st.set_page_config(page_title="SOV & Time Series Combo", layout="wide")
st.title("📈 Automation Report Insights")

load_dotenv(dotenv_path=".secretcontainer/.env", override=True)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

if OPENAI_MODEL.strip().startswith("os.getenv"):
    st.error("OPENAI_MODEL di .env berisi ekspresi, bukan nama model. Contoh yang benar: OPENAI_MODEL=gpt-4.1-mini")


BANNED_CHARS = [";", ":", "—", "–", "≈"]
# catatan: kita juga hilangkan pattern " - " secara hati-hati saat post-processing


# ==== Helpers for narrative (add below _get_openai_client) ====
def _word_trim(s: str, max_words=200) -> str:
    if not s: return s
    w = s.split()
    return " ".join(w[:max_words])

def _sanitize_output(s: str) -> str:
    if not s: return s
    for ch in BANNED_CHARS:
        s = s.replace(ch, " ")
    s = s.replace(" - ", " ")        # hapus dash di tengah kalimat
    s = s.replace("Others", "selebihnya").replace("others", "selebihnya")
    # rapihkan spasi ganda
    return " ".join(s.split())

def _sov_share_lookup(tbl_df, name: str, post=True):
    # ambil share % yang sudah dihitung di tabel 'tbl'
    col = "SOV Post%" if post else "SOV Engagement%"
    try:
        row = tbl_df.loc[tbl_df["Campaign"].astype(str)==str(name), col]
        if len(row): 
            return float(row.iloc[0])
    except:
        pass
    return None

def _top_topics_from_column(df_src, campaign_name, topic_col, eng_col=None, top_k_post=3, top_k_eng=2):
    sub = df_src[df_src[campaign_col].astype(str)==str(campaign_name)].copy()
    out = {"topics_by_post": [], "topics_by_engagement": []}
    if topic_col and topic_col in sub.columns:
        # by post
        top_post_topics = (sub.groupby(topic_col).size()
                           .sort_values(ascending=False).head(top_k_post).index.astype(str).tolist())
        out["topics_by_post"] = top_post_topics
        # by engagement
        if eng_col and eng_col in sub.columns:
            vals = coerce_numeric(sub[eng_col])
            top_eng_topics = (sub.assign(_v=vals).groupby(topic_col)["_v"]
                                .sum(min_count=1).sort_values(ascending=False)
                                .head(top_k_eng).index.astype(str).tolist())
            out["topics_by_engagement"] = top_eng_topics
    return out

def _content_examples(df_src, campaign_name, title_col, content_col, eng_col, n_post=5):
    sub = df_src[df_src[campaign_col].astype(str)==str(campaign_name)].copy()
    if eng_col and eng_col in sub.columns:
        sub["_eng"] = coerce_numeric(sub[eng_col])
        sub = sub.sort_values("_eng", ascending=False)
    out = []
    for _, r in sub.head(n_post).iterrows():
        tt = (str(r[title_col]).strip() if title_col and title_col in sub.columns and pd.notna(r[title_col]) else "")
        cc = (str(r[content_col]).strip() if content_col and content_col in sub.columns and pd.notna(r[content_col]) else "")
        out.append({"title": _shorten_words(tt, 18), "caption": _shorten_words(cc, 28)})
    return out


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    # Ambil digit pertama yang ketemu; fallback ke default kalau tak ada
    m = re.search(r"\d+", str(raw))
    try:
        return int(m.group(0)) if m else int(default)
    except Exception:
        return int(default)

OPENAI_MAX_OUT = _env_int("OPENAI_MAX_OUT", 700)


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
    if s in {"printed","print","koran","majalah","newspaper","tabloid"}: return "Printed"
    if s in {"tv","television","televisi"}: return "TV"
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
    
    # hitung headroom utk range normal
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


def _get_openai_client():
    try:
        # SDK akan otomatis baca OPENAI_API_KEY dari .env
        return OpenAI()
    except Exception as e:
        st.error(f"Gagal init OpenAI client: {e}")
        return None


def _shorten_words(s, max_words=30):
    if pd.isna(s): return ""
    w = str(s).split()
    return " ".join(w[:max_words])

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


# ==== Helpers khusus narasi berbasis Content/Title & tanggal ====
_ID_MONTHS = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]

def _fmt_date_id_short(x):
    if pd.isna(x): return ""
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts): return ""
    return f"{ts.day:02d} " + _ID_MONTHS[int(ts.month)-1]

def _clean_text_basic(s, max_words=40):
    if pd.isna(s) or not str(s).strip(): return ""
    t = str(s)
    t = re.sub(r"http\S+|www\.\S+", " ", t)   # buang URL
    t = re.sub(r"@\w+", " ", t)               # buang mention
    t = t.replace("#", " ")                   # hilangkan '#'
    t = re.sub(r"\s+", " ", t).strip()
    return _shorten_words(t, max_words=max_words)

def _content_or_title(row, content_col, title_col, max_words=40):
    c = (str(row[content_col]).strip() if content_col and content_col in row and pd.notna(row[content_col]) else "")
    t = (str(row[title_col]).strip()   if title_col   and title_col   in row and pd.notna(row[title_col])   else "")
    txt = c if c else t
    return _clean_text_basic(txt, max_words=max_words)

def _finalize_paragraph(s, max_words=200):
    s = _word_trim(s, max_words=max_words)
    # pastikan tidak "nanggung": potong ke titik terakhir jika perlu
    if not s.endswith((".", "!", "?")):
        last = max(s.rfind("."), s.rfind("!"), s.rfind("?"))
        if last >= 50:  # kasih toleransi biar gak kependekan banget
            s = s[:last+1]
    return _sanitize_output(s)

def _top_dates_by_sum(df_src, date_col, value_series, top_k=3):
    if df_src.empty: return []
    work = df_src.copy()
    work["__date"] = pd.to_datetime(work[date_col], errors="coerce").dt.date
    work = work.dropna(subset=["__date"])
    agg = pd.DataFrame({"__date": work["__date"]})
    agg["_val"] = value_series.loc[work.index].values
    byday = agg.groupby("__date")["_val"].sum(min_count=1).sort_values(ascending=False)
    out = []
    for d, v in byday.head(top_k).items():
        out.append({"date": d, "value": float(v or 0)})
    return out

def _top_dates_by_count(df_src, date_col, top_k=2):
    if df_src.empty: return []
    work = df_src.copy()
    work["__date"] = pd.to_datetime(work[date_col], errors="coerce").dt.date
    work = work.dropna(subset=["__date"])
    byday = work.groupby("__date").size().sort_values(ascending=False)
    return [{"date": d, "count": int(c)} for d, c in byday.head(top_k).items()]

def _top_posts_social(df_src, eng_col, content_col, title_col, n=2):
    if df_src.empty or not eng_col or eng_col not in df_src.columns:
        return []
    vals = coerce_numeric(df_src[eng_col])
    tmp = df_src.assign(_eng=vals)
    # skip baris tanpa content & title
    keep = tmp.apply(lambda r: bool(_content_or_title(r, content_col, title_col, 40)), axis=1)
    tmp = tmp[keep].sort_values("_eng", ascending=False)
    out = []
    for _, r in tmp.head(n).iterrows():
        out.append({
            "date": pd.to_datetime(r[date_col], errors="coerce"),
            "campaign": str(r[campaign_col]) if campaign_col in tmp.columns else "",
            "preview": _content_or_title(r, content_col, title_col, 30),
            "eng": float(r["_eng"]) if pd.notna(r["_eng"]) else 0.0
        })
    return out

def _sample_previews_on_date(df_src, date_col, on_date, sort_key=None, content_col=None, title_col=None, k=2):
    if df_src.empty: return []
    w = df_src.copy()
    w["__date"] = pd.to_datetime(w[date_col], errors="coerce").dt.date
    w = w[w["__date"] == on_date]
    if sort_key and sort_key in w.columns:
        w = w.sort_values(sort_key, ascending=False)
    out = []
    for _, r in w.head(k).iterrows():
        txt = _content_or_title(r, content_col, title_col, 28)
        if txt:
            out.append({
                "campaign": str(r[campaign_col]) if campaign_col in w.columns else "",
                "preview": txt
            })
    return out



def responses_create_safe(client, **kwargs):
    try:
        return client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e)
        # auto-strip params yang sering tidak didukung model kecil
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




def _preview_text(x, max_words=40):
    if pd.isna(x): return ""
    words = str(x).strip().split()
    return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")


def _trim_complete_paragraph(s: str, max_words=200) -> str:
    """
    Pangkas ke <= max_words, sanitize, lalu pastikan berakhir di akhir kalimat.
    Jika tidak ada tanda akhir kalimat, kembalikan apa adanya (sudah dipangkas).
    """
    if not s:
        return s
    s = _sanitize_output(_word_trim(s, max_words))
    if not s:
        return s
    if s[-1] in ".!?":
        return s
    # cari akhir kalimat terakhir
    last_dot = max(s.rfind("."), s.rfind("!"), s.rfind("?"))
    if last_dot != -1:
        return s[: last_dot + 1]
    return s  # fallback: biarkan apa adanya

def _examples_for_date(df_src, date_col, date_value, title_col, content_col, rank_col=None, n=2):
    """
    Ambil contoh konten untuk sebuah tanggal. Urutkan by rank_col desc jika ada,
    else by tanggal terbaru.
    """
    work = df_src.copy()
    work["_dt_date"] = pd.to_datetime(work[date_col], errors="coerce").dt.date
    work = work[work["_dt_date"] == date_value]
    if rank_col and rank_col in work.columns:
        vals = coerce_numeric(work[rank_col])
        work = work.assign(_rank=vals).sort_values("_rank", ascending=False)
    else:
        t = pd.to_datetime(work[date_col], errors="coerce")
        work = work.assign(_rank=t).sort_values("_rank", ascending=False)

    out = []
    for _, r in work.head(n).iterrows():
        out.append({
            "title": _shorten_words(str(r.get(title_col, "")), 20),
            "content_preview": _preview_text(r.get(content_col, ""), 30)
        })
    return out


def _examples_for_campaign(df_src, campaign_name, title_col, content_col, eng_col, date_col, n=2, by_eng=False):
    work = df_src[df_src[campaign_col].astype(str) == str(campaign_name)].copy()
    if by_eng and eng_col and eng_col in work.columns:
        work["_eng"] = coerce_numeric(work[eng_col])
        work = work.sort_values("_eng", ascending=False)
    elif date_col and date_col in work.columns:
        work["_dt"] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values("_dt", ascending=False)
    out = []
    for _, r in work.head(n).iterrows():
        out.append({
            "title": (str(r[title_col]).strip() if (title_col and title_col in work.columns) else "")[:160],
            "content_preview": _preview_text(r[content_col]) if (content_col and content_col in work.columns) else "",
            "engagement": int(float(r[eng_col])) if (eng_col and eng_col in work.columns and pd.notna(r[eng_col])) else 0,
            "date": (pd.to_datetime(r[date_col], errors="coerce").strftime("%Y-%m-%d")
                     if (date_col and date_col in work.columns and pd.notna(r.get(date_col))) else "")
        })
    return out


# ==== Helpers for Conversation Trends Narrative ====
def _format_date_id(d):
    try:
        return pd.to_datetime(d).strftime("%d-%b-%Y")
    except Exception:
        return str(d)

def _subset_by_date_range(df_in, date_col, dmin=None, dmax=None):
    work = df_in.copy()
    work["__date"] = pd.to_datetime(work[date_col], errors="coerce").dt.date
    work = work.dropna(subset=["__date"])
    if dmin: work = work[work["__date"] >= dmin]
    if dmax: work = work[work["__date"] <= dmax]
    return work

def _top_dates_metrics(df_in, engage_col, k_eng=3, k_post=2):
    """Kembalikan dua DF: top_eng (__date, eng) dan top_post (__date, post)."""
    if "__date" not in df_in.columns:
        raise ValueError("Kolom __date belum ada. Jalankan _subset_by_date_range dulu.")
    # post per tanggal
    g_post = df_in.groupby("__date").size().rename("post").reset_index()
    top_post = g_post.sort_values(["post", "__date"], ascending=[False, False]).head(k_post)

    # engagement per tanggal (fallback bila eng_col tidak ada)
    if engage_col and engage_col in df_in.columns:
        vals = coerce_numeric(df_in[engage_col])
        g_eng = df_in.assign(_v=vals).groupby("__date")["_v"].sum(min_count=1).rename("eng").reset_index()
        top_eng = g_eng.sort_values(["eng", "__date"], ascending=[False, False]).head(k_eng)
    else:
        top_eng = pd.DataFrame(columns=["__date","eng"])

    return top_eng, top_post

def _context_for_dates(df_in, date_list, topic_col, title_col, content_col, eng_col, examples_per_date=2):
    """Bangun konteks per tanggal: topik by post dan by engagement + contoh konten singkat."""
    ctx = []
    for d in date_list:
        sub = df_in[df_in["__date"] == d].copy()
        topics_by_post, topics_by_eng = [], []

        if topic_col and topic_col in sub.columns:
            topics_by_post = (sub.groupby(topic_col).size()
                                .sort_values(ascending=False).head(3).index.astype(str).tolist())
            if eng_col and eng_col in sub.columns:
                vals = coerce_numeric(sub[eng_col])
                topics_by_eng = (sub.assign(_v=vals).groupby(topic_col)["_v"].sum(min_count=1)
                                   .sort_values(ascending=False).head(2).index.astype(str).tolist())

        # contoh konten tertinggi berdasar engagement (jika ada), kalau tidak pakai urutan asli
        if eng_col and eng_col in sub.columns:
            sub = sub.assign(_eng=coerce_numeric(sub[eng_col])).sort_values("_eng", ascending=False)

        examples = []
        for _, r in sub.head(examples_per_date).iterrows():
            tt = (str(r[title_col]).strip() if title_col and title_col in sub.columns and pd.notna(r[title_col]) else "")
            cc = (str(r[content_col]).strip() if content_col and content_col in sub.columns and pd.notna(r[content_col]) else "")
            examples.append({"title": _shorten_words(tt, 18), "caption": _shorten_words(cc, 28)})

        ctx.append({
            "date": _format_date_id(d),
            "topics_by_post": topics_by_post,
            "topics_by_engagement": topics_by_eng,
            "examples": examples
        })
    return ctx

def _pick_metric_for_eng_peak(df_subset, eng_name, pr_col, buzz_name):
    """Pilih kolom metrik untuk puncak engagement. Urutan: Engagement → PR Value → Buzz → None."""
    if eng_name and eng_name in df_subset.columns:
        return eng_name
    if pr_col and pr_col in df_subset.columns:
        return pr_col
    if buzz_name and buzz_name in df_subset.columns:
        return buzz_name
    return None

def _build_trends_prompt(ctx_top_eng, ctx_top_post):
    return f"""
Anda adalah analis insight senior. Tulis satu paragraf maksimum 200 kata Bahasa Indonesia yang santai namun profesional.
Hindari karakter titik dua, titik koma, dash, dan simbol aneh. Jangan buat bullet. Jadikan cerita mengalir.

Tugas:
1) Soroti tiga puncak engagement berdasarkan tanggal dan jelaskan ringkas apa yang terjadi pada tanggal itu memakai tema dari kolom Topic atau simpulkan dari contoh judul atau caption.
2) Sebutkan dua puncak volume post berdasarkan tanggal dan jelaskan tema singkatnya.
3) Tutup dengan satu kalimat takeaway arah percakapan pada periode yang difilter.

Data:
Top engagement: {json.dumps(ctx_top_eng, ensure_ascii=False)}
Top post: {json.dumps(ctx_top_post, ensure_ascii=False)}
""".strip()



def _pie_context_json_rich(tbl_df, top_post_df, top_eng_df,
                           df_src, campaign_col, title_col, content_col, eng_col, date_col,
                           max_campaigns=3, examples_per_campaign=2):
    tp = top_post_df.head(max_campaigns)["Campaign"].astype(str).tolist() if "Campaign" in top_post_df.columns else []
    te = top_eng_df.head(max_campaigns)["Campaign"].astype(str).tolist()  if "Campaign" in top_eng_df.columns else []
    uniq_camps = list(dict.fromkeys(tp + te))  # urutan dipertahankan

    cols = ["Campaign","Total Post","SOV Post%","Total Engagement","SOV Engagement%"]
    keep = [c for c in cols if c in tbl_df.columns]
    table_preview = tbl_df[keep].head(100).to_dict(orient="records")

    examples = {camp: {
        "post_examples": _examples_for_campaign(df_src, camp, title_col, content_col, eng_col, date_col,
                                                 n=examples_per_campaign, by_eng=False),
        "engagement_examples": _examples_for_campaign(df_src, camp, title_col, content_col, eng_col, date_col,
                                                      n=examples_per_campaign, by_eng=True),
    } for camp in uniq_camps}

    ctx = {
        "table_preview": table_preview,
        "top_post": top_post_df.head(max_campaigns).to_dict("records"),
        "top_engagement": top_eng_df.head(max_campaigns).to_dict("records"),
        "examples_by_campaign": examples,
        "notes": "Fokus pada Post & Engagement. Gunakan Title/Content untuk menjelaskan tema & pendorong engagement."
    }
    return json.dumps(ctx, ensure_ascii=False)


def _is_reasoning_model(name: str) -> bool:
    s = (name or "").lower()
    return s.startswith("gpt-5") or "-r" in s


# ============== Sidebar: Upload & Global Controls ==============

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


# ==== Session Narrative State (persist until refresh) ====
ds_sig = f"{getattr(uploaded, 'name', 'uploaded')}::{sheet_name}"
if "dataset_sig" not in st.session_state or st.session_state["dataset_sig"] != ds_sig:
    # dataset berubah → reset narasi
    st.session_state["dataset_sig"] = ds_sig
    st.session_state["narr_chart1"] = ""
    st.session_state["narr_trend_social"] = ""
    st.session_state["narr_trend_main"] = ""
else:
    # init jika belum ada
    for k in ("narr_chart1", "narr_trend_social", "narr_trend_main"):
        if k not in st.session_state:
            st.session_state[k] = ""


# auto-detect columns (tanpa UI)
campaign_col = find_campaign_column(df.columns) or df.columns[0]
date_col     = find_date_column(df) or df.columns[0]
channel_col  = pick_col(df, ["Channel","Channels","Platform","Media","Source","Tipe","Type"]) or df.columns[0]
# ⬇️ deteksi Title/Content untuk narasi
title_col   = pick_col(df, ["Title", "Judul", "Post Title", "title"])
content_col = pick_col(df, ["Content", "Konten", "Caption", "Text", "Isi", "content", "caption", "text"])


# ============== Sidebar: Campaign + Channel Filters ==============

# Campaign multiselect di SIDEBAR
campaign_values_all = sorted(df[campaign_col].astype(str).dropna().unique().tolist())
selected_campaigns_chart1 = st.sidebar.multiselect(
    "Filter Campaign", options=campaign_values_all, default=campaign_values_all, key="campaign_filter_chart1"
)
if not selected_campaigns_chart1:
    selected_campaigns_chart1 = campaign_values_all

# Social/Mainstream channel options (dipakai SEMUA chart)
st.sidebar.markdown("---")
st.sidebar.subheader("Chart 4 — Social Media Channels")
SOCIAL_ORDER = ["Instagram", "Tiktok", "Twitter", "Facebook", "YouTube", "Forum"]
selected_social_channels = st.sidebar.multiselect(
    "Pilih channel Social", options=SOCIAL_ORDER, default=SOCIAL_ORDER, key="social_channels_sel"
)

st.sidebar.subheader("Chart 5 — Mainstream Channels")
MAINSTREAM_ORDER = ["Online Media", "Printed", "TV"]
selected_mainstream_channels = st.sidebar.multiselect(
    "Pilih channel Mainstream", options=MAINSTREAM_ORDER, default=MAINSTREAM_ORDER, key="mainstream_channels_sel"
)

# Tombol RESET di PALING BAWAH sidebar
def _reset_all():
    st.session_state["campaign_filter_chart1"]  = campaign_values_all
    st.session_state["social_channels_sel"]     = SOCIAL_ORDER
    st.session_state["mainstream_channels_sel"] = MAINSTREAM_ORDER

    # cari kandidat numerik utk default right axis
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


st.sidebar.markdown("---")
st.sidebar.button("🔄 Reset Semua Filter", use_container_width=True, on_click=_reset_all)

# ============== Data kerja berdasarkan filter campaign ==============
df_work = df[df[campaign_col].astype(str).isin(selected_campaigns_chart1)].copy()
if df_work.empty:
    df_work = df.copy()

# Deteksi kolom metrik
eng_name  = pick_col(df, ["Engagement", "Total Engagement", "engagement", "interactions", "interaction"])
buzz_name = pick_col(df, ["Buzz", "buzz", "PR Value","pr value","pr_value","prvalue"])  # Buzz atau PR Value jika ada

# ============== Chart 1: SOV + Metrics + 3 Pie ==============
sov_work, total_post_work = prep_sov(df_work, campaign_col)
# ==== DATA SOCIAL & MAINSTREAM ====
df_work_norm = df_work.copy()
df_work_norm["__chan_norm"] = df_work_norm[channel_col].map(normalize_channel)

SOCIAL_ORDER = ["Instagram", "Tiktok", "Twitter", "Facebook", "YouTube", "Forum"]
MAINSTREAM_ORDER = ["Online Media","Printed","TV"]

df_social_metrics = df_work_norm[df_work_norm["__chan_norm"].isin(SOCIAL_ORDER)]
df_main_metrics   = df_work_norm[df_work_norm["__chan_norm"].isin(MAINSTREAM_ORDER)]

eng_name  = pick_col(df, ["Engagement","Total Engagement","engagement","interactions","interaction"])
pr_col    = pick_col(df, ["PR Value","pr value","pr_value","prvalue"])
buzz_name = pick_col(df, ["Buzz","buzz","PR Value","pr value","pr_value","prvalue"])

# Data Social
tot_post_social = int(len(df_social_metrics))
tot_eng_social   = int(coerce_numeric(df_social_metrics[eng_name]).sum()) if eng_name and eng_name in df_social_metrics.columns else 0
tot_buzz_social  = int(coerce_numeric(df_social_metrics[buzz_name]).sum()) if buzz_name and buzz_name in df_social_metrics.columns else 0

st.markdown("### Data Social")
cS1, cS2, cS3 = st.columns(3)
with cS1: st.metric("Total Post", f"{tot_post_social:,}")
with cS2: st.metric("Total Engagement", f"{tot_eng_social:,}")
with cS3: st.metric("Total Buzz/PR Value", f"{tot_buzz_social:,}")

# Data Mainstream
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

st.subheader("Share of Voice — Comparison")
def _pie(df_in, title):
    if df_in.empty: return go.Figure()
    fig = px.pie(df_in, names="Campaign", values="Value", hole=0.0, title=title)
    fig.update_traces(texttemplate="%{label}<br>%{percent}", textinfo="label+percent",
                      hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>", showlegend=False)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
    return fig

top_n = 8
top_post = topn_generic(tbl.rename(columns={"Total Post":"Value"}),       "Campaign", "Value", top_n)
top_eng   = topn_generic(tbl.rename(columns={"Total Engagement":"Value"}), "Campaign", "Value", top_n)
top_buzz  = topn_generic(tbl.rename(columns={"Total Buzz":"Value"}),       "Campaign", "Value", top_n)

c_p1, c_p2, c_p3 = st.columns(3)
with c_p1: st.plotly_chart(_pie(top_post, "SOV by Total Post"),       use_container_width=True)
with c_p2: st.plotly_chart(_pie(top_eng,   "SOV by Total Engagement"), use_container_width=True)
with c_p3: st.plotly_chart(_pie(top_buzz,  "SOV by Total Buzz"),       use_container_width=True)



# === AI Narrative — Chart 1 (Post & Engagement, ≤200 kata) ===
st.markdown("---")
st.subheader("AI Narrative — Chart 1 (Post & Engagement)")
st.caption("Satu paragraf, ≤200 kata, gaya santai profesional. Tanpa titik dua, titik koma, dash, atau simbol aneh. Fokus ke ringkasan isi konten (Content/Title).")

# Tampilkan narasi yang tersimpan (persist)
st.markdown(st.session_state["narr_chart1"] or "_Belum ada narasi. Klik Generate._")

c_gen, c_clear = st.columns([1,1])
with c_gen:
    if st.button("✨ Generate Narrative (≤200 kata)", key="btn_ai_pie"):
        with st.spinner("Menghasilkan narasi…"):
            client = _get_openai_client()
            if not client:
                st.stop()

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

            if not keep_names:
                st.warning("Tidak ada kampanye untuk diringkas.")
                st.stop()

            ctx = []
            for name in keep_names:
                sub = df_work[df_work[campaign_col].astype(str) == str(name)].copy()
                eg = []
                if eng_name and eng_name in sub.columns:
                    sub["_eng"] = coerce_numeric(sub[eng_name])
                    for _, r in sub.sort_values("_eng", ascending=False).head(3).iterrows():
                        p = _content_or_title(r, content_col, title_col, 28)
                        if p: eg.append(p)
                if not eg:
                    sub["_dt"] = pd.to_datetime(sub[date_col], errors="coerce")
                    for _, r in sub.sort_values("_dt", ascending=False).head(3).iterrows():
                        p = _content_or_title(r, content_col, title_col, 28)
                        if p: eg.append(p)

                ctx.append({
                    "campaign": name,
                    "post_share_pct": _sov_share_lookup(tbl, name, post=True),
                    "eng_share_pct":  _sov_share_lookup(tbl, name, post=False),
                    "content_examples": eg[:3]
                })

            prompt = f"""
Anda adalah analis insight senior. Tulis satu paragraf maksimum 190 kata, Bahasa Indonesia santai namun profesional.
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

            try:
                req = dict(model=OPENAI_MODEL, input=prompt, max_output_tokens=max(380, OPENAI_MAX_OUT), temperature=0.2)
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

                narrative = _finalize_paragraph(narrative_raw, max_words=190)
                if narrative:
                    st.session_state["narr_chart1"] = narrative
                    st.success("Narrative diperbarui.")
                    st.markdown(narrative)
                else:
                    st.error("Model tidak mengembalikan teks.")
            except Exception as e:
                st.error(f"Gagal menghasilkan narasi: {e}")

with c_clear:
    if st.button("🗑️ Clear", key="btn_clear_pie"):
        st.session_state["narr_chart1"] = ""
        st.info("Narrative dikosongkan.")





# ============== Conversation Trends (Time Series) ==============
st.markdown("---")
st.subheader("Conversation Trends")

# Kandidat metrik TS
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
preferred_metric_names = ["engagement","likes","comments","shares","views","impressions","reach","er","pr value","pr_value","prvalue","buzz"]
for c in df.columns:
    if c not in numeric_cols and c.lower() in preferred_metric_names:
        numeric_cols.append(c)
# Guard: kalau tak ada kolom numerik untuk sumbu kanan (line)
if not numeric_cols:
    st.warning("Tidak ada kolom numerik terdeteksi untuk time-series (sumbu kanan). Bagian ini akan dilewati.")

# Date range berdasar df_work
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

# Helper plot — AUTOSCALE by default
def _combo_ts(df_src, left_key, left_title, right_key, right_title, title):
    ts = build_timeseries(df_src, date_col, left_key, right_key, dmin, dmax)
    ts_vis = ts.copy().fillna(0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    BAR_COLOR, LINE_COLOR = "#f2b01e", "#1f77b4"
    fig.add_bar(x=ts_vis["Date"], y=ts_vis["Left"], name=left_title,
                marker_color=BAR_COLOR, opacity=0.95,
                text=ts_vis["Left"], texttemplate="%{text:,.0f}",
                textposition="inside", insidetextanchor="start",
                textfont=dict(size=10), cliponaxis=False, secondary_y=False)
    fig.add_scatter(x=ts_vis["Date"], y=ts_vis["Right"], name=right_title,
                    mode="lines+markers", line=dict(width=3, color=LINE_COLOR),
                    marker=dict(size=6), cliponaxis=False, secondary_y=True)
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
    fig.update_layout(title=title, height=420, bargap=0.35, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    # headroom agar label line tidak ketiban
    # hitung headroom utk range normal (ambil dari ts_vis, bukan agg)
    left_max  = float(np.nanmax(ts_vis["Left"]))  if len(ts_vis) else 0.0
    right_max = float(np.nanmax(ts_vis["Right"])) if len(ts_vis) else 0.0

    # tampilkan SEMUA tanggal (harian) + auto shrink tick font
    n_days = int(pd.Series(ts_vis["Date"]).nunique()) if len(ts_vis) else 0
    if   n_days <= 14: tick_size = 12
    elif n_days <= 21: tick_size = 10
    elif n_days <= 31: tick_size = 9
    elif n_days <= 45: tick_size = 8
    else:              tick_size = 7

    fig.update_xaxes(
        type="date",
        title_text="Date",
        tickmode="linear",     # paksa linear ticks
        dtick="D1",            # setiap 1 hari
        tickformat="%d-%b",
        ticklabelmode="instant",
        tickangle=0,
        tickfont=dict(size=tick_size),
        automargin=True,
        showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )

    # NON-AUTOSCALE (normal), start from 0
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

# NORMALISASI channel + filter sesuai pilihan sidebar
df_work_norm = df_work.copy()
df_work_norm["__chan_norm"] = df_work_norm[channel_col].map(normalize_channel)

# === GUARD + Controls & charts (Conversation Trends) ===
if numeric_cols:
    # === Controls & chart SOCIAL TS ===
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


    # === AI Narrative — Social Trends (≤200 kata, santai profesional, berbasis Content) ===
    st.caption("AI Narrative — Social Trends")
    st.markdown(st.session_state["narr_trend_social"] or "_Belum ada narasi. Klik Generate._")

    colGs, colCs = st.columns([1,1])
    with colGs:
        if st.button("✨ Generate Narrative (≤200 kata)", key="btn_ai_trend_social"):
            with st.spinner("Menghasilkan narasi…"):
                client = _get_openai_client()
                if not client:
                    st.stop()
                if not eng_name or eng_name not in df_social.columns:
                    st.warning("Kolom Engagement tidak ditemukan untuk Social Trends.")
                    st.stop()

                vals_eng = coerce_numeric(df_social[eng_name])
                top3_eng_dates = _top_dates_by_sum(df_social, date_col, vals_eng, top_k=3)
                top2_post_dates = _top_dates_by_count(df_social, date_col, top_k=2)
                top2_posts = _top_posts_social(df_social, eng_col=eng_name, content_col=content_col, title_col=title_col, n=2)

                peaks_detail = []
                for d in top3_eng_dates:
                    samples = _sample_previews_on_date(
                        df_social, date_col, d["date"], sort_key=eng_name,
                        content_col=content_col, title_col=title_col, k=2
                    )
                    peaks_detail.append({
                        "date_label": _fmt_date_id_short(d["date"]),
                        "sum_engagement": d["value"],
                        "examples": samples
                    })

                ctx = {
                    "peaks_by_engagement": peaks_detail,
                    "peaks_by_post": [{"date_label": _fmt_date_id_short(x["date"]), "count": x["count"]} for x in top2_post_dates],
                    "top_posts": [{
                        "date_label": _fmt_date_id_short(p["date"]),
                        "campaign": p["campaign"],
                        "preview": p["preview"]
                    } for p in top2_posts]
                }

                prompt = f"""
    Anda adalah analis insight senior. Tulis satu paragraf maksimum 190 kata, gaya santai namun profesional.
    Fokus pada isi konten dari tanggal puncak. Jangan gunakan tahun dalam penulisan tanggal, cukup format DD MMM.
    Hindari titik dua, titik koma, dash, dan simbol aneh. Jangan buat bullet. Abaikan baris yang kosong konten dan judul.

    Tugas:
    1) Jelaskan tiga puncak engagement harian dengan menyebut tanggal versi singkat dan merangkum isi konten/konteks pemicu interaksi.
    2) Sebut dua puncak volume post dan benang merah kontennya.
    3) Sentuh singkat dua post teratas lintas periode bila relevan.
    4) Tutup dengan satu kalimat takeaway yang menyatukan arah percakapan.

    Data:
    {json.dumps(ctx, ensure_ascii=False)}
    """.strip()

                try:
                    req = dict(model=OPENAI_MODEL, input=prompt, max_output_tokens=max(380, OPENAI_MAX_OUT), temperature=0.2)
                    if _is_reasoning_model(OPENAI_MODEL):
                        req["reasoning"] = {"effort": "low"}
                    resp = responses_create_safe(client, **req)
                    out = _safe_output_text(resp)
                    out = _finalize_paragraph(out, max_words=190)
                    if out:
                        st.session_state["narr_trend_social"] = out
                        st.success("Narrative diperbarui.")
                        st.markdown(out)
                    else:
                        st.error("Model tidak mengembalikan teks.")
                except Exception as e:
                    st.error(f"Gagal menghasilkan narasi: {e}")
    with colCs:
        if st.button("🗑️ Clear", key="btn_clear_trend_social"):
            st.session_state["narr_trend_social"] = ""
            st.info("Narrative dikosongkan.")




    # === Controls & chart MAINSTREAM TS ===
    LEFT_ALIAS_MS_TS = "Total Artikel"
    right_choices_ms = numeric_cols.copy()
    def_idx_pr = 0
    for nm in ["PR Value","pr value","pr_value","prvalue"]:
        if nm in right_choices_ms:
            def_idx_pr = right_choices_ms.index(nm); break

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

    # === AI Narrative — Mainstream Trends (≤200 kata, santai profesional, TANPA engagement) ===
    st.caption("AI Narrative — Mainstream Trends")
    st.markdown(st.session_state["narr_trend_main"] or "_Belum ada narasi. Klik Generate._")

    colGm, colCm = st.columns([1,1])
    with colGm:
        if st.button("✨ Generate Narrative (≤200 kata)", key="btn_ai_trend_main"):
            with st.spinner("Menghasilkan narasi…"):
                client = _get_openai_client()
                if not client:
                    st.stop()

                top3_art_dates = _top_dates_by_count(df_main, date_col, top_k=3)
                top2_art_dates = _top_dates_by_count(df_main, date_col, top_k=2)

                value_col = None
                for nm in ["PR Value","pr value","pr_value","prvalue","Buzz","buzz"]:
                    if nm in df_main.columns:
                        value_col = nm; break

                peaks_detail = []
                for d in top3_art_dates:
                    sort_key = value_col if value_col else date_col
                    samples = _sample_previews_on_date(
                        df_main, date_col, d["date"], sort_key=sort_key,
                        content_col=content_col, title_col=title_col, k=2
                    )
                    peaks_detail.append({
                        "date_label": _fmt_date_id_short(d["date"]),
                        "count": d["count"] if "count" in d else None,
                        "examples": samples
                    })

                ctx = {
                    "peaks_by_articles": peaks_detail,
                    "peaks_volume": [{"date_label": _fmt_date_id_short(x["date"]), "count": x["count"]} for x in top2_art_dates],
                    "note": "Mainstream tidak menggunakan metrik engagement. Fokus pada jumlah artikel dan isi konten."
                }

                prompt = f"""
Anda adalah analis insight senior. Tulis satu paragraf maksimum 190 kata, gaya santai namun profesional.
Fokus pada jumlah artikel dan isi konten. Jangan gunakan tahun pada tanggal, cukup DD MMM.
Hindari titik dua, titik koma, dash, dan simbol aneh. Jangan buat bullet.

Tugas:
1) Jelaskan tiga puncak artikel harian dengan menyebut tanggal singkat dan ringkas isi konten yang mendominasi.
2) Sebut dua puncak volume artikel dan tema yang membuat pemberitaan ramai.
3) Tutup dengan satu kalimat takeaway yang merangkum arah percakapan periode ini.

Data:
{json.dumps(ctx, ensure_ascii=False)}
""".strip()

                try:
                    req = dict(model=OPENAI_MODEL, input=prompt, max_output_tokens=max(380, OPENAI_MAX_OUT), temperature=0.2)
                    if _is_reasoning_model(OPENAI_MODEL):
                        req["reasoning"] = {"effort": "low"}
                    resp = responses_create_safe(client, **req)
                    out = _safe_output_text(resp)
                    out = _finalize_paragraph(out, max_words=190)
                    if out:
                        st.session_state["narr_trend_main"] = out
                        st.success("Narrative diperbarui.")
                        st.markdown(out)
                    else:
                        st.error("Model tidak mengembalikan teks.")
                except Exception as e:
                    st.error(f"Gagal menghasilkan narasi: {e}")
    with colCm:
        if st.button("🗑️ Clear", key="btn_clear_trend_main"):
            st.session_state["narr_trend_main"] = ""
            st.info("Narrative dikosongkan.")



# ============== Sentiment Breakdown (Social & Mainstream) ==============
st.markdown("---")
st.subheader("Sentiment Breakdown")

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
    if df_src.empty:
        st.info(f"Tidak ada data untuk {title}."); return
    sent_col = pick_col(df, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix","sentiment level","new sentiment level","new_sentiment_level"]) or df.columns[0]
    if not sent_col:
        st.info(f"Kolom Sentiment tidak ditemukan untuk {title}."); return
    work = df_src.copy()
    work["__sent"] = work[sent_col].map(normalize_sentiment)
    order  = ["Positive", "Neutral", "Negative"]
    colors = {"Positive": "#43A047", "Neutral": "#BDBDBD", "Negative": "#E53935"}
    labels_bg = {"Positive": "rgba(67,160,71,0.85)", "Neutral": "rgba(189,189,189,0.85)", "Negative": "rgba(229,57,53,0.85)"}
    counts = work["__sent"].value_counts().reindex(order, fill_value=0)
    total  = int(counts.sum())
    if total == 0:
        st.info(f"Tidak ada data sentiment pada {title}."); return
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

c_sb1, c_sb2 = st.columns(2)

with c_sb1:
    st.markdown("<div style='font-weight:600; font-size:16px; margin:4px 0 6px;'>Data Social</div>", unsafe_allow_html=True)
    sentiment_bar(df_social_metrics, "Social Media")

with c_sb2:
    st.markdown("<div style='font-weight:600; font-size:16px; margin:4px 0 6px;'>Data Mainstream</div>", unsafe_allow_html=True)
    sentiment_bar(df_main_metrics, "Mainstream Media")



# ============== Chart 4: Social Media Channel Breakdown ==============
st.markdown("---")
st.subheader("Social Media Channel Breakdown")

# 1) Siapkan kandidat metrik numerik untuk opsi sumbu kanan (line)
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

# 2) ===== PIE SOV per Channel (DITAMPILKAN DI ATAS FILTER) =====
# Urutan display channel sesuai permintaan
CHART4_ORDER = ["Instagram", "Tiktok", "Twitter", "Facebook", "YouTube", "Forum"]
channels_for_chart = [c for c in CHART4_ORDER if c in selected_social_channels]

# Subset data social dengan urutan channel paksa
df_social_section = df_work_norm[df_work_norm["__chan_norm"].isin(channels_for_chart)].copy()

# Data untuk pie SOV by Post
post_counts = (
    df_social_section["__chan_norm"].value_counts()
    .reindex(channels_for_chart, fill_value=0)
    .reset_index()
)
post_counts.columns = ["Channel", "Value"]

# Data untuk pie SOV by Engagement
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


# TAMPILKAN 2 PIE DI ATAS FILTER (lebih lebar supaya label tidak kepotong)
pie_l, pie_r = st.columns(2)
with pie_l:
    st.plotly_chart(_pie_channels(post_counts, "SOV by Post (Channel)"), use_container_width=True)
with pie_r:
    if not eng_sum.empty:
        st.plotly_chart(_pie_channels(eng_sum, "SOV by Engagement (Channel)"), use_container_width=True)
    else:
        st.info("Kolom Engagement tidak ditemukan untuk pie.")

# 3) ===== FILTER & COMBO CHART (di BAWAH pie) =====
c41, c42 = st.columns(2)
with c41:
    left_choice4  = st.selectbox("Left Axis (Bar)", left_choices4, index=0, key="ch4_left")
with c42:
    right_choice4 = st.selectbox("Right Axis (Line)", right_choices4, index=right_default4, key="ch4_right")

left_key4   = "__count_post__" if left_choice4 == LEFT_ALIAS_SOC else left_choice4
left_title4 = LEFT_ALIAS_SOC if left_choice4 == LEFT_ALIAS_SOC else left_choice4
right_key4  = right_choice4
right_title4= right_choice4

# Build combo chart dengan urutan channel paksa
fig_social_combo, _ = build_channel_combo_v2(
    df_in=df_social_section,
    channel_list=channels_for_chart,
    channel_col=channel_col,
    left_key=left_key4, left_title=left_title4,
    right_key=right_key4, right_title=right_title4,
    title="Social Media Channels"
)
st.plotly_chart(fig_social_combo, use_container_width=True)



# ============== Chart 5: Mainstream Media Channel Breakdown ==============
st.subheader("Mainstream Media Channel Breakdown")

# 1) Siapkan kandidat metrik numerik (untuk sumbu kanan / line)
numeric_cols_ch2 = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
pref_names_ch2 = {"engagement","interaction","interactions","views","impressions","reach",
                  "pr value","pr_value","prvalue","ad value","buzz"}
for c in df.columns:
    if c not in numeric_cols_ch2 and c.lower() in pref_names_ch2:
        numeric_cols_ch2.append(c)

LEFT_ALIAS_MS = "Volume Artikel"
left_choices5  = [LEFT_ALIAS_MS] + numeric_cols_ch2
right_choices5 = numeric_cols_ch2

# default kanan: PR Value bila ada
def_idx_pr = 0
for nm in ["PR Value","pr value","pr_value","prvalue","ad value"]:
    if nm in right_choices5:
        def_idx_pr = right_choices5.index(nm); break

# ---- PIE SOV per Channel (DITAMPILKAN DI ATAS FILTER) ----
# urutan display + subset data
channels_for_main = [c for c in MAINSTREAM_ORDER if c in selected_mainstream_channels]
df_main_section = df_work_norm[df_work_norm["__chan_norm"].isin(channels_for_main)].copy()

# SOV by Articles (jumlah artikel per channel)
art_counts = (
    df_main_section["__chan_norm"].value_counts()
      .reindex(channels_for_main, fill_value=0)
      .reset_index()
)
art_counts.columns = ["Channel", "Value"]

# SOV by PR Value (atau fallback lain jika tidak ada)
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

# gunakan helper pie yang sama; sudah di-set textposition="outside" sebelumnya
pie_l, pie_r = st.columns(2)
with pie_l:
    st.plotly_chart(_pie_channels(art_counts, "SOV by Articles (Channel)"), use_container_width=True)
with pie_r:
    if not val_sum_main.empty:
        st.plotly_chart(_pie_channels(val_sum_main, title_value_pie), use_container_width=True)
    else:
        st.info("Nilai PR/Buzz tidak ditemukan untuk pie.")

# ---- FILTER & COMBO CHART (di BAWAH pie) ----
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


# ============== Topic Charts (Social & Mainstream; Pos/Neu/Neg × Post/Eng/Buzz) ==============
st.markdown("---")
st.subheader("Topic Charts")

# helper: angka jadi string “1.234”
def _fmt_int(v):
    try: return f"{int(round(float(v))):,}"
    except: return str(v)

# helper: bar horizontal dengan annotation putih + border warna sentimen
def _topic_bar_plot(df_src, topic_col, title, color_hex, metric_kind, value_col=None, top_n=None):
    if topic_col not in df_src.columns or df_src.empty:
        st.info("Data Not Found"); return

    if metric_kind == "post":
        agg = df_src.groupby(topic_col).size().rename("Value").reset_index()
    else:
        vals = coerce_numeric(df_src[value_col]) if value_col and value_col in df_src.columns else None
        if vals is None: st.info("Data Not Found"); return
        agg = df_src.assign(_v=vals).groupby(topic_col)["_v"].sum(min_count=1).rename("Value").reset_index()

    # sort desc, batasi jika top_n>0
    agg = agg.sort_values("Value", ascending=False)
    if top_n and top_n > 0:
        agg = agg.head(top_n)
    if agg.empty: 
        st.info("Data Not Found"); return


    # bar
    fig = go.Figure()
    fig.add_bar(
        x=agg["Value"], 
        y=agg[topic_col],
        orientation="h", marker_color=color_hex, text=None
    )


    # annotation angka: font putih + border warna sentimen (biar kebaca)
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



# deteksi kolom Topic & Sentiment
topic_col = pick_col(df, ["Topic","topic","Topik"])
sent_col  = pick_col(df, ["Sentiment","New Sentiment","new_sentiment","Sentiment Fix",
                          "sentiment level","new sentiment level","new_sentiment_level"])

if not topic_col or topic_col not in df_work.columns:
    st.info("Data Not Found (kolom 'Topic' tidak tersedia).")
elif not sent_col:
    st.info("Kolom Sentiment tidak ditemukan — Topic charts tidak bisa dibuat.")
else:
    # normalisasi sentiment
    tmp_topics = df_work_norm.copy()
    tmp_topics["__sent"] = tmp_topics[sent_col].map(normalize_sentiment)


    # --- Slider Top N Topic (0 = semua) ---
    total_topics = int(tmp_topics[topic_col].dropna().nunique()) if topic_col in tmp_topics.columns else 0
    slider_max   = min(20, total_topics or 20)           # batas atas 20
    default_n    = min(10, slider_max)                   # default 10

    topic_topn = st.slider(
        "Top N Topic (0 = semua)",
        min_value=0, max_value=slider_max, value=default_n, step=1,
        help="Geser ke 0 untuk menampilkan semua topic di semua chart."
    )
    # --- end slider ---



    # siapkan metric columns
    ENG_COL  = pick_col(df, ["Engagement", "Total Engagement", "engagement","interactions","interaction"])
    BUZZ_COL = pick_col(df, ["Buzz","buzz","PR Value","pr value","pr_value","prvalue"])

    def _render_topic_grid(df_subset, section_title, top_n):
        st.markdown(f"### {section_title}")
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



    # ===== Social only =====
    df_topic_social = tmp_topics[tmp_topics["__chan_norm"].isin(selected_social_channels)]
    _render_topic_grid(df_topic_social, "Social", top_n=topic_topn)

    # ===== Mainstream only =====
    df_topic_main = tmp_topics[tmp_topics["__chan_norm"].isin(selected_mainstream_channels)]
    _render_topic_grid(df_topic_main, "Mainstream", top_n=topic_topn)

