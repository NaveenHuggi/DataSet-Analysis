"""
AI Data Scientist & ML Mentor
─────────────────────────────────────────────────────────────────────────────
Refactored to include:
  1. .env-based API key persistence (enter once, auto-loaded on restart)
  2. ydata-profiling integration — inline EDA charts + downloadable HTML report
  3. In-app Code Sandbox (exec-based) with df, pd, np, plt, sns, px pre-injected
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys
import contextlib
import traceback
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv, set_key
from groq import Groq

# ── Model Routing ───────────────────────────────────────────────────────────
ANALYSIS_MODEL = "llama-3.3-70b-versatile"   # heavy analytical model
CHAT_MODEL     = "llama-3.1-8b-instant"       # fast conversational model

# ── .env Path (same folder as this script) ─────────────────────────────────
ENV_FILE = Path(__file__).parent / ".env"

# ── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Automated AI Data Scientist and Machine Learning Mentor. Your purpose is to analyze dataset profiles and guide the user step-by-step through preprocessing and model training.

You operate in two distinct phases:

### Phase 1: The Initial Analysis
When the user first uploads or pastes a dataset profile, you must immediately provide a structured report following exactly these four sections:
1. **Dataset Summary**: Explain in 2-3 simple, jargon-free sentences what this dataset represents and its likely real-world domain.
2. **Data Health Check**: Highlight immediate issues the user needs to fix before training (e.g., missing data percentages, high correlations, or skewed distributions).
3. **Problem Type**: State clearly whether this is a Classification, Regression, Clustering, or Time-Series problem.
4. **Recommended Models**: Recommend 2-3 specific Machine Learning or Deep Learning models best suited for this data. Briefly justify each choice.

At the end of Phase 1, ask the user: "Which of these models would you like to build first? Let me know, and we will start prepping the code!"

### Phase 2: The Interactive Walkthrough
Once the user selects a path, you transition into a highly conversational, step-by-step mentor.
Follow these strict rules for Phase 2:
- **Pacing**: Never output a massive, complete training script all at once. Break the process down into logical chunks (e.g., Step 1: Handling Nulls, Step 2: Scaling, Step 3: Model Architecture).
- **Code Quality**: Provide clean, well-commented Python code (using Pandas, Scikit-Learn, or PyTorch) only for the immediate next step.
- **Verification**: Always end your response by asking the user if the code ran successfully or if they need clarification before moving to the next step."""

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Scientist Mentor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark base */
    .stApp { background: #0f0f1a; color: #e2e8f0; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1a2e;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid #2d2d44;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 8px 20px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }

    /* Card containers */
    .card {
        background: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #1a1a2e 100%);
        border: 1px solid #3730a3;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h2 { color: #a5b4fc; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #6b7280; font-size: 0.85rem; margin: 0.3rem 0 0; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a5b4fc;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2d2d44;
    }

    /* Sandbox code area */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        background: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    }

    /* Chat messages */
    .stChatMessage { background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 12px; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a2e !important;
        border-radius: 8px !important;
        color: #a5b4fc !important;
    }

    /* Success / Info */
    .stSuccess { background: #052e16 !important; border-color: #166534 !important; }
    .stInfo    { background: #0c1a3a !important; border-color: #1e3a8a !important; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load .env & auto-login ──────────────────────────────────────────────────
load_dotenv(ENV_FILE)
_env_key = os.getenv("GROQ_API_KEY", "").strip()

# ── Session State Init ──────────────────────────────────────────────────────
defaults = {
    "api_key_set":     False,
    "client":          None,
    "messages":        [],
    "phase":           1,
    "dataset_profile": None,
    "df":              None,
    "profile_html":    None,
    "analysis_done":   False,
    "sandbox_output":  "",
    "sandbox_figs":    [],
    "sandbox_code":    "# df is preloaded from your uploaded dataset\nprint(df.shape)\ndf.head()",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auto-login if key found in .env ────────────────────────────────────────
if _env_key and not st.session_state.api_key_set:
    try:
        _c = Groq(api_key=_env_key)
        _c.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        st.session_state.client      = _c
        st.session_state.api_key_set = True
    except Exception:
        pass  # fall through to setup screen

# ═══════════════════════════════════════════════════════════════════════════
# SETUP SCREEN  (shown only once — key saved to .env after validation)
# ═══════════════════════════════════════════════════════════════════════════
if not st.session_state.api_key_set:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none; }
        .setup-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 8vh;
        }
        .setup-card {
            background: linear-gradient(145deg, #1e1b4b, #1a1a2e);
            border: 1px solid #3730a3;
            border-radius: 1.4rem;
            padding: 3rem;
            width: 100%;
            max-width: 520px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 80px rgba(99,102,241,0.08);
        }
        .setup-card h1 { font-size: 2.2rem; margin-bottom: 0.2rem; }
        .setup-card .sub {
            color: #9ca3af; font-size: 0.95rem; margin-bottom: 0.6rem;
        }
        .setup-badge {
            display: inline-flex; align-items: center; gap: 6px;
            background: #052e16; color: #4ade80;
            border: 1px solid #166534; border-radius: 999px;
            font-size: 0.78rem; padding: 4px 12px; margin-bottom: 1.6rem;
        }
        </style>
        <div class="setup-wrapper">
          <div class="setup-card">
            <h1>🤖 AI Data Scientist</h1>
            <p class="sub">Your personal ML mentor powered by Groq + Llama 3.</p>
            <div class="setup-badge">🔒 Key saved once — never asked again</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, col_form, _ = st.columns([1, 2, 1])
    with col_form:
        st.markdown("<br>", unsafe_allow_html=True)
        key_input = st.text_input(
            "Groq API Key", type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
        )
        if st.button("Get Started →", use_container_width=True, type="primary"):
            if key_input.strip():
                with st.spinner("Validating key…"):
                    try:
                        test_client = Groq(api_key=key_input.strip())
                        test_client.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=[{"role": "user", "content": "hi"}],
                            max_tokens=5,
                        )
                        # ── Persist to .env ──────────────────────────────
                        ENV_FILE.touch(exist_ok=True)
                        set_key(str(ENV_FILE), "GROQ_API_KEY", key_input.strip())
                        # ── Set session ──────────────────────────────────
                        st.session_state.client      = test_client
                        st.session_state.api_key_set = True
                        st.success(" Key saved to .env — you won't be asked again!")
                        st.rerun()
                    except Exception as e:
                        st.error(f" Invalid API key or connection error: {e}")
            else:
                st.warning("Please enter your Groq API key.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP  (API key confirmed)
# ═══════════════════════════════════════════════════════════════════════════
client = st.session_state.client

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
        <span style="font-size:2.4rem;">🤖</span>
        <div>
            <h1 style="margin:0;font-size:1.9rem;background:linear-gradient(135deg,#a5b4fc,#c4b5fd);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                AI Data Scientist &amp; ML Mentor
            </h1>
            <p style="margin:0;color:#6b7280;font-size:0.9rem;">
                Upload a CSV · Get instant EDA · Chat with your mentor · Run code live
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 3-Tab Layout ────────────────────────────────────────────────────────────
tab_eda, tab_chat, tab_sandbox = st.tabs(
    [" Dataset Analysis", " AI Mentor Chat", " Code Sandbox"]
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def generate_llm_profile(df: pd.DataFrame) -> str:
    """Compact text summary for the LLM context window."""
    buf = io.StringIO()
    df.info(buf=buf)
    missing = df.isnull().sum()
    missing_str = (
        missing[missing > 0].to_string()
        if missing.sum() > 0 else "No missing values."
    )
    out  = f"--- Dataset Info ---\n{buf.getvalue()}\n\n"
    out += f"--- Missing Values ---\n{missing_str}\n\n"
    out += f"--- Statistical Summary ---\n{df.describe(include='all').to_string()}\n\n"
    out += f"--- Sample Data (First 5 rows) ---\n{df.head().to_string()}\n\n"
    return out


def _pandas_html_report(df: pd.DataFrame) -> str:
    """Pure-pandas fallback profiling report — no extra dependencies needed."""
    import html as _html

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(exclude=np.number).columns.tolist()
    missing      = df.isnull().sum()
    missing_pct  = (missing / len(df) * 100).round(2)

    def _row(label, value, highlight=False):
        bg = "#1e1b4b" if highlight else "#1a1a2e"
        return f'<tr style="background:{bg}"><td style="padding:6px 12px;color:#9ca3af;width:220px">{_html.escape(str(label))}</td><td style="padding:6px 12px;color:#e2e8f0">{_html.escape(str(value))}</td></tr>'

    def _section(title):
        return f'<h2 style="color:#a5b4fc;border-bottom:1px solid #3730a3;padding-bottom:8px;margin-top:2rem">{title}</h2>'

    _m_total = missing.sum()
    _m_raw   = _m_total / (len(df) * len(df.columns)) * 100
    _m_pct   = (
        f"{_m_raw:.1f}%"  if _m_raw >= 0.1
        else "<0.1%"       if _m_total > 0
        else "0%"
    )

    rows_overview = [
        _row("Rows",                 f"{len(df):,}"),
        _row("Columns",              f"{len(df.columns):,}"),
        _row("Numeric columns",      len(numeric_cols)),
        _row("Categorical columns",  len(cat_cols)),
        _row("Total missing cells",  f"{_m_total:,} ({_m_pct})", True),
        _row("Duplicate rows",       f"{df.duplicated().sum():,}", True),
        _row("Memory usage",         f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"),
    ]

    # Missing values table
    missing_table = ""
    cols_with_missing = missing[missing > 0].sort_values(ascending=False)
    if cols_with_missing.empty:
        missing_table = '<p style="color:#4ade80"> No missing values.</p>'
    else:
        missing_table = '<table style="width:100%;border-collapse:collapse">'
        missing_table += '<tr style="background:#1e1b4b"><th style="padding:8px;text-align:left;color:#a5b4fc">Column</th><th style="padding:8px;color:#a5b4fc">Missing Count</th><th style="padding:8px;color:#a5b4fc">Missing %</th><th style="padding:8px;color:#a5b4fc">Severity</th></tr>'
        for col in cols_with_missing.index:
            pct  = missing_pct[col]
            sev  = (" HIGH" if pct > 30 else " MEDIUM" if pct > 10 else " LOW")
            missing_table += f'<tr style="background:#1a1a2e;border-bottom:1px solid #2d2d44"><td style="padding:7px 12px;color:#e2e8f0">{_html.escape(col)}</td><td style="padding:7px 12px;color:#e2e8f0">{missing[col]:,}</td><td style="padding:7px 12px;color:#e2e8f0">{pct}%</td><td style="padding:7px 12px">{sev}</td></tr>'
        missing_table += '</table>'

    # Statistics table
    desc = df.describe(include="all").T.reset_index()
    desc.columns = ["Column"] + list(df.describe(include="all").index)
    stat_html = '<table style="width:100%;border-collapse:collapse;font-size:0.82rem">'
    stat_html += '<tr style="background:#1e1b4b">' + "".join(
        f'<th style="padding:7px 10px;color:#a5b4fc;text-align:left">{_html.escape(str(c))}</th>'
        for c in desc.columns
    ) + "</tr>"
    for i, row_data in desc.iterrows():
        bg = "#1a1a2e" if i % 2 == 0 else "#15152a"
        stat_html += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:6px 10px;color:#e2e8f0;border-bottom:1px solid #2d2d44">{_html.escape(str(v) if pd.notna(v) else "—")}</td>'
            for v in row_data
        ) + "</tr>"
    stat_html += "</table>"

    # Correlation table (numeric only)
    corr_html = ""
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().round(4)
        corr_html = '<table style="width:100%;border-collapse:collapse;font-size:0.82rem">'
        corr_html += '<tr style="background:#1e1b4b"><th style="padding:7px 10px;color:#a5b4fc">Column</th>' + "".join(
            f'<th style="padding:7px 10px;color:#a5b4fc">{_html.escape(c)}</th>' for c in numeric_cols
        ) + "</tr>"
        for col in numeric_cols:
            corr_html += '<tr style="background:#1a1a2e">'
            corr_html += f'<td style="padding:6px 10px;color:#a5b4fc;border-bottom:1px solid #2d2d44"><b>{_html.escape(col)}</b></td>'
            for col2 in numeric_cols:
                val = corr.loc[col, col2]
                r = int((val + 1) / 2 * 255)
                b = int((1 - (val + 1) / 2) * 255)
                color = f"rgba({r},80,{b},0.3)"
                corr_html += f'<td style="padding:6px 10px;color:#e2e8f0;background:{color};border-bottom:1px solid #2d2d44">{val:.3f}</td>'
            corr_html += "</tr>"
        corr_html += "</table>"

    # Sample data
    sample_html = '<table style="width:100%;border-collapse:collapse;font-size:0.82rem">'
    sample_html += '<tr style="background:#1e1b4b">' + "".join(
        f'<th style="padding:7px 10px;color:#a5b4fc;text-align:left">{_html.escape(str(c))}</th>'
        for c in df.columns
    ) + "</tr>"
    for i, row_data in df.head(10).iterrows():
        bg = "#1a1a2e" if i % 2 == 0 else "#15152a"
        sample_html += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:6px 10px;color:#e2e8f0;border-bottom:1px solid #2d2d44">{_html.escape(str(v))}</td>'
            for v in row_data
        ) + "</tr>"
    sample_html += "</table>"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Dataset Profiling Report</title>
      <style>
        body {{ font-family: 'Inter', sans-serif; background: #0f0f1a; color: #e2e8f0; margin: 0; padding: 2rem; }}
        .badge {{ display:inline-block;background:#1e1b4b;border:1px solid #3730a3;border-radius:999px;padding:4px 14px;font-size:0.8rem;color:#a5b4fc;margin-bottom:1.5rem }}
        table {{ border-radius: 8px; overflow: hidden; }}
        h1 {{ background: linear-gradient(135deg,#a5b4fc,#c4b5fd); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
      </style>
    </head>
    <body>
      <h1>📊 Dataset Profiling Report</h1>
      <div class="badge">Generated by AI Data Scientist Mentor · Pandas fallback engine</div>

      {_section('📋 Overview')}
      <table style="border-collapse:collapse">{''.join(rows_overview)}</table>

      {_section(' Missing Values')}
      {missing_table}

      {_section('📊 Statistical Summary')}
      {stat_html}

      {_section('🔗 Correlation Matrix (Numeric)')}
      {corr_html if corr_html else '<p style="color:#9ca3af">Not enough numeric columns.</p>'}

      {_section('👁️ Sample Data (first 10 rows)')}
      {sample_html}
    </body>
    </html>
    """
    return html


def generate_profiling_report(df: pd.DataFrame, minimal: bool = False) -> tuple[str, str]:
    """
    Generate a profiling HTML report using the best available engine.
    Returns (html_string, engine_name).
    Tries: ydata-profiling → sweetviz → custom pandas HTML.
    """
    # ── Engine 1: ydata-profiling ──────────────────────────────────────────
    try:
        from ydata_profiling import ProfileReport
        report = ProfileReport(
            df,
            title="Dataset Profiling Report",
            minimal=minimal,
            explorative=not minimal,
            progress_bar=False,
        )
        return report.to_html(), "ydata-profiling"
    except ImportError:
        pass
    except Exception as e:
        st.warning(f"ydata-profiling error (falling back): {e}")

    # ── Engine 2: sweetviz ─────────────────────────────────────────────────
    try:
        import sweetviz as sv
        import tempfile, pathlib
        report = sv.analyze(df)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            tmp_path = f.name
        report.show_html(filepath=tmp_path, open_browser=False, layout="vertical")
        html = pathlib.Path(tmp_path).read_text(encoding="utf-8")
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        return html, "sweetviz"
    except ImportError:
        pass
    except Exception as e:
        st.warning(f"sweetviz error (falling back): {e}")

    # ── Engine 3: Custom pandas HTML (always works) ────────────────────────
    return _pandas_html_report(df), "pandas (built-in)"


# ── Inline EDA Charts ───────────────────────────────────────────────────────
_DARK_BG   = "#0f0f1a"
_CARD_BG   = "#1a1a2e"
_ACCENT    = "#6366f1"
_ACCENT2   = "#8b5cf6"
_TEXT      = "#e2e8f0"
_MUTED     = "#9ca3af"

_CHART_STYLE = {
    "figure.facecolor":  _DARK_BG,
    "axes.facecolor":    _CARD_BG,
    "axes.edgecolor":    "#2d2d44",
    "axes.labelcolor":   _TEXT,
    "axes.titlecolor":   _TEXT,
    "xtick.color":       _MUTED,
    "ytick.color":       _MUTED,
    "text.color":        _TEXT,
    "grid.color":        "#2d2d44",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
}

def _apply_style(ax):
    for k, v in _CHART_STYLE.items():
        if k.startswith("axes."):
            pass  # handled at plt.style level
    ax.set_facecolor(_CARD_BG)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d44")


def render_inline_charts(df: pd.DataFrame):
    plt.rcParams.update(_CHART_STYLE)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ── 1. Dataset overview metrics ─────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Dataset Overview</p>', unsafe_allow_html=True)
    total_missing = df.isnull().sum().sum()
    _raw_pct      = total_missing / (len(df) * len(df.columns)) * 100
    _pct_str      = (
        f"({_raw_pct:.1f}%)"   if _raw_pct >= 0.1
        else "(<0.1%)"          if total_missing > 0
        else ""
    )
    dup_rows      = df.duplicated().sum()
    c1, c2, c3, c4 = st.columns(4)
    for col_widget, label, val in [
        (c1, "Rows",           f"{len(df):,}"),
        (c2, "Columns",        f"{len(df.columns):,}"),
        (c3, "Missing Cells",  f"{total_missing:,} {_pct_str}".strip()),
        (c4, "Duplicate Rows", f"{dup_rows:,}"),
    ]:
        col_widget.markdown(
            f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

    # ── 2. Missing values bar chart ─────────────────────────────────────────
    st.markdown('<p class="section-header"> Missing Values</p>', unsafe_allow_html=True)
    missing_s = df.isnull().mean().mul(100).round(2).sort_values(ascending=False)
    has_missing = missing_s[missing_s > 0]
    if has_missing.empty:
        st.success(" No missing values detected across all columns!")
    else:
        fig, ax = plt.subplots(figsize=(8, max(2.5, len(has_missing) * 0.36)))
        colors = ["#ef4444" if v > 30 else "#f97316" if v > 10 else "#eab308"
                  for v in has_missing.values]
        ax.barh(has_missing.index[::-1], has_missing.values[::-1], color=colors[::-1], edgecolor="none")
        ax.set_xlabel("Missing %", color=_TEXT)
        ax.set_title("Missing Values by Column (%)", fontsize=13, color=_TEXT, pad=10)
        _apply_style(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── 3. Numeric distributions ─────────────────────────────────────────────
    if numeric_cols:
        st.markdown('<p class="section-header"> Numeric Distributions</p>', unsafe_allow_html=True)
        cols_per_row = 4
        n_rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row,
                                 figsize=(14, 2.6 * n_rows), squeeze=False)
        fig.patch.set_facecolor(_DARK_BG)
        gradient_colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#818cf8", "#c4b5fd"]
        for idx, col in enumerate(numeric_cols):
            r, c = divmod(idx, cols_per_row)
            ax = axes[r][c]
            data = df[col].dropna()
            ax.hist(data, bins=min(40, len(data.unique())),
                    color=gradient_colors[idx % len(gradient_colors)],
                    edgecolor=_DARK_BG, alpha=0.9)
            ax.set_title(col, fontsize=10, pad=6)
            ax.set_ylabel("Count", fontsize=8)
            _apply_style(ax)
            # overlay mean line
            ax.axvline(data.mean(), color="#f43f5e", linewidth=1.2, linestyle="--", label="mean")
            ax.legend(fontsize=7, facecolor=_CARD_BG, edgecolor="#2d2d44", labelcolor=_TEXT)
        # hide unused axes
        for idx in range(len(numeric_cols), n_rows * cols_per_row):
            r, c = divmod(idx, cols_per_row)
            axes[r][c].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── 4. Correlation heatmap ───────────────────────────────────────────────
    if len(numeric_cols) > 1:
        st.markdown('<p class="section-header"> Correlation Heatmap</p>', unsafe_allow_html=True)
        corr = df[numeric_cols].corr()
        fig_h = max(4, len(numeric_cols) * 0.5)
        fig_w = max(6, len(numeric_cols) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(_DARK_BG)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show full matrix
        sns.heatmap(
            corr, ax=ax,
            annot=True, fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5, linecolor=_DARK_BG,
            annot_kws={"size": 8, "color": _TEXT},
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        )
        ax.set_facecolor(_DARK_BG)
        ax.set_title("Feature Correlation Matrix", fontsize=14, color=_TEXT, pad=14)
        ax.tick_params(colors=_MUTED, labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Highlight top correlations
        pairs = (
            corr.where(np.tril(np.ones(corr.shape), -1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        pairs["Abs"] = pairs["Correlation"].abs()
        top = pairs.sort_values("Abs", ascending=False).head(10).drop(columns="Abs")
        with st.expander(" Top 10 Feature Correlations"):
            st.dataframe(
                top.style.background_gradient(cmap="coolwarm", subset=["Correlation"])
                         .format({"Correlation": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

    # ── 5. Categorical distributions ─────────────────────────────────────────
    if cat_cols:
        st.markdown('<p class="section-header"> Categorical Columns</p>', unsafe_allow_html=True)
        show_cats = cat_cols[:6]  # max 6 to keep UI clean
        cols_per_row = 3
        n_rows = (len(show_cats) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row,
                                 figsize=(13, 2.8 * n_rows), squeeze=False)
        fig.patch.set_facecolor(_DARK_BG)
        cat_colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#7c3aed", "#4f46e5", "#818cf8"]
        for idx, col in enumerate(show_cats):
            r, c = divmod(idx, cols_per_row)
            ax = axes[r][c]
            vc = df[col].value_counts().head(12)
            ax.barh(vc.index.astype(str)[::-1], vc.values[::-1],
                    color=cat_colors[idx % len(cat_colors)], edgecolor="none")
            ax.set_title(f"{col}  ({df[col].nunique()} unique)", fontsize=10, pad=6)
            ax.set_xlabel("Count", fontsize=8)
            _apply_style(ax)
        for idx in range(len(show_cats), n_rows * cols_per_row):
            r, c = divmod(idx, cols_per_row)
            axes[r][c].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── 6. Data types breakdown ──────────────────────────────────────────────
    st.markdown('<p class="section-header"> Data Types</p>', unsafe_allow_html=True)
    dtype_counts = df.dtypes.astype(str).value_counts()
    _dt_col, _ = st.columns([1, 1])   # render at half width
    with _dt_col:
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor(_DARK_BG)
        wedges, texts, autotexts = ax.pie(
            dtype_counts.values,
            labels=dtype_counts.index,
            autopct="%1.0f%%",
            colors=["#6366f1", "#8b5cf6", "#a78bfa", "#7c3aed", "#4f46e5"],
            startangle=140,
            wedgeprops={"edgecolor": _DARK_BG, "linewidth": 2},
        )
        for t in texts + autotexts:
            t.set_color(_TEXT)
            t.set_fontsize(9)
        ax.set_title("Column Data Types", color=_TEXT, fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_eda:
    # ── File uploader ─────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload your Dataset (CSV)",
        type=["csv"],
        help="Supported: CSV files up to ~50 MB",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

            st.success(f" **{uploaded_file.name}** loaded — {len(df):,} rows × {len(df.columns)} columns")

            with st.expander("👁️ Preview Data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)

            # ── Inline EDA Charts ──────────────────────────────────────────
            render_inline_charts(df)

            # ── ydata-profiling report ─────────────────────────────────────
            st.markdown('<p class="section-header"> Full Profiling Report (ydata-profiling)</p>',
                        unsafe_allow_html=True)

            minimal_mode = st.checkbox(
                "⚡ Minimal mode (faster, skips expensive correlations)",
                value=False,
                help="Use for large datasets (>100K rows) to avoid timeouts.",
            )

            col_gen, col_dl = st.columns([2, 1])
            with col_gen:
                if st.button("🔬 Generate Full Profiling Report", type="primary"):
                    with st.spinner("Generating profiling report — trying best available engine…"):
                        html, engine = generate_profiling_report(df, minimal=minimal_mode)
                        st.session_state.profile_html = html
                        st.session_state.profile_engine = engine
                        st.success(f"✅ Report generated using **{engine}**!")

            if st.session_state.get("profile_html"):
                with col_dl:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="⬇️ Download HTML Report",
                        data=st.session_state.profile_html,
                        file_name="profiling_report.html",
                        mime="text/html",
                        use_container_width=True,
                    )

                engine_used = st.session_state.get("profile_engine", "unknown")
                with st.expander(f"🔍 View Profiling Report Inline  ·  engine: {engine_used}", expanded=True):
                    st.components.v1.html(
                        st.session_state.profile_html,
                        height=800,
                        scrolling=True,
                    )

            # ── AI Analysis ────────────────────────────────────────────────
            st.markdown('<p class="section-header">🤖 AI Deep Analysis</p>', unsafe_allow_html=True)

            if not st.session_state.analysis_done:
                if st.button("🚀 Generate AI Analysis", type="primary"):
                    with st.spinner("Analyzing with the heavy analytical model — please wait…"):
                        profile_text = generate_llm_profile(df)
                        st.session_state.dataset_profile = profile_text
                        try:
                            response = client.chat.completions.create(
                                model=ANALYSIS_MODEL,
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user",   "content": (
                                        f"Here is my dataset profile. Please provide the Phase 1 Initial Analysis.\n\n{profile_text}"
                                    )},
                                ],
                                temperature=0.3,
                            )
                            analysis = response.choices[0].message.content
                            st.session_state.messages.append({"role": "assistant", "content": analysis})
                            st.session_state.analysis_done = True
                            st.session_state.phase = 2
                            st.rerun()
                        except Exception as e:
                            st.error(f"LLM error: {e}")
            else:
                st.info("✅ AI analysis complete — switch to the **💬 AI Mentor Chat** tab to continue the walkthrough.")
                if st.button("🔄 Re-run AI Analysis"):
                    st.session_state.analysis_done   = False
                    st.session_state.messages        = []
                    st.session_state.phase           = 1
                    st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.markdown(
            """
            <div class="card" style="text-align:center;padding:3rem;">
                <div style="font-size:4rem;margin-bottom:1rem;">📂</div>
                <h3 style="color:#a5b4fc;">Upload a CSV to get started</h3>
                <p style="color:#6b7280;">
                    You'll get instant inline charts, a downloadable profiling report,<br>
                    and an AI-powered dataset analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — AI MENTOR CHAT
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(
        '<p style="color:#6b7280;font-size:0.9rem;margin-bottom:1rem;">'
        'Powered by <b>llama-3.1-8b-instant</b> — fast, conversational, and context-aware of your dataset.'
        '</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.phase == 1 and not st.session_state.analysis_done:
        st.info("📊 Upload a dataset on the **Dataset Analysis** tab and run AI Analysis first to unlock the chat.")
    else:
        # Render existing messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask your AI mentor anything…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder   = st.empty()
                full_response = ""

                llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if st.session_state.dataset_profile:
                    llm_messages[0]["content"] += (
                        f"\n\nContext — User's Dataset Profile:\n{st.session_state.dataset_profile}"
                    )
                llm_messages.extend(
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                )

                try:
                    stream = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=llm_messages,
                        stream=True,
                        temperature=0.5,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            full_response += delta
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"LLM error: {e}")

        st.divider()
        if st.button("🔄 Start Over (new dataset)"):
            for k in ["messages", "phase", "dataset_profile", "df",
                      "profile_html", "analysis_done"]:
                st.session_state[k] = defaults[k]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CODE SANDBOX
# ═══════════════════════════════════════════════════════════════════════════
with tab_sandbox:
    st.markdown(
        """
        <div class="card" style="display:flex;align-items:flex-start;gap:12px;padding:1rem 1.4rem;">
            <span style="font-size:1.4rem;">⚠️</span>
            <div>
                <b style="color:#fbbf24;">Local Execution Only</b>
                <p style="color:#9ca3af;margin:0.2rem 0 0;font-size:0.85rem;">
                    This sandbox runs Python directly on your machine using <code>exec()</code>.
                    It is safe for local use. Do not deploy this app publicly.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:
        st.info("📊 Upload a CSV on the **Dataset Analysis** tab first — `df` will then be pre-loaded here.")
    else:
        df = st.session_state.df
        st.success(
            f"✅ `df` is ready — **{len(df):,} rows × {len(df.columns)} columns**  "
            f"| Columns: `{', '.join(df.columns[:8].tolist())}{'…' if len(df.columns) > 8 else ''}`"
        )

    st.markdown(
        '<p class="section-header">🧪 Python Sandbox</p>',
        unsafe_allow_html=True,
    )

    # Hint snippets
    with st.expander("💡 Starter Snippets"):
        st.markdown(
            """
            ```python
            # Basic exploration
            print(df.shape)
            df.describe()
            df.dtypes

            # Missing values
            df.isnull().sum().sort_values(ascending=False)

            # Histogram of a column
            import matplotlib.pyplot as plt
            plt.hist(df['column_name'], bins=30)
            plt.title('Distribution')
            plt.show()

            # Correlation
            import seaborn as sns
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            plt.show()

            # Plotly scatter
            import plotly.express as px
            fig = px.scatter(df, x='col_a', y='col_b', color='target')
            fig.show()
            ```
            """
        )

    # Code editor
    code_input = st.text_area(
        "Python Code",
        value=st.session_state.sandbox_code,
        height=300,
        key="code_editor",
        label_visibility="collapsed",
        placeholder="# Write Python here. df, pd, np, plt, sns, px are pre-loaded.\nprint(df.shape)",
    )
    st.session_state.sandbox_code = code_input

    col_run, col_clear, _ = st.columns([1, 1, 4])
    with col_run:
        run_clicked = st.button("▶ Run Code", type="primary", use_container_width=True)
    with col_clear:
        if st.button(" Clear Output", use_container_width=True):
            st.session_state.sandbox_output = ""
            st.session_state.sandbox_figs   = []
            st.rerun()

    if run_clicked and code_input.strip():
        # ── Build execution scope ────────────────────────────────────────
        _df = st.session_state.df if st.session_state.df is not None else pd.DataFrame()

        _scope = {
            "df":  _df,
            "pd":  pd,
            "np":  np,
            "plt": plt,
            "sns": sns,
            "px":  px,
            "go":  go,
            "st":  st,
            "__builtins__": __builtins__,
        }

        # ── Inject ML / science libraries if available ───────────────────
        _optional_imports = [
            ("sklearn",                    "sklearn"),
            ("sklearn.preprocessing",      "preprocessing"),
            ("sklearn.model_selection",    "model_selection"),
            ("sklearn.metrics",            "metrics"),
            ("sklearn.linear_model",       "linear_model"),
            ("sklearn.ensemble",           "ensemble"),
            ("sklearn.tree",               "tree"),
            ("sklearn.svm",                "svm"),
            ("sklearn.cluster",            "cluster"),
            ("sklearn.decomposition",      "decomposition"),
            ("sklearn.pipeline",           "pipeline"),
            ("sklearn.neighbors",          "neighbors"),
            ("scipy",                      "scipy"),
            ("scipy.stats",                "stats"),
        ]
        for _mod_path, _alias in _optional_imports:
            try:
                import importlib as _il
                _scope[_alias] = _il.import_module(_mod_path)
            except ImportError:
                pass  # silently skip missing packages

        # ── Capture stdout ───────────────────────────────────────────────
        stdout_capture = io.StringIO()
        _error = None

        plt.close("all")  # clear any stale figures

        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(code_input, "<sandbox>", "exec"), _scope)  # noqa: S102
        except Exception:
            _error = traceback.format_exc()

        stdout_text = stdout_capture.getvalue()

        # Collect any open matplotlib figures
        _figs = [plt.figure(n) for n in plt.get_fignums()]

        # ── Display output ───────────────────────────────────────────────
        st.markdown(
            '<p class="section-header"> Output</p>',
            unsafe_allow_html=True,
        )

        if stdout_text:
            st.code(stdout_text, language="text")

        if _error:
            st.error("**Runtime Error**")
            st.code(_error, language="python")
        elif not stdout_text and not _figs:
            # Check if the last expression in scope returned something
            # (e.g., df.head() returns a DataFrame)
            # We re-exec the last line with eval to capture its repr
            lines = [l for l in code_input.strip().splitlines() if l.strip()]
            if lines:
                try:
                    result = eval(  # noqa: S307
                        compile(lines[-1], "<sandbox_eval>", "eval"), _scope
                    )
                    if result is not None:
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result, use_container_width=True)
                        elif isinstance(result, pd.Series):
                            st.dataframe(result.to_frame(), use_container_width=True)
                        elif hasattr(result, "show"):  # plotly fig
                            st.plotly_chart(result, use_container_width=True)
                        else:
                            st.code(repr(result), language="text")
                except Exception:
                    st.info("Code executed with no output.")
            else:
                st.info("Code executed with no output.")

        # ── Render matplotlib figures ────────────────────────────────────
        for fig in _figs:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Check scope for plotly figures ───────────────────────────────
        for obj in _scope.values():
            if isinstance(obj, go.Figure):
                try:
                    st.plotly_chart(obj, use_container_width=True)
                except Exception:
                    pass

        if not _error:
            st.success(" Code executed successfully.")
