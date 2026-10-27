import os
import tempfile
import time
import json
import re
import html
import pandas as pd
import numpy as np
import yaml
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from pipeline import run_pipeline

try:
    from realtime.stream_engine import RealTimeStreamEngine
except ImportError:
    from src.realtime.stream_engine import RealTimeStreamEngine

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BeyondBots - Discourse & Inauthentic Coordination Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. LOAD CONFIGURATION
# -----------------------------------------------------------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# -----------------------------------------------------------------------------
# 3. INSTITUTIONAL DESIGN SYSTEM & CSS (Light Theme Workbench)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Institutional Light Theme Variables */
    :root {
        --canvas-base: #F8FAFC;
        --surface-panel: #FFFFFF;
        --border-panel: #E2E8F0;
        --border-divider: #EEF2F6;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-tertiary: #94A3B8;
        
        --color-organic-text: #047857;
        --color-organic-bg: #ECFDF5;
        --color-organic-border: #A7F3D0;

        --color-fan-text: #1D4ED8;
        --color-fan-bg: #EFF6FF;
        --color-fan-border: #BFDBFE;

        --color-paid-text: #B91C1C;
        --color-paid-bg: #FEF2F2;
        --color-paid-border: #FECACA;

        --color-warn-text: #B45309;
        --color-warn-bg: #FFFBEB;
        --color-warn-border: #FDE68A;
    }

    /* Base Typography & Numbers */
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: var(--text-primary);
    }
    
    .stApp {
        background-color: var(--canvas-base);
    }

    .mono-num {
        font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
        font-variant-numeric: tabular-nums;
    }

    /* Header Structure */
    .app-header {
        background: #FFFFFF;
        border: 1px solid var(--border-panel);
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    }
    .app-header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .app-header-subtitle {
        color: var(--text-secondary);
        font-size: 0.875rem;
        line-height: 1.4;
    }

    /* Structured Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: var(--surface-panel);
        border: 1px solid var(--border-panel);
        border-radius: 6px;
        padding: 14px 18px;
        transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-box:hover {
        border-color: #CBD5E1;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-box-label {
        font-size: 0.725rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .metric-box-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: "JetBrains Mono", "SF Mono", monospace;
        font-variant-numeric: tabular-nums;
        line-height: 1.1;
    }
    .metric-box-sub {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        margin-top: 6px;
        font-variant-numeric: tabular-nums;
    }

    /* Verdict Banners */
    .verdict-card {
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .verdict-organic {
        background: var(--color-organic-bg);
        border: 1px solid var(--color-organic-border);
        color: var(--color-organic-text);
    }
    .verdict-fan {
        background: var(--color-fan-bg);
        border: 1px solid var(--color-fan-border);
        color: var(--color-fan-text);
    }
    .verdict-paid {
        background: var(--color-paid-bg);
        border: 1px solid var(--color-paid-border);
        color: var(--color-paid-text);
    }
    .verdict-warn {
        background: var(--color-warn-bg);
        border: 1px solid var(--color-warn-border);
        color: var(--color-warn-text);
    }

    /* Token Attribution Spans */
    .token-text-container {
        background: #FFFFFF;
        border: 1px solid var(--border-panel);
        border-radius: 6px;
        padding: 16px 18px;
        font-size: 0.95rem;
        line-height: 1.8;
        color: var(--text-primary);
        margin-bottom: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .token-promo {
        background: rgba(239, 68, 68, 0.12);
        color: #B91C1C;
        border-bottom: 2px solid #DC2626;
        padding: 2px 4px;
        border-radius: 2px;
        font-weight: 600;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.875rem;
    }
    .token-fan {
        background: rgba(59, 130, 246, 0.12);
        color: #1D4ED8;
        border-bottom: 2px solid #2563EB;
        padding: 2px 4px;
        border-radius: 2px;
        font-weight: 600;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.875rem;
    }
    .token-spec {
        background: rgba(16, 185, 129, 0.12);
        color: #047857;
        border-bottom: 2px solid #059669;
        padding: 2px 4px;
        border-radius: 2px;
        font-weight: 600;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.875rem;
    }

    /* Monospace Legend Strip */
    .legend-strip {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.725rem;
        color: var(--text-secondary);
        padding: 8px 12px;
        background: #F1F5F9;
        border-radius: 4px;
        margin-bottom: 16px;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .legend-indicator {
        width: 10px;
        height: 10px;
        border-radius: 2px;
        display: inline-block;
    }

    /* Button and Interactive Transitions */
    .stButton>button {
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.875rem;
        transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .stDownloadButton>button {
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.875rem;
        transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--border-panel);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
        border-radius: 4px 4px 0 0;
        transition: color 0.15s ease, border-color 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #0F172A !important;
        font-weight: 600;
        border-bottom: 2px solid #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. APP HEADER (Austere, Institutional Visuals)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-header-title">BeyondBots: Computational Discourse & Inauthentic Coordination Intelligence</div>
    <div class="app-header-subtitle">Multi-stage NLP forensics workbench for detecting astroturfing campaigns, fan-army coordination, and organic opinion dynamics in cinema discourse.</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. SIDEBAR: DATASET CONTROLS & PIPELINE EXECUTION
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Ingestion & Dataset Controls")

mode = st.sidebar.radio(
    "Corpus Source Mode",
    ["High-Volume Datasets (1,000+ Posts)", "Upload External File"],
    index=0
)

films = config.get("films", [])
film_options = {f["title"]: f["id"] for f in films}
selected_film_title = st.sidebar.selectbox("Target Release Entity", list(film_options.keys()))
selected_film_id = film_options[selected_film_title]
selected_film_cfg = next((f for f in films if f["id"] == selected_film_id), {})

override_path = None
if mode == "High-Volume Datasets (1,000+ Posts)":
    dataset_mapping = {
        "Animal": "data/animal_2023_posts.csv",
        "Jailer": "data/jailer_2023_posts.csv",
        "Kalki_2898_AD": "data/kalki_2024_posts.csv",
        "RRR": "data/rrr_2022_posts.csv",
        "Jawan": "data/jawan_2023_posts.csv",
        "Pathaan": "data/pathaan_2023_posts.csv",
        "Pushpa 2": "data/pushpa2_2024_posts.csv"
    }
    override_path = dataset_mapping.get(selected_film_title, f"data/{selected_film_id}_posts.csv")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Ingestion File (.csv, .txt, .pdf, .docx)",
        type=["csv", "txt", "pdf", "docx", "doc"]
    )
    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            override_path = tmp.name

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Execute Pipeline", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 6. PIPELINE EXECUTION & CACHE RETRIEVAL
# -----------------------------------------------------------------------------
if run_btn:
    progress_bar = st.sidebar.progress(0.0)
    status_box = st.sidebar.empty()

    def update_progress(stage: int, name: str, pct: float, msg: str):
        progress_bar.progress(min(max(pct, 0.0), 1.0))
        status_box.text(f"Stage {stage} [{name}]: {msg}")

    with st.spinner(f"Executing deterministic analysis for {selected_film_title}..."):
        try:
            os.environ["BEYONDBOTS_MOCK_EMBEDDINGS"] = "1"
            os.environ["BEYONDBOTS_MOCK_SENTIMENT"] = "1"

            df_res = run_pipeline(
                film_id=selected_film_id,
                config_path="config.yaml",
                override_input_file=override_path,
                progress_callback=update_progress
            )
            st.session_state["df_labeled"] = df_res
            st.session_state["current_film"] = selected_film_id

            shap_path = f"data/labeled/{selected_film_id}_shap.json"
            if os.path.exists(shap_path):
                with open(shap_path, "r", encoding="utf-8") as f_shap:
                    st.session_state["shap_data"] = json.load(f_shap)
            else:
                st.session_state["shap_data"] = None

            if os.path.exists("data/processed/muril_representation.parquet"):
                st.session_state["df_embeddings"] = pd.read_parquet("data/processed/muril_representation.parquet")
            else:
                st.session_state["df_embeddings"] = None

            cp_path = f"data/labeled/{selected_film_id}_cross_platform.json"
            if os.path.exists(cp_path):
                with open(cp_path, "r", encoding="utf-8") as f_cp:
                    st.session_state["cross_platform_summary"] = json.load(f_cp)
            else:
                st.session_state["cross_platform_summary"] = None

            imdb_path = f"data/labeled/{selected_film_id}_imdb.json"
            if os.path.exists(imdb_path):
                with open(imdb_path, "r", encoding="utf-8") as f_imdb:
                    st.session_state["imdb_data"] = json.load(f_imdb)
            else:
                st.session_state["imdb_data"] = None

            st.sidebar.success("Execution completed successfully.")
        except Exception as e:
            st.sidebar.error(f"Execution failed: {e}")
        finally:
            if mode == "Upload External File" and override_path and os.path.exists(override_path):
                try:
                    os.remove(override_path)
                except Exception:
                    pass

# Dynamic per-film data loading (instant sync when dropdown changes)
film_parquet = f"data/labeled/{selected_film_id}_labeled.parquet"
if os.path.exists(film_parquet):
    try:
        st.session_state["df_labeled"] = pd.read_parquet(film_parquet)
        st.session_state["current_film"] = selected_film_id

        film_shap = f"data/labeled/{selected_film_id}_shap.json"
        if os.path.exists(film_shap):
            with open(film_shap, "r", encoding="utf-8") as f_shap:
                st.session_state["shap_data"] = json.load(f_shap)

        film_cp = f"data/labeled/{selected_film_id}_cross_platform.json"
        if os.path.exists(film_cp):
            with open(film_cp, "r", encoding="utf-8") as f_cp:
                st.session_state["cross_platform_summary"] = json.load(f_cp)

        film_imdb = f"data/labeled/{selected_film_id}_imdb.json"
        if not os.path.exists(film_imdb):
            film_imdb = f"data/imdb/{selected_film_id}_imdb.json"
        if os.path.exists(film_imdb):
            with open(film_imdb, "r", encoding="utf-8") as f_imdb:
                st.session_state["imdb_data"] = json.load(f_imdb)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 7. DETERMINISTIC TEXT INFERENCE ENGINE (Single-Post Forensic Sandbox)
# -----------------------------------------------------------------------------
def analyze_single_post_deterministic(
    text: str, 
    film_cfg: dict, 
    sensitivity: float = 0.50, 
    penalize_duplicates: bool = True
):
    """
    Evaluates raw text against empirical lexicons and stylometric heuristics in <50ms.
    Returns feature scores, calibrated probabilities, label, anomaly level, and highlighted HTML.
    """
    if not text.strip():
        return None

    # Retrieve film-specific and global lexicons
    promo_keywords = set([k.lower() for k in film_cfg.get("promotional_keywords", [])])
    global_promo = {
        "blockbuster", "masterpiece", "mega hit", "historic opening", "all time record",
        "must watch", "pure cinema", "box office rampage", "unprecedented", "mass film of the year",
        "records broken", "sensational", "goosebumps", "visual spectacle", "all time blockbuster",
        "super hit", "industry hit", "collection", "crores", "mindblowing", "phenomenal",
        "ticket booked", "book your tickets", "storm", "historic", "record breaking"
    }
    all_promo = promo_keywords.union(global_promo)

    fan_markers = set([m.lower() for m in film_cfg.get("fan_markers", [])])
    global_fan = {
        "king khan", "badshah", "srkian", "thalapathy", "anirudh", "icon star", "allu arjun",
        "bunny", "pushpa raj", "thaggede le", "rebel star", "prabhas", "kalkiarmy", "tarak",
        "cherry", "ram charan", "pathaanarmy", "megastar", "superstar", "box office king",
        "boss", "darling", "vanga", "ranbir", "rk"
    }
    all_fan = fan_markers.union(global_fan)

    cast_and_crew = set([c.lower() for c in film_cfg.get("cast", [])])
    if film_cfg.get("director"):
        cast_and_crew.add(film_cfg.get("director", "").lower())
    
    technical_craft = {
        "screenplay", "cinematography", "direction", "bgm", "editing", "pacing", "dialogue",
        "vfx", "story", "acting", "performance", "character", "interval", "climax", "scene",
        "first half", "second half", "plot", "soundtrack", "production design", "camerawork"
    }
    critique_negative = {
        "disappointing", "boring", "flaws", "lag", "poor", "average", "waste", "slow",
        "cringe", "bad", "worst", "overhyped", "dragged", "predictable", "weak", "lacks",
        "mediocre", "disaster", "terrible", "unconvincing"
    }

    tokens_raw = re.findall(r'\b\w+\b', text)
    tokens_lower = [t.lower() for t in tokens_raw]
    total_tokens = len(tokens_lower) if tokens_lower else 1
    unique_tokens = len(set(tokens_lower))

    # 1. Feature 1: Promotional Vocabulary Density
    matched_promo_spans = []
    text_lower = text.lower()
    for phrase in sorted(all_promo, key=lambda x: -len(x)):
        if phrase in text_lower:
            matched_promo_spans.append(phrase)
    promo_token_hits = sum(len(phrase.split()) for phrase in matched_promo_spans)
    score_marketing_copy = min(float(promo_token_hits / total_tokens * 2.5), 1.0)

    # 2. Feature 2: Fan Identity & Honorific Ratio
    matched_fan_spans = []
    for phrase in sorted(all_fan, key=lambda x: -len(x)):
        if phrase in text_lower:
            matched_fan_spans.append(phrase)
    fan_token_hits = sum(len(phrase.split()) for phrase in matched_fan_spans)
    score_fan_identity = min(float(fan_token_hits / total_tokens * 2.5), 1.0)

    # 3. Feature 3: Entity & Narrative Specificity
    matched_spec_spans = []
    for entity in sorted(cast_and_crew.union(technical_craft), key=lambda x: -len(x)):
        if entity in text_lower:
            matched_spec_spans.append(entity)
    spec_hits = len(matched_spec_spans)
    numbers_count = len(re.findall(r'\b\d+\b', text))
    proper_nouns = sum(1 for i, t in enumerate(tokens_raw) if i > 0 and t[0].isupper())
    score_specificity = min(float((spec_hits * 1.5 + numbers_count * 0.5 + proper_nouns * 0.2) / total_tokens), 1.0)

    # 4. Feature 4: Critical Variance & Criticism Absence
    matched_crit_spans = [w for w in critique_negative if w in text_lower]
    has_critique = len(matched_crit_spans) > 0
    if has_critique:
        score_criticism_absence = max(0.05, 0.40 - len(matched_crit_spans) * 0.15)
    else:
        # Absence of critique is high if praise is high without dissent
        score_criticism_absence = min(0.95, 0.65 + score_marketing_copy * 0.3)

    # Repetition / duplicate token metric
    rep_ratio = float(unique_tokens / total_tokens) if total_tokens > 0 else 1.0
    duplicate_penalty = (1.0 - rep_ratio) if penalize_duplicates else 0.0

    # Calibrated Weight Attribution
    w_paid = (score_marketing_copy * 3.2) + (score_criticism_absence * 1.5) - (score_specificity * 1.4) + (duplicate_penalty * 1.2)
    w_fan = (score_fan_identity * 3.4) + (score_criticism_absence * 0.8) + (duplicate_penalty * 0.9)
    w_organic = (score_specificity * 2.6) + ((1.0 - score_criticism_absence) * 2.0) + 0.3 - (score_marketing_copy * 1.2)

    # Apply sensitivity factor
    sens_adj = (sensitivity - 0.50) * 2.0  # -1.0 to +1.0
    w_paid += sens_adj * 1.0
    w_fan += sens_adj * 0.5

    # Exponentiate for calibrated Softmax distribution
    exp_paid = np.exp(np.clip(w_paid, -5.0, 5.0))
    exp_fan = np.exp(np.clip(w_fan, -5.0, 5.0))
    exp_org = np.exp(np.clip(w_organic, -5.0, 5.0))
    total_exp = exp_paid + exp_fan + exp_org

    prob_paid = float(exp_paid / total_exp)
    prob_fan = float(exp_fan / total_exp)
    prob_organic = float(exp_org / total_exp)

    # Final label resolution
    probs = {
        "PAID-COMMERCIAL": prob_paid,
        "FAN-COORDINATED": prob_fan,
        "ORGANIC": prob_organic
    }
    pred_label = max(probs, key=probs.get)
    confidence = probs[pred_label]

    # Anomaly Index
    if prob_paid > 0.70 or prob_fan > 0.80:
        anomaly_index = "HIGH"
    elif prob_paid > 0.45 or prob_fan > 0.55:
        anomaly_index = "ELEVATED"
    elif prob_paid > 0.25 or prob_fan > 0.35:
        anomaly_index = "MODERATE"
    else:
        anomaly_index = "LOW"

    # In-Situ Token Highlighting Span Generation
    # Replace matched keywords with styled <mark> tags using regex word boundaries
    highlighted_text = html.escape(text)
    
    # Highlight technical craft / specificity (Green)
    for term in sorted(matched_spec_spans, key=lambda x: -len(x)):
        escaped_term = re.escape(html.escape(term))
        highlighted_text = re.sub(
            rf'(?i)\b({escaped_term})\b',
            r'<mark class="token-spec">\1</mark>',
            highlighted_text
        )

    # Highlight fan honorifics (Blue)
    for term in sorted(matched_fan_spans, key=lambda x: -len(x)):
        escaped_term = re.escape(html.escape(term))
        highlighted_text = re.sub(
            rf'(?i)\b({escaped_term})\b',
            r'<mark class="token-fan">\1</mark>',
            highlighted_text
        )

    # Highlight promotional vocabulary (Red)
    for term in sorted(matched_promo_spans, key=lambda x: -len(x)):
        escaped_term = re.escape(html.escape(term))
        highlighted_text = re.sub(
            rf'(?i)\b({escaped_term})\b',
            r'<mark class="token-promo">\1</mark>',
            highlighted_text
        )

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "anomaly_index": anomaly_index,
        "probabilities": {
            "Organic": prob_organic,
            "Fan-Coordinated": prob_fan,
            "Paid-Commercial": prob_paid
        },
        "scores": {
            "score_marketing_copy": score_marketing_copy,
            "score_fan_identity": score_fan_identity,
            "score_specificity": score_specificity,
            "score_criticism_absence": score_criticism_absence
        },
        "highlighted_text": highlighted_text,
        "stats": {
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "unique_ratio": rep_ratio
        }
    }

# -----------------------------------------------------------------------------
# 8. WORKBENCH TABS
# -----------------------------------------------------------------------------
tab_realtime, tab_classifier, tab_overview, tab_imdb, tab_embeddings, tab_explainers, tab_explorer, tab_benchmark = st.tabs([
    "⚡ Real-Time Stream Analytics",
    "Deterministic Text Classifier",
    "Discourse Overview",
    "Cross-Platform Intelligence (IMDb)",
    "Semantic Embedding Space",
    "Feature Importance & SHAP",
    "Cluster & Post Explorer",
    "Cross-Film Benchmark Comparison"
])

# =============================================================================
# TAB 0: REAL-TIME STREAM ANALYTICS (Release-Day FDFS Engine)
# =============================================================================
with tab_realtime:
    st.markdown("#### ⚡ Real-Time Streaming Forensics (Release-Day FDFS Monitor)")
    st.markdown(
        "<p style='color: #475569; font-size: 0.875rem; margin-top: -8px;'>"
        "Sub-millisecond stream processor simulating live X/Twitter discourse on release day. "
        "Operates via a <b>Cascaded Fast-Path architecture</b>: combines online EWMA burst detection, "
        "real-time MinHash template registration, and streaming cross-platform Sentiment Divergence Index (SDI) "
        "without requiring batch pipeline re-clustering."
        "</p>",
        unsafe_allow_html=True
    )

    # Initialize Real-Time Engine in Session State
    imdb_meta_rating = 7.0
    if st.session_state.get("imdb_data"):
        imdb_meta_rating = float(st.session_state["imdb_data"].get("rating", 7.0))

    if (
        "rt_engine" not in st.session_state or 
        st.session_state.get("rt_film_id") != selected_film_id
    ):
        st.session_state["rt_engine"] = RealTimeStreamEngine(selected_film_cfg, imdb_rating=imdb_meta_rating)
        st.session_state["rt_film_id"] = selected_film_id
        st.session_state["rt_stream_idx"] = 0

    rt_engine: RealTimeStreamEngine = st.session_state["rt_engine"]

    # Stream Controls
    c_ctrl1, c_ctrl2 = st.columns([1.1, 0.9], gap="medium")

    with c_ctrl1:
        st.markdown("**Live Ingestion Source Selection**")
        rt_source_mode = st.radio(
            "Select Stream Corpus:",
            [
                "Pushpa 2 Benchmark Test Stream (24 Curated PR, Fan & Organic Posts)",
                f"{selected_film_title} High-Volume Corpus Stream",
                "Custom Injected Live Posts Only"
            ],
            index=0 if selected_film_id == "pushpa2_2024" else 1,
            horizontal=False
        )

        # Load candidate corpus
        stream_items = []
        if rt_source_mode == "Pushpa 2 Benchmark Test Stream (24 Curated PR, Fan & Organic Posts)":
            test_path = "data/pushpa2_test_samples.csv"
            if os.path.exists(test_path):
                df_stream = pd.read_csv(test_path)
                for _, r in df_stream.iterrows():
                    stream_items.append({
                        "text": str(r["text"]),
                        "author": str(r.get("author", "user")),
                        "timestamp": float(r.get("created_utc", time.time()))
                    })
        elif rt_source_mode == f"{selected_film_title} High-Volume Corpus Stream":
            hv_path = override_path if override_path and os.path.exists(override_path) else f"data/{selected_film_id}_posts.csv"
            if os.path.exists(hv_path):
                df_stream = pd.read_csv(hv_path)
                for _, r in df_stream.iterrows():
                    txt = str(r.get("text", r.get("body", "")))
                    if txt:
                        stream_items.append({
                            "text": txt,
                            "author": str(r.get("author", r.get("username", "user"))),
                            "timestamp": float(r.get("created_utc", time.time()))
                        })

        st.session_state["rt_available_items"] = stream_items
        total_available = len(stream_items)
        curr_idx = st.session_state.get("rt_stream_idx", 0)

        # Action Buttons
        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        with btn_c1:
            step_5 = st.button("▶ Stream 5", help="Process next 5 posts through live stream", use_container_width=True)
        with btn_c2:
            step_20 = st.button("⏩ Stream 20", help="Process next 20 posts through live stream", use_container_width=True)
        with btn_c3:
            step_all = st.button("⚡ Stream All", help="Stream all remaining posts into the engine", use_container_width=True)
        with btn_c4:
            reset_stream = st.button("🔄 Reset", help="Reset real-time streaming state", use_container_width=True)

        if reset_stream:
            st.session_state["rt_engine"] = RealTimeStreamEngine(selected_film_cfg, imdb_rating=imdb_meta_rating)
            st.session_state["rt_stream_idx"] = 0
            st.rerun()

        # Step processing logic
        posts_to_process = 0
        if step_5:
            posts_to_process = 5
        elif step_20:
            posts_to_process = 20
        elif step_all:
            posts_to_process = max(0, total_available - curr_idx)

        if posts_to_process > 0 and total_available > 0:
            end_idx = min(curr_idx + posts_to_process, total_available)
            for i in range(curr_idx, end_idx):
                item = stream_items[i]
                rt_engine.process_post(
                    text=item["text"],
                    author=item["author"],
                    timestamp=item.get("timestamp", time.time())
                )
            st.session_state["rt_stream_idx"] = end_idx
            st.rerun()

    with c_ctrl2:
        st.markdown("**Live Single-Post Injection (Simulate Live User)**")
        manual_text = st.text_input(
            "Inject text into live stream:",
            placeholder="e.g. Pushpa 2 is sensational blockbuster! Records broken everywhere! #Pushpa2"
        )
        manual_author = st.text_input("Author Handle:", value="live_user_x", max_chars=30)
        inject_btn = st.button("🚀 Inject Post into Live Stream", type="secondary", use_container_width=True)
        if inject_btn and manual_text.strip():
            rt_engine.process_post(
                text=manual_text.strip(),
                author=manual_author.strip() or "live_user",
                timestamp=time.time()
            )
            st.success("Post successfully ingested and analyzed by stream engine!")
            st.rerun()

    # Streaming State Snapshot
    summary = rt_engine.get_summary_state()
    sdi_info = summary["sdi_data"]

    st.markdown("---")
    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-title">Streamed Volume</div>
            <div class="metric-box-val">{summary['posts_processed']}</div>
            <div class="metric-box-sub">Queue Progress: {st.session_state.get('rt_stream_idx', 0)} / {len(st.session_state.get('rt_available_items', []))}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        burst_hist = summary["burst_history"]
        last_rate = burst_hist[-1]["rate"] if burst_hist else 0.0
        is_burst = burst_hist[-1]["is_burst"] if burst_hist else False
        burst_badge = "<span style='color: #DC2626; font-weight:700;'>🚨 BURST SPIKE</span>" if is_burst else "<span style='color: #16A34A;'>NORMAL</span>"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-title">Live Velocity (EWMA)</div>
            <div class="metric-box-val">{last_rate:.0f} <span style="font-size:0.9rem; font-weight:400; color:#64748B;">posts/min</span></div>
            <div class="metric-box-sub">Status: {burst_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        active_tpl_count = summary["active_templates_count"]
        total_hits = sum(t["hits"] for t in summary["active_templates"])
        tpl_badge_color = "#DC2626" if active_tpl_count > 0 else "#16A34A"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-title">Active PR/Fan Templates</div>
            <div class="metric-box-val" style="color: {tpl_badge_color};">{active_tpl_count}</div>
            <div class="metric-box-sub">Total Duplicates Caught: {total_hits} posts</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        sdi_val = sdi_info["sdi"]
        sdi_color = "#DC2626" if abs(sdi_val) > 0.20 else "#16A34A"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-title">Live Sentiment Divergence (SDI)</div>
            <div class="metric-box-val" style="color: {sdi_color};">{sdi_val:+.2f}</div>
            <div class="metric-box-sub">{sdi_info['status']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Active Live Alerts
    if summary["alerts"]:
        st.markdown("##### 🚨 Active Forensics Stream Alerts")
        for alert in summary["alerts"][-3:]:
            alert_color = "#FEF2F2" if alert.get("severity") == "CRITICAL" else "#FFFBEB"
            border_color = "#F87171" if alert.get("severity") == "CRITICAL" else "#FCD34D"
            text_color = "#991B1B" if alert.get("severity") == "CRITICAL" else "#92400E"
            st.markdown(f"""
            <div style="background:{alert_color}; border:1px solid {border_color}; border-radius:6px; padding:10px 14px; margin-bottom:8px; color:{text_color}; font-size:0.875rem;">
                <b>[{alert.get('type')}]</b> ({alert.get('severity')}) - {alert.get('message')}
            </div>
            """, unsafe_allow_html=True)

    # Live Visualizations (Velocity & Motive Charts)
    ch_col1, ch_col2 = st.columns([1.2, 0.8], gap="medium")

    with ch_col1:
        st.markdown("**Live Streaming Velocity & Burst Spike Monitor**")
        if burst_hist:
            df_burst = pd.DataFrame(burst_hist)
            fig_burst = go.Figure()
            fig_burst.add_trace(go.Scatter(
                y=df_burst["rate"],
                mode="lines+markers",
                name="Post Velocity (posts/min)",
                line=dict(color="#2563EB", width=2),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)"
            ))
            fig_burst.add_trace(go.Scatter(
                y=df_burst["ewma"],
                mode="lines",
                name="EWMA Baseline",
                line=dict(color="#64748B", dash="dash", width=1.5)
            ))
            # Highlight spikes
            spikes = df_burst[df_burst["is_burst"] == True]
            if not spikes.empty:
                fig_burst.add_trace(go.Scatter(
                    x=spikes.index,
                    y=spikes["rate"],
                    mode="markers",
                    name="Spike Trigger",
                    marker=dict(color="#DC2626", size=10, symbol="triangle-up")
                ))

            fig_burst.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=10, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Streaming Intervals", showgrid=False),
                yaxis=dict(title="Velocity (posts/min)", showgrid=True, gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_burst, use_container_width=True)
        else:
            st.info("Start the stream simulation above to view the live velocity and burst monitor.")

    with ch_col2:
        st.markdown("**Live Motive Composition Stream**")
        mot_counts = summary["motive_counts"]
        if sum(mot_counts.values()) > 0:
            df_mot = pd.DataFrame([
                {"Motive": "Organic", "Count": mot_counts["organic"], "Color": "#10B981"},
                {"Motive": "Fan-Coordinated", "Count": mot_counts["fan-coordinated"], "Color": "#3B82F6"},
                {"Motive": "Paid-Commercial", "Count": mot_counts["paid-commercial"], "Color": "#EF4444"}
            ])
            fig_donut = px.pie(
                df_mot,
                values="Count",
                names="Motive",
                hole=0.55,
                color="Motive",
                color_discrete_map={
                    "Organic": "#10B981",
                    "Fan-Coordinated": "#3B82F6",
                    "Paid-Commercial": "#EF4444"
                }
            )
            fig_donut.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No streamed posts processed yet.")

    # Active Template Watchlist & Live Feed
    tab_tpl_watch, tab_live_feed = st.tabs([
        "📋 Active PR & Fan Template Registry Watchlist",
        "📡 Live Forensic Ingestion Feed"
    ])

    with tab_tpl_watch:
        active_templates = summary["active_templates"]
        if active_templates:
            st.markdown(
                f"<p style='color:#475569; font-size:0.85rem;'>"
                f"Displaying <b>{len(active_templates)} active template clusters</b> caught by streaming MinHash fingerprinting with $\\ge 2$ repetitions."
                f"</p>",
                unsafe_allow_html=True
            )
            for tpl in active_templates:
                m_color = "#B91C1C" if tpl["motive"] == "paid-commercial" else "#1D4ED8"
                m_bg = "#FEF2F2" if tpl["motive"] == "paid-commercial" else "#EFF6FF"
                m_border = "#F87171" if tpl["motive"] == "paid-commercial" else "#BFDBFE"
                authors_str = ", ".join(tpl["authors"][:4])
                if len(tpl["authors"]) > 4:
                    authors_str += f" (+{len(tpl['authors']) - 4} more)"

                st.markdown(f"""
                <div style="background:{m_bg}; border:1px solid {m_border}; border-radius:6px; padding:12px 16px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-weight:700; color:{m_color}; font-size:0.875rem;">
                            IDENTIFIED TEMPLATE [{tpl['id']}] • {tpl['motive'].upper()}
                        </span>
                        <span style="background:{m_color}; color:#FFFFFF; border-radius:12px; padding:2px 10px; font-size:0.75rem; font-weight:700;">
                            {tpl['hits']} Repetitions
                        </span>
                    </div>
                    <div style="font-size:0.9rem; color:#1E293B; line-height:1.5; margin-bottom:6px; font-style:italic;">
                        "{tpl['text']}"
                    </div>
                    <div style="font-size:0.75rem; color:#64748B;">
                        <b>Coordinated Accounts:</b> {authors_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#64748B; font-size:0.875rem; padding:16px; border:1px dashed #CBD5E1; border-radius:6px; text-align:center;'>"
                "No repeating coordinated templates detected yet. Stream more posts to activate template matching."
                "</div>",
                unsafe_allow_html=True
            )

    with tab_live_feed:
        recent_posts = summary["recent_posts"]
        if recent_posts:
            st.markdown(
                f"<p style='color:#475569; font-size:0.85rem;'>"
                f"Showing last <b>{len(recent_posts)} incoming posts</b> analyzed with in-situ token attribution:"
                f"</p>",
                unsafe_allow_html=True
            )
            for post in recent_posts:
                mot = post["motive"]
                if mot == "paid-commercial":
                    badge_style = "background:#FEF2F2; color:#B91C1C; border:1px solid #F87171;"
                elif mot == "fan-coordinated":
                    badge_style = "background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE;"
                else:
                    badge_style = "background:#F0FDF4; color:#15803D; border:1px solid #BBF7D0;"

                tpl_tag = ""
                if post.get("template_id"):
                    tpl_tag = f"<span style='background:#F1F5F9; color:#475569; border:1px solid #CBD5E1; border-radius:4px; padding:2px 6px; font-size:0.7rem; margin-left:6px;'>TPL: {post['template_id']} (Hit #{post.get('template_hits', 1)})</span>"

                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:12px 16px; margin-bottom:8px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <div>
                            <span style="font-weight:600; color:#0F172A; font-size:0.85rem;">@{post['author']}</span>
                            <span style="color:#94A3B8; font-size:0.75rem; margin-left:8px;">{post['post_id']}</span>
                            {tpl_tag}
                        </div>
                        <span style="{badge_style} border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700;">
                            {mot.upper()} ({int(post['confidence'] * 100)}%)
                        </span>
                    </div>
                    <div style="font-size:0.9rem; line-height:1.6; color:#1E293B;">
                        {post['highlighted_text']}
                    </div>
                    <div style="font-size:0.725rem; color:#64748B; margin-top:6px; display:flex; gap:12px;">
                        <span>Marketing: <b>{post['scores']['marketing']:.2f}</b></span>
                        <span>Fan Marker: <b>{post['scores']['fan']:.2f}</b></span>
                        <span>Specificity: <b>{post['scores']['specificity']:.2f}</b></span>
                        <span>Criticism Absence: <b>{post['scores']['criticism_absence']:.2f}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No posts streamed yet. Click 'Stream 5' or 'Stream All' above to begin.")

# =============================================================================
# TAB 1: DETERMINISTIC TEXT CLASSIFIER (Single-Post Forensic Sandbox)
# =============================================================================
with tab_classifier:
    st.markdown("#### Real-Time Forensic Post Inspector")
    st.markdown(
        "<p style='color: #475569; font-size: 0.875rem; margin-top: -8px;'>"
        "Single-post deterministic inference engine. Evaluates raw text against entity lexicons, stylometric densities, "
        "and syntactical absence metrics without requiring batch pipeline executions."
        "</p>",
        unsafe_allow_html=True
    )

    c_left, c_right = st.columns([1.1, 1.0], gap="medium")

    with c_left:
        st.markdown("**Benchmark Empirical Test Vectors**")
        sample_choice = st.radio(
            "Select predefined empirical test case:",
            [
                "Custom Text Input",
                "Case A: Commercial / PR Astroturf Pattern",
                "Case B: Fan-Army Coordinated Campaign",
                "Case C: Uncoordinated Organic Critique"
            ],
            index=0,
            horizontal=False
        )

        preset_texts = {
            "Case A: Commercial / PR Astroturf Pattern": 
                f"Unprecedented historic opening! #{selected_film_title} creates all-time record at the box office. "
                f"Pure cinema and sheer masterpiece. A sensational blockbuster that is an absolute must-watch for everyone. "
                f"Go book your tickets now!",
            "Case B: Fan-Army Coordinated Campaign": 
                f"King Khan is back! Megastar rampage begins today. Badshah of Bollywood proving once again why he rules cinema. "
                f"Goosebumps throughout the theater. All Hail Badshah #{selected_film_title}Day #Pride",
            "Case C: Uncoordinated Organic Critique": 
                f"Watched the morning show of {selected_film_title}. The interval sequence and technical background score were solid, "
                f"but the screenplay drags heavily in the second half. Routine tropes and pacing flaws throughout."
        }

        default_text = preset_texts.get(sample_choice, "")

        input_text = st.text_area(
            "Input Discourse Post Text:",
            value=default_text,
            height=140,
            placeholder="Enter raw tweet, review, or social post text for forensic attribution..."
        )

        with st.expander("Configuration & Sensitivity Constraints", expanded=False):
            cfg_col1, cfg_col2 = st.columns(2)
            with cfg_col1:
                threshold_sensitivity = st.slider(
                    "Threshold Sensitivity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.05,
                    help="Calibrates the classification decision boundary for promotional signals."
                )
            with cfg_col2:
                penalize_repetition = st.checkbox(
                    "Duplicate Token Penalty",
                    value=True,
                    help="Penalizes repetitive vocabulary patterns characteristic of template amplification."
                )

        eval_btn = st.button("Evaluate Post Attribution", type="primary", use_container_width=True)

    with c_right:
        active_text = input_text.strip()
        if active_text:
            start_infer_time = time.perf_counter()
            result = analyze_single_post_deterministic(
                text=active_text,
                film_cfg=selected_film_cfg,
                sensitivity=threshold_sensitivity,
                penalize_duplicates=penalize_repetition
            )
            infer_ms = (time.perf_counter() - start_infer_time) * 1000.0

            if result:
                label = result["pred_label"]
                conf = result["confidence"] * 100.0
                anomaly = result["anomaly_index"]

                # Verdict Banner Selection
                banner_class = "verdict-organic"
                if label == "PAID-COMMERCIAL":
                    banner_class = "verdict-paid"
                elif label == "FAN-COORDINATED":
                    banner_class = "verdict-fan"

                st.markdown(f"""
                <div class="verdict-card {banner_class}">
                    <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; opacity: 0.9;">Attributed Motive</div>
                    <div style="font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 2px 0 6px 0;">{label}</div>
                    <div style="display: flex; gap: 20px; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">
                        <span>CONFIDENCE: <strong>{conf:.2f}%</strong></span>
                        <span>ANOMALY LEVEL: <strong>{anomaly}</strong></span>
                        <span>LATENCY: <strong>{infer_ms:.1f}ms</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability Distribution Split
                p_org = result["probabilities"]["Organic"] * 100.0
                p_fan = result["probabilities"]["Fan-Coordinated"] * 100.0
                p_paid = result["probabilities"]["Paid-Commercial"] * 100.0

                prob_df = pd.DataFrame({
                    "Class": ["Organic", "Fan-Coordinated", "Paid-Commercial"],
                    "Probability (%)": [p_org, p_fan, p_paid]
                })

                fig_prob = px.bar(
                    prob_df,
                    x="Probability (%)",
                    y="Class",
                    orientation="h",
                    color="Class",
                    color_discrete_map={
                        "Organic": "#047857",
                        "Fan-Coordinated": "#1D4ED8",
                        "Paid-Commercial": "#B91C1C"
                    },
                    range_x=[0, 100]
                )
                fig_prob.update_layout(
                    height=140,
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False,
                    xaxis=dict(gridcolor="#E2E8F0", zeroline=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # 4-Quadrant Stylometric Score Matrix
                st.markdown("**Underlying Linguistic & Stylometric Indicators**")
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-box-label">Marketing Lexicon Density</div>
                        <div class="metric-box-val">{result['scores']['score_marketing_copy']:.3f}</div>
                        <div class="metric-box-sub">Promotional keyword weight</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-box" style="margin-top: 10px;">
                        <div class="metric-box-label">Reference Specificity</div>
                        <div class="metric-box-val">{result['scores']['score_specificity']:.3f}</div>
                        <div class="metric-box-sub">Entities, craft & plot tokens</div>
                    </div>
                    """, unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-box-label">Fan Identity Ratio</div>
                        <div class="metric-box-val">{result['scores']['score_fan_identity']:.3f}</div>
                        <div class="metric-box-sub">Honorifics & community markers</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-box" style="margin-top: 10px;">
                        <div class="metric-box-label">Absence of Criticism</div>
                        <div class="metric-box-val">{result['scores']['score_criticism_absence']:.3f}</div>
                        <div class="metric-box-sub">Absence of critical variance</div>
                    </div>
                    """, unsafe_allow_html=True)

                # In-Situ Token Highlighting
                st.markdown("**In-Situ Attribution Token Attribution**")
                st.markdown(f"""
                <div class="token-text-container">
                    {result['highlighted_text']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="legend-strip">
                    <span class="legend-item"><span class="legend-indicator" style="background: #DC2626;"></span> Promotional Lexicon</span>
                    <span class="legend-item"><span class="legend-indicator" style="background: #2563EB;"></span> Fan-Army Honorific</span>
                    <span class="legend-item"><span class="legend-indicator" style="background: #059669;"></span> Entity / Craft Specificity</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Input text or select an empirical benchmark vector to execute classification.")

# -----------------------------------------------------------------------------
# 9. DASHBOARD METRIC STRIP (If Pipeline Data Available)
# -----------------------------------------------------------------------------
if "df_labeled" in st.session_state and not st.session_state["df_labeled"].empty:
    df_raw = st.session_state["df_labeled"]

    # Sidebar Filter Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Forensic Filters")

    search_query = st.sidebar.text_input("Filter Text or Author", "")
    motive_filter = st.sidebar.multiselect(
        "Motive Classes",
        ["organic", "fan-coordinated", "paid-commercial"],
        default=["organic", "fan-coordinated", "paid-commercial"]
    )
    sentiment_range = st.sidebar.slider(
        "Sentiment Score Boundary",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05
    )
    hide_noise = st.sidebar.checkbox("Exclude Unclustered Noise (Cluster -1)", value=False)

    df = df_raw.copy()
    if motive_filter:
        df = df[df["motive_label"].isin(motive_filter)]
    if "sentiment_score" in df.columns:
        df = df[(df["sentiment_score"] >= sentiment_range[0]) & (df["sentiment_score"] <= sentiment_range[1])]
    if hide_noise:
        df = df[df["cluster_id"] != -1]
    if search_query:
        mask = (
            df["text"].str.contains(search_query, case=False, na=False) |
            df["author"].str.contains(search_query, case=False, na=False)
        )
        df = df[mask]

    total_reviews = len(df)
    motive_counts = df["motive_label"].value_counts()
    organic_count = motive_counts.get("organic", 0)
    fan_count = motive_counts.get("fan-coordinated", 0)
    paid_count = motive_counts.get("paid-commercial", 0)

    organic_pct = (organic_count / total_reviews) * 100.0 if total_reviews > 0 else 0.0
    fan_pct = (fan_count / total_reviews) * 100.0 if total_reviews > 0 else 0.0
    paid_pct = (paid_count / total_reviews) * 100.0 if total_reviews > 0 else 0.0
    cluster_count = len(df["cluster_id"].unique())

    # Institutional Metric Strip
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-box-label">Corpus Posts</div>
            <div class="metric-box-val">{total_reviews:,}</div>
            <div class="metric-box-sub">{cluster_count} active clusters</div>
        </div>
        <div class="metric-box">
            <div class="metric-box-label">Organic Proportion</div>
            <div class="metric-box-val">{organic_pct:.1f}%</div>
            <div class="metric-box-sub">{organic_count:,} verified posts</div>
        </div>
        <div class="metric-box">
            <div class="metric-box-label">Fan-Coordinated</div>
            <div class="metric-box-val">{fan_pct:.1f}%</div>
            <div class="metric-box-sub">{fan_count:,} amplified posts</div>
        </div>
        <div class="metric-box">
            <div class="metric-box-label">Paid-Commercial</div>
            <div class="metric-box-val" style="color: #B91C1C;">{paid_pct:.1f}%</div>
            <div class="metric-box-sub">{paid_count:,} astroturfed posts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: DISCOURSE OVERVIEW
    # =========================================================================
    with tab_overview:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("#### Motive Attribution Breakdown")
            motive_df = pd.DataFrame({
                "Motive": ["Organic", "Fan-Coordinated", "Paid-Commercial"],
                "Count": [organic_count, fan_count, paid_count]
            })
            fig_pie = px.pie(
                motive_df,
                names="Motive",
                values="Count",
                hole=0.45,
                color="Motive",
                color_discrete_map={
                    "Organic": "#047857",
                    "Fan-Coordinated": "#1D4ED8",
                    "Paid-Commercial": "#B91C1C"
                }
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Volume Timeline & Burst Coordination")
            if "created_utc" in df.columns and not df.empty:
                df_time = df.copy()
                df_time["datetime"] = pd.to_datetime(df_time["created_utc"], unit="s")
                df_time = df_time.sort_values("datetime")
                df_time.set_index("datetime", inplace=True)

                resampled = df_time.resample("12h").agg({"id": "size"}).fillna(0)
                resampled.rename(columns={"id": "Post Count"}, inplace=True)
                resampled.reset_index(inplace=True)

                fig_line = px.line(
                    resampled,
                    x="datetime",
                    y="Post Count",
                    labels={"datetime": "Timeline (12-Hour Windows)", "Post Count": "Posts"}
                )
                fig_line.update_traces(line_color="#1D4ED8")
                fig_line.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    xaxis=dict(gridcolor="#E2E8F0"),
                    yaxis=dict(gridcolor="#E2E8F0")
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Temporal timestamp coordinates unavailable in current corpus.")

        st.markdown("#### Stylometric Feature Profiles Across Attribution Classes")
        score_cols = ["score_marketing_copy", "score_fan_identity", "score_specificity", "score_criticism_absence"]
        existing_cols = [c for c in score_cols if c in df.columns]
        if existing_cols and not df.empty:
            avg_scores = df.groupby("motive_label")[existing_cols].mean().reset_index()
            avg_melted = pd.melt(
                avg_scores,
                id_vars=["motive_label"],
                value_vars=existing_cols,
                var_name="Feature",
                value_name="Average Score"
            )
            avg_melted["Feature"] = avg_melted["Feature"].map({
                "score_marketing_copy": "Marketing Vocabulary",
                "score_fan_identity": "Fan Identity Honorifics",
                "score_specificity": "Reference Specificity",
                "score_criticism_absence": "Absence of Criticism"
            })
            fig_bar = px.bar(
                avg_melted,
                x="Feature",
                y="Average Score",
                color="motive_label",
                barmode="group",
                color_discrete_map={
                    "organic": "#047857",
                    "fan-coordinated": "#1D4ED8",
                    "paid-commercial": "#B91C1C"
                },
                range_y=[0, 1.05]
            )
            fig_bar.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis=dict(gridcolor="#E2E8F0"),
                xaxis=dict(gridcolor="#E2E8F0")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # =========================================================================
    # TAB 3: IMDB ANALYSIS
    # =========================================================================
    with tab_imdb:
        cp_summary = st.session_state.get("cross_platform_summary")
        imdb_info = st.session_state.get("imdb_data")

        if cp_summary and imdb_info:
            st.markdown(f"#### IMDb Ground-Truth Baseline: {imdb_info.get('title')} ({imdb_info.get('imdb_id')})")

            col_meta1, col_meta2 = st.columns([2, 1])
            with col_meta1:
                st.write(f"**Director:** {imdb_info.get('director', 'N/A')}")
                st.write(f"**Cast:** {', '.join(imdb_info.get('cast', []))}")
                st.write(f"**Genres:** {', '.join(imdb_info.get('genres', []))}")
                if imdb_info.get("summary"):
                    st.write(f"**Synopsis:** {imdb_info.get('summary')}")
            with col_meta2:
                st.metric("IMDb Verified Rating", f"{imdb_info.get('rating', 0.0)}/10", f"{imdb_info.get('total_votes', 0):,} verified votes")

            verdict = cp_summary.get("diagnostic_status", "Harmonious")
            severity = cp_summary.get("diagnostic_severity", "Low")
            reasons = cp_summary.get("diagnostic_reasons", [])

            banner_sev = "verdict-organic"
            if severity == "High":
                banner_sev = "verdict-paid"
            elif severity == "Medium":
                banner_sev = "verdict-warn"

            st.markdown(f"""
            <div class="verdict-card {banner_sev}">
                <div style="font-weight: 600; font-size: 0.8rem; text-transform: uppercase;">Cross-Platform Divergence Status: {verdict} (Severity: {severity})</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">{'; '.join(reasons) if reasons else 'No anomalous divergence detected.'}</div>
            </div>
            """, unsafe_allow_html=True)

            im1, im2, im3, im4 = st.columns(4)
            sdi = cp_summary.get("sentiment_divergence_index", 0.0)
            bps = cp_summary.get("rating_distribution_metrics", {}).get("bimodal_polarization_score", 0.0)
            one_pct = cp_summary.get("rating_distribution_metrics", {}).get("one_star_pct", 0.0) * 100
            ten_pct = cp_summary.get("rating_distribution_metrics", {}).get("ten_star_pct", 0.0) * 100

            with im1:
                st.metric("Social Mean Sentiment", f"{cp_summary.get('social_mean_sentiment', 0.5):.2f}/1.0")
            with im2:
                st.metric("Sentiment Divergence Index", f"{sdi:+.2f}", "Gap vs IMDb rating")
            with im3:
                st.metric("Polarization Score", f"{bps:.2f}", "1 Star vs 10 Star variance")
            with im4:
                st.metric("1 Star vs 10 Star Ratio", f"{one_pct:.1f}% / {ten_pct:.1f}%", "Extreme vote shares")

            col_dist, col_comp = st.columns(2)
            with col_dist:
                st.markdown("#### 1-10 Star Rating Distribution")
                dist_data = imdb_info.get("rating_distribution", {})
                if dist_data:
                    dist_df = pd.DataFrame({
                        "Stars": [f"{k} Star" for k in range(1, 11)],
                        "Votes": [dist_data.get(str(k), dist_data.get(k, 0)) for k in range(1, 11)]
                    })
                    fig_dist = px.bar(
                        dist_df,
                        x="Stars",
                        y="Votes",
                        labels={"Stars": "Star Rating", "Votes": "Vote Count"}
                    )
                    fig_dist.update_traces(marker_color="#1D4ED8")
                    fig_dist.update_layout(
                        margin=dict(t=20, b=20, l=20, r=20),
                        yaxis=dict(gridcolor="#E2E8F0")
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            with col_comp:
                st.markdown("#### Cross-Platform Sentiment Contrast")
                comp_df = pd.DataFrame({
                    "Platform": ["X / Social Posts", "IMDb User Reviews", "IMDb Overall Rating"],
                    "Normalized Sentiment": [
                        cp_summary.get("social_mean_sentiment", 0.5),
                        cp_summary.get("imdb_review_mean_sentiment", 0.7),
                        imdb_info.get("rating", 7.0) / 10.0
                    ]
                })
                fig_comp = px.bar(
                    comp_df,
                    x="Platform",
                    y="Normalized Sentiment",
                    color="Platform",
                    color_discrete_sequence=["#1D4ED8", "#047857", "#475569"],
                    range_y=[0, 1.05]
                )
                fig_comp.update_layout(
                    showlegend=False, 
                    margin=dict(t=20, b=20, l=20, r=20),
                    yaxis=dict(gridcolor="#E2E8F0")
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            from collection.imdb_loader import IMDbLoader
            raw_imdb = IMDbLoader().load_movie_data(selected_film_id, config)
            if isinstance(raw_imdb.reviews, pd.DataFrame) and not raw_imdb.reviews.empty:
                st.markdown("#### Verified IMDb User Long-Form Reviews")
                st.dataframe(
                    raw_imdb.reviews[["author", "rating", "title", "text"]],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("Execute pipeline to compute IMDb cross-platform divergence metrics.")

    # =========================================================================
    # TAB 4: EMBEDDINGS
    # =========================================================================
    with tab_embeddings:
        st.markdown("#### 2D MuRIL Dense Vector Semantic Projection")
        st.markdown(
            "<p style='color: #475569; font-size: 0.875rem;'>"
            "Principal Component Analysis (PCA) projection of 768-dimensional MuRIL embeddings. "
            "Dense clusters represent templated or coordinated message dissemination."
            "</p>",
            unsafe_allow_html=True
        )

        df_emb = st.session_state.get("df_embeddings")
        if df_emb is not None:
            emb_cols = [str(i) for i in range(768) if str(i) in df_emb.columns]
            if len(emb_cols) == 768:
                df_merged = df.merge(df_emb[["id"] + emb_cols], on="id", how="inner")
                if not df_merged.empty:
                    X_emb = df_merged[emb_cols].values
                    pca = PCA(n_components=2, random_state=42)
                    coords = pca.fit_transform(X_emb)

                    df_merged["Component 1"] = coords[:, 0]
                    df_merged["Component 2"] = coords[:, 1]

                    fig_scatter = px.scatter(
                        df_merged,
                        x="Component 1",
                        y="Component 2",
                        color="motive_label",
                        symbol="motive_label",
                        hover_data=["author", "text"],
                        color_discrete_map={
                            "organic": "#047857",
                            "fan-coordinated": "#1D4ED8",
                            "paid-commercial": "#B91C1C"
                        }
                    )
                    fig_scatter.update_traces(marker=dict(size=7, opacity=0.8))
                    fig_scatter.update_layout(
                        margin=dict(t=20, b=20, l=20, r=20),
                        xaxis=dict(gridcolor="#E2E8F0"),
                        yaxis=dict(gridcolor="#E2E8F0")
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("No matching vectors found for current filter subset.")
            else:
                st.warning("Embedding dimensional matrix mismatch.")
        else:
            st.info("Embedding matrix not located. Execute pipeline to calculate dense vector embeddings.")

    # =========================================================================
    # TAB 5: SHAP EXPLAINER
    # =========================================================================
    with tab_explainers:
        st.markdown("#### Global Feature Attribution (SHAP Surrogate Model)")
        st.markdown(
            "<p style='color: #475569; font-size: 0.875rem;'>"
            "Mean absolute SHAP value distributions quantifying feature importance in the motive classification pipeline."
            "</p>",
            unsafe_allow_html=True
        )

        shap_data = st.session_state.get("shap_data")
        if shap_data and "shap_values" in shap_data:
            feature_names = shap_data.get("feature_names", ["marketing_copy", "fan_identity", "specificity", "criticism_absence"])
            shap_values = np.array(shap_data["shap_values"])

            display_names = {
                "marketing_copy": "Marketing Vocabulary Density",
                "fan_identity": "Fan Identity Honorifics",
                "specificity": "Reference Specificity (NER / Cast)",
                "criticism_absence": "Absence of Criticism (Sentiment Uniformity)"
            }
            readable_feats = [display_names.get(name, name) for name in feature_names]

            num_feats = len(feature_names)
            if shap_values.ndim == 1:
                global_importance = np.abs(shap_values)
            elif shap_values.ndim == 2:
                global_importance = np.mean(np.abs(shap_values), axis=0) if shap_values.shape[1] == num_feats else np.mean(np.abs(shap_values), axis=1)
            elif shap_values.ndim == 3:
                global_importance = np.mean(np.abs(shap_values), axis=(0, 1)) if shap_values.shape[2] == num_feats else np.mean(np.abs(shap_values), axis=(0, 2))
            else:
                global_importance = np.zeros(num_feats)

            fig_shap = px.bar(
                x=global_importance,
                y=readable_feats,
                orientation="h",
                labels={"x": "Mean Absolute SHAP Value", "y": "Feature"},
                color_discrete_sequence=["#1D4ED8"]
            )
            fig_shap.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(gridcolor="#E2E8F0")
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("Execute pipeline to calculate SHAP global attribution weights.")

        with st.expander("Methodology & Mathematical Formulas"):
            st.markdown("""
            - **Marketing Vocabulary Density:** Ratio of TF-IDF feature weights corresponding to commercial and promotional lexicons.
            - **Fan Identity Markers:** Density of star honorifics and community slogans matching curated code-mixed dictionaries.
            - **Reference Specificity:** Density of verified IMDb named entities, plot sequences, and technical filmmaking terminology.
            - **Absence of Criticism:** Variance of sentiment valence across cluster members. Low variance indicates orchestrated consensus.
            """)

    # =========================================================================
    # TAB 6: EXPLORER
    # =========================================================================
    with tab_explorer:
        st.markdown("#### Coordinated Cluster & Post Drilldown")
        clusters = sorted([c for c in df["cluster_id"].unique() if c != -1])
        if not clusters:
            st.info("No coordinated clusters identified under active filter constraints.")
        else:
            selected_cluster = st.selectbox("Select Cluster Identifier", clusters, format_func=lambda x: f"Cluster {x}")
            if selected_cluster is not None:
                c_df = df[df["cluster_id"] == selected_cluster]
                c_motive = c_df["motive_label"].iloc[0]
                c_topic = c_df["cluster_topic"].iloc[0] if "cluster_topic" in c_df.columns else "N/A"
                c_rationale = c_df["motive_rationale"].iloc[0] if "motive_rationale" in c_df.columns else "Threshold decision."

                st.write(f"**Classification:** `{c_motive.upper()}` ({len(c_df)} posts)")
                st.write(f"**Cluster Keywords:** {c_topic}")
                st.write(f"**Attribution Rationale:** {c_rationale}")

                st.dataframe(
                    c_df[["author", "text", "score_marketing_copy", "score_fan_identity", "score_specificity"]],
                    use_container_width=True,
                    hide_index=True
                )

        st.markdown("#### Filtered Discourse Corpus")
        st.dataframe(
            df[["id", "author", "text", "motive_label", "cluster_id", "score_marketing_copy", "score_fan_identity", "score_specificity"]],
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(index=False).encode('utf-8')
        json_data = df.to_json(orient="records", indent=2).encode('utf-8')

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Export Filtered Subset (CSV)",
                data=csv_data,
                file_name=f"beyondbots_{selected_film_id}_corpus.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="Export Filtered Subset (JSON)",
                data=json_data,
                file_name=f"beyondbots_{selected_film_id}_corpus.json",
                mime="application/json",
                use_container_width=True
            )

    # =========================================================================
    # TAB 7: BENCHMARK
    # =========================================================================
    with tab_benchmark:
        st.markdown("#### Cross-Film Coordination Matrix")
        st.markdown(
            "<p style='color: #475569; font-size: 0.875rem;'>"
            "Comparative analysis of coordination ratios, IMDb verified scores, and sentiment divergence across Indian releases."
            "</p>",
            unsafe_allow_html=True
        )

        comparison_data = []
        for f in films:
            f_id = f["id"]
            f_title = f["title"]
            cp_f = f"data/labeled/{f_id}_cross_platform.json"
            labeled_f = f"data/labeled/{f_id}_labeled.parquet"

            if os.path.exists(cp_f) and os.path.exists(labeled_f):
                with open(cp_f, "r", encoding="utf-8") as jf:
                    cp_data = json.load(jf)
                f_df = pd.read_parquet(labeled_f)

                tot = len(f_df)
                counts = f_df["motive_label"].value_counts()
                org_p = (counts.get("organic", 0) / tot) * 100.0 if tot > 0 else 0.0
                fan_p = (counts.get("fan-coordinated", 0) / tot) * 100.0 if tot > 0 else 0.0
                paid_p = (counts.get("paid-commercial", 0) / tot) * 100.0 if tot > 0 else 0.0

                comparison_data.append({
                    "Release": f_title,
                    "Total Corpus": tot,
                    "Organic %": round(org_p, 1),
                    "Fan %": round(fan_p, 1),
                    "Paid PR %": round(paid_p, 1),
                    "IMDb Rating": cp_data.get("imdb_rating", 7.0),
                    "Polarization Index": cp_data.get("rating_distribution_metrics", {}).get("bimodal_polarization_score", 0.0),
                    "Status": cp_data.get("diagnostic_status", "Harmonious")
                })

        if comparison_data:
            comp_table = pd.DataFrame(comparison_data)
            st.dataframe(comp_table, use_container_width=True, hide_index=True)

            c_bench1, c_bench2 = st.columns(2)
            with c_bench1:
                st.markdown("#### Motive Breakdown Across Releases")
                melted_motives = pd.melt(
                    comp_table,
                    id_vars=["Release"],
                    value_vars=["Organic %", "Fan %", "Paid PR %"],
                    var_name="Attribution",
                    value_name="Percentage"
                )
                fig_bench_motives = px.bar(
                    melted_motives,
                    x="Release",
                    y="Percentage",
                    color="Attribution",
                    barmode="stack",
                    color_discrete_map={
                        "Organic %": "#047857",
                        "Fan %": "#1D4ED8",
                        "Paid PR %": "#B91C1C"
                    }
                )
                fig_bench_motives.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    yaxis=dict(gridcolor="#E2E8F0")
                )
                st.plotly_chart(fig_bench_motives, use_container_width=True)

            with c_bench2:
                st.markdown("#### IMDb Rating vs. Polarization Index")
                fig_pol = px.scatter(
                    comp_table,
                    x="IMDb Rating",
                    y="Polarization Index",
                    color="Status",
                    size="Total Corpus",
                    text="Release",
                    range_x=[5.0, 9.0],
                    range_y=[0.0, 1.1]
                )
                fig_pol.update_traces(textposition="top center")
                fig_pol.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    xaxis=dict(gridcolor="#E2E8F0"),
                    yaxis=dict(gridcolor="#E2E8F0")
                )
                st.plotly_chart(fig_pol, use_container_width=True)
        else:
            st.info("Execute pipeline across other film releases to populate cross-film comparison matrix.")
else:
    # If full pipeline hasn't loaded yet, show informational prompt for the remaining tabs
    with tab_overview:
        st.info("Select a film release and click 'Execute Pipeline' in the sidebar to populate discourse metrics.")
    with tab_imdb:
        st.info("Select a film release and click 'Execute Pipeline' in the sidebar to view IMDb cross-platform divergence.")
    with tab_embeddings:
        st.info("Select a film release and click 'Execute Pipeline' in the sidebar to view MuRIL vector projections.")
    with tab_explainers:
        st.info("Select a film release and click 'Execute Pipeline' in the sidebar to inspect SHAP feature importance.")
    with tab_explorer:
        st.info("Select a film release and click 'Execute Pipeline' in the sidebar to explore cluster posts.")
    with tab_benchmark:
        st.info("Select a film release and click 'Execute Pipeline' in the sidebar to inspect the benchmark matrix.")
