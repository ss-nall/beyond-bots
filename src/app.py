import os
import tempfile
import time
import json
import pandas as pd
import numpy as np
import yaml
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from pipeline import run_pipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BeyondBots - Film Discourse Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. LOAD CONFIGURATION
# -----------------------------------------------------------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# -----------------------------------------------------------------------------
# 3. CLEAN, ROBUST STYLING (Respects Native Theme & Ensures Readability)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean typography & spacing */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. APP HEADER
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>BeyondBots: Film Discourse Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='main-subtitle'>Multi-stage NLP pipeline for detecting coordinated, fan-driven, and organic discourse around Indian cinema.</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. SIDEBAR: DATA & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration & Data")

mode = st.sidebar.radio(
    "Data Source Mode",
    ["High-Volume Datasets (1,000+ Posts)", "Upload Custom File"],
    index=0
)

films = config.get("films", [])
film_options = {f["title"]: f["id"] for f in films}
selected_film_title = st.sidebar.selectbox("Select Target Film", list(film_options.keys()))
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
        "Upload File (.csv, .txt, .pdf, .docx)",
        type=["csv", "txt", "pdf", "docx", "doc"]
    )
    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            override_path = tmp.name

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run Full Pipeline", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 6. PIPELINE EXECUTION & DYNAMIC FILM DATA LOADER
# -----------------------------------------------------------------------------
if run_btn:
    progress_bar = st.sidebar.progress(0.0)
    status_box = st.sidebar.empty()

    def update_progress(stage: int, name: str, pct: float, msg: str):
        progress_bar.progress(min(max(pct, 0.0), 1.0))
        status_box.text(f"Stage {stage} [{name}]: {msg}")

    with st.spinner(f"Analyzing discourse for {selected_film_title}..."):
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

            st.sidebar.success("Pipeline finished successfully.")
        except Exception as e:
            st.sidebar.error(f"Execution failed: {e}")
        finally:
            if mode == "Upload Custom File" and override_path and os.path.exists(override_path):
                try:
                    os.remove(override_path)
                except Exception:
                    pass

# Dynamic per-film data loading (instant update when dropdown changes)
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
# 7. DASHBOARD DISPLAY & DYNAMIC FILTERS
# -----------------------------------------------------------------------------
if "df_labeled" in st.session_state and not st.session_state["df_labeled"].empty:
    df_raw = st.session_state["df_labeled"]

    # Interactive Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.header("Interactive Filters")

    search_query = st.sidebar.text_input("Search Text or Author", "")
    
    motive_filter = st.sidebar.multiselect(
        "Filter Motives",
        ["organic", "fan-coordinated", "paid-commercial"],
        default=["organic", "fan-coordinated", "paid-commercial"]
    )
    
    sentiment_range = st.sidebar.slider(
        "Sentiment Score Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05
    )
    
    hide_noise = st.sidebar.checkbox("Exclude Unclustered Noise (Cluster -1)", value=False)

    # Filter data dynamically
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

    # Metrics
    total_reviews = len(df)
    motive_counts = df["motive_label"].value_counts()
    organic_count = motive_counts.get("organic", 0)
    fan_count = motive_counts.get("fan-coordinated", 0)
    paid_count = motive_counts.get("paid-commercial", 0)

    organic_pct = (organic_count / total_reviews) * 100 if total_reviews > 0 else 0
    fan_pct = (fan_count / total_reviews) * 100 if total_reviews > 0 else 0
    paid_pct = (paid_count / total_reviews) * 100 if total_reviews > 0 else 0

    # Top Metric Cards (Native Streamlit)
    st.markdown(f"### Current Film: **{selected_film_title}**")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Analyzed Posts", f"{total_reviews:,}", f"{len(df['cluster_id'].unique())} clusters")
    with m2:
        st.metric("Organic Posts", f"{organic_pct:.1f}%", f"{organic_count} posts")
    with m3:
        st.metric("Fan-Coordinated", f"{fan_pct:.1f}%", f"{fan_count} posts")
    with m4:
        st.metric("Paid-Commercial", f"{paid_pct:.1f}%", f"{paid_count} posts")

    st.markdown("---")

    # Tabs
    tab_overview, tab_imdb, tab_embeddings, tab_explainers, tab_explorer, tab_benchmark = st.tabs([
        "Discourse Overview", 
        "IMDb Cross-Platform Analysis",
        "Semantic Embedding Space", 
        "Feature Importance & SHAP",
        "Cluster & Post Explorer",
        "Cross-Film Benchmark Comparison"
    ])

    # ------------------ TAB 1: OVERVIEW ------------------
    with tab_overview:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("Motive Breakdown")
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
                    "Organic": "#2ca02c", 
                    "Fan-Coordinated": "#1f77b4", 
                    "Paid-Commercial": "#d62728"
                }
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("Volume Timeline & Burst Activity")
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
                fig_line.update_traces(line_color="#1f77b4")
                fig_line.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No temporal data available for volume timeline.")

        st.subheader("Linguistic & Stylometric Score Profiles")
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
                    "organic": "#2ca02c", 
                    "fan-coordinated": "#1f77b4", 
                    "paid-commercial": "#d62728"
                },
                range_y=[0, 1.05]
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------ TAB 2: IMDB ANALYSIS ------------------
    with tab_imdb:
        cp_summary = st.session_state.get("cross_platform_summary")
        imdb_info = st.session_state.get("imdb_data")

        if cp_summary and imdb_info:
            st.subheader(f"IMDb Ground-Truth: {imdb_info.get('title')} ({imdb_info.get('imdb_id')})")
            
            # Film metadata info
            col_meta1, col_meta2 = st.columns([2, 1])
            with col_meta1:
                st.write(f"**Director:** {imdb_info.get('director', 'N/A')}")
                st.write(f"**Cast:** {', '.join(imdb_info.get('cast', []))}")
                st.write(f"**Genres:** {', '.join(imdb_info.get('genres', []))}")
                if imdb_info.get("summary"):
                    st.write(f"**Synopsis:** {imdb_info.get('summary')}")
            with col_meta2:
                st.metric("IMDb Verified Rating", f"{imdb_info.get('rating', 0.0)}/10", f"{imdb_info.get('total_votes', 0):,} votes")

            # Cross-Platform Verdict
            verdict = cp_summary.get("diagnostic_status", "Harmonious")
            severity = cp_summary.get("diagnostic_severity", "Low")
            reasons = cp_summary.get("diagnostic_reasons", [])

            if severity == "High":
                st.error(f"Cross-Platform Verdict: {verdict} (Severity: High)")
            elif severity == "Medium":
                st.warning(f"Cross-Platform Verdict: {verdict} (Severity: Medium)")
            else:
                st.success(f"Cross-Platform Verdict: {verdict} (Severity: Low)")

            for r in reasons:
                st.write(f"- {r}")

            # Cross-Platform Metric Cards
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
                st.metric("Polarization Score", f"{bps:.2f}", "1★ vs 10★ variance")
            with im4:
                st.metric("1★ vs 10★ Share", f"{one_pct:.1f}% / {ten_pct:.1f}%", "Extreme vote ratios")

            col_dist, col_comp = st.columns(2)
            with col_dist:
                st.subheader("1-10 Star Vote Distribution")
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
                    fig_dist.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_dist, use_container_width=True)

            with col_comp:
                st.subheader("Platform Sentiment Comparison")
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
                    range_y=[0, 1.05]
                )
                fig_comp.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_comp, use_container_width=True)

            # Verified Reviews Table
            from collection.imdb_loader import IMDbLoader
            raw_imdb = IMDbLoader().load_movie_data(selected_film_id, config)
            if isinstance(raw_imdb.reviews, pd.DataFrame) and not raw_imdb.reviews.empty:
                st.subheader("Verified IMDb User Reviews")
                st.dataframe(
                    raw_imdb.reviews[["author", "rating", "title", "text"]],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("Run the pipeline to generate IMDb cross-platform analysis.")

    # ------------------ TAB 3: EMBEDDINGS ------------------
    with tab_embeddings:
        st.subheader("2D MuRIL Dense Vector Semantic Projection")
        st.write("2D PCA projection of the 768-dimensional MuRIL embeddings. Posts with identical or templated phrasing group closely together.")
        
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
                            "organic": "#2ca02c", 
                            "fan-coordinated": "#1f77b4", 
                            "paid-commercial": "#d62728"
                        }
                    )
                    fig_scatter.update_traces(marker=dict(size=8, opacity=0.85))
                    fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("No matching embeddings for the current filter subset.")
            else:
                st.warning("Embedding dimensions mismatch.")
        else:
            st.info("Embedding file not found. Run pipeline to compute dense embeddings.")

    # ------------------ TAB 4: SHAP EXPLAINER ------------------
    with tab_explainers:
        st.subheader("Global Feature Importance (SHAP Surrogate Model)")
        st.write("Measures how each linguistic feature influences the classifier's motive assignment.")
        
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
                labels={"x": "Mean Absolute SHAP Value", "y": "Feature"}
            )
            fig_shap.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("Run the pipeline to calculate SHAP values.")

        with st.expander("Feature Methodology Details"):
            st.markdown("""
            - **Marketing Vocabulary Density:** Evaluates the TF-IDF weight of promotional vocabulary (e.g. *blockbuster, masterpiece, box office rampage*).
            - **Fan Identity Markers:** Matches star honorifics and fan club markers (e.g. *King Khan, Superstar, Rebel Star*).
            - **Reference Specificity:** Measures specific mentions of verified IMDb cast, directors, plot sequences, and named entities.
            - **Absence of Criticism:** Evaluates post-level sentiment variance within a cluster. Zero variance indicates uniform praise.
            """)

    # ------------------ TAB 5: EXPLORER ------------------
    with tab_explorer:
        st.subheader("Cluster Inspector")
        clusters = sorted([c for c in df["cluster_id"].unique() if c != -1])
        if not clusters:
            st.info("No coordinated clusters active under current filter settings.")
        else:
            selected_cluster = st.selectbox("Select Cluster ID", clusters, format_func=lambda x: f"Cluster {x}")
            if selected_cluster is not None:
                c_df = df[df["cluster_id"] == selected_cluster]
                c_motive = c_df["motive_label"].iloc[0]
                c_topic = c_df["cluster_topic"].iloc[0] if "cluster_topic" in c_df.columns else "N/A"
                c_rationale = c_df["motive_rationale"].iloc[0] if "motive_rationale" in c_df.columns else "Threshold decision."
                
                st.write(f"**Classification:** `{c_motive}` ({len(c_df)} posts)")
                st.write(f"**Topic Keywords:** {c_topic}")
                st.write(f"**Classification Rationale:** {c_rationale}")
                
                st.dataframe(
                    c_df[["author", "text", "score_marketing_copy", "score_fan_identity", "score_specificity"]],
                    use_container_width=True,
                    hide_index=True
                )

        st.subheader("All Filtered Posts")
        st.dataframe(
            df[["id", "author", "text", "motive_label", "cluster_id", "score_marketing_copy", "score_fan_identity", "score_specificity"]],
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"beyondbots_{selected_film_id}_data.csv",
            mime="text/csv"
        )

    # ------------------ TAB 6: CROSS-FILM BENCHMARK COMPARISON ------------------
    with tab_benchmark:
        st.subheader("Cross-Film Intelligence Comparison")
        st.write("Compare coordination levels, IMDb ratings, and sentiment divergence across all 7 Indian film releases.")

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
                org_p = (counts.get("organic", 0) / tot) * 100 if tot > 0 else 0
                fan_p = (counts.get("fan-coordinated", 0) / tot) * 100 if tot > 0 else 0
                paid_p = (counts.get("paid-commercial", 0) / tot) * 100 if tot > 0 else 0

                comparison_data.append({
                    "Film": f_title,
                    "Total Posts": tot,
                    "Organic %": round(org_p, 1),
                    "Fan %": round(fan_p, 1),
                    "PR %": round(paid_p, 1),
                    "IMDb Rating": cp_data.get("imdb_rating", 7.0),
                    "Social Sentiment": cp_data.get("social_mean_sentiment", 0.5),
                    "Polarization Score": cp_data.get("rating_distribution_metrics", {}).get("bimodal_polarization_score", 0.0),
                    "Verdict": cp_data.get("diagnostic_status", "Harmonious")
                })

        if comparison_data:
            comp_table = pd.DataFrame(comparison_data)
            st.dataframe(comp_table, use_container_width=True, hide_index=True)

            c_bench1, c_bench2 = st.columns(2)
            with c_bench1:
                st.subheader("Motive Breakdown Across Films")
                melted_motives = pd.melt(
                    comp_table,
                    id_vars=["Film"],
                    value_vars=["Organic %", "Fan %", "PR %"],
                    var_name="Category",
                    value_name="Percentage"
                )
                fig_bench_motives = px.bar(
                    melted_motives,
                    x="Film",
                    y="Percentage",
                    color="Category",
                    barmode="stack",
                    color_discrete_map={"Organic %": "#2ca02c", "Fan %": "#1f77b4", "PR %": "#d62728"}
                )
                fig_bench_motives.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_bench_motives, use_container_width=True)

            with c_bench2:
                st.subheader("IMDb Rating vs. Polarization Index")
                fig_pol = px.scatter(
                    comp_table,
                    x="IMDb Rating",
                    y="Polarization Score",
                    color="Verdict",
                    size="Total Posts",
                    text="Film",
                    range_x=[5.0, 8.5],
                    range_y=[0.0, 1.1]
                )
                fig_pol.update_traces(textposition="top center")
                fig_pol.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_pol, use_container_width=True)
        else:
            st.info("Run the pipeline across other films to populate the cross-film comparison matrix.")
else:
    st.info("Select a film and click 'Run Full Pipeline' in the sidebar to view the dashboard.")
