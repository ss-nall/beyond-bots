import os
import sys
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any

# Ensure src directory and project root are in python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collection.collector import XCollector
from collection.imdb_loader import IMDbLoader, IMDbMovieData
from preprocessing.cleaner import preprocess_dataframe
from representation.vectorizer import SparseVectorizer, DenseVectorizer, save_representations
from clustering.clusterer import run_clustering_stage
from sentiment.analyzer import analyze_sentiment
from sentiment.imdb_comparator import IMDbCrossPlatformComparator
from topic_modeling.topic_modeler import compute_cluster_topics
from feature_engineering.scorers import compute_features
from classification.classifier import classify_dataframe_clusters
from validation.validator import CaseStudyValidator

def run_pipeline(film_id: str, config_path: str = "config.yaml", override_input_file: str = None, progress_callback: Any = None) -> pd.DataFrame:
    """
    Runs the complete 7-stage BeyondBots text mining pipeline end to end with IMDb cross-platform integration.
    Saves outputs and prints summary.
    """
    def update_progress(stage: int, name: str, pct: float, msg: str):
        if progress_callback:
            progress_callback(stage, name, pct, msg)
        print(f"[{name}] {msg}")

    update_progress(0, "Collection", 0.05, f"Starting BeyondBots Text Mining Pipeline for: {film_id}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if override_input_file is not None:
        config["collection_settings"]["input_file"] = override_input_file
        
    film_config = next((f for f in config.get("films", []) if f["id"] == film_id), None)
    if not film_config:
        raise ValueError(f"Film ID '{film_id}' not found in configuration.")
        
    # Stage 0a: Ingest IMDb Ground-Truth & Movie Metadata
    update_progress(0, "IMDb Ingestion", 0.08, "Loading IMDb ground-truth metadata, rating curves, and reviews...")
    imdb_loader = IMDbLoader()
    imdb_data: IMDbMovieData = imdb_loader.load_movie_data(film_id, config)
    imdb_entities = imdb_data.get_entity_lexicon()
    update_progress(0, "IMDb Ingestion", 0.10, f"Loaded IMDb data for '{imdb_data.title}' (Rating: {imdb_data.rating}/10, Entities: {len(imdb_entities)}).")

    # Stage 0b: Collection (Social Media Posts)
    update_progress(0, "Collection", 0.12, "Initializing social media data collector...")
    collector = XCollector(config)
    df_raw = collector.collect_data(film_id)
    update_progress(0, "Collection", 0.15, f"Collected {len(df_raw)} raw tweets/replies.")
    if df_raw.empty:
        update_progress(0, "Collection", 0.15, "Error: No data collected. Exiting.")
        return df_raw
        
    # Stage 1: Preprocessing
    update_progress(1, "Preprocessing", 0.25, "Cleaning, tokenizing, and normalizing texts...")
    df_preprocessed = preprocess_dataframe(df_raw, text_column="text")
    update_progress(1, "Preprocessing", 0.35, "Stage 1: Preprocessing complete.")
    
    # Stage 2: Representation
    update_progress(2, "Representation", 0.40, "Generating TF-IDF sparse and MuRIL dense vector representations...")
    sparse_vec = SparseVectorizer()
    sparse_matrix = sparse_vec.fit_transform(df_preprocessed["cleaned_text"].tolist())
    
    dense_vec = DenseVectorizer(config["model_names"]["dense_embeddings"])
    dense_embeddings = dense_vec.embed_texts(df_preprocessed["cleaned_text"].tolist())
    
    # Persist Stage 2 representations
    save_representations(df_preprocessed, sparse_matrix, dense_embeddings, "./data/processed")
    update_progress(2, "Representation", 0.50, "Stage 2: Representation complete. Embeddings saved.")
    
    # Stage 3: Clustering (near-duplicate + burst + community)
    update_progress(3, "Clustering", 0.55, "Detecting near-duplicate document clusters, temporal spikes, and reply communities...")
    df_clustered = run_clustering_stage(df_preprocessed, dense_embeddings, film_config["release_date"])
    update_progress(3, "Clustering", 0.60, "Stage 3: Clustering complete.")
    
    # Stage 6: Sentiment (run before features to feed criticism-absence)
    update_progress(4, "Sentiment Analysis", 0.65, "Computing post-level sentiment polarity scores...")
    df_sentiment = analyze_sentiment(df_clustered)
    update_progress(4, "Sentiment Analysis", 0.70, "Stage 6: Sentiment analysis complete.")
    
    # Stage 5: Topic Modeling (run before features to assist summaries)
    update_progress(5, "Topic Modeling", 0.75, "Running BERTopic / LDA per cluster to extract keywords...")
    df_topics = compute_cluster_topics(df_sentiment, dense_embeddings)
    update_progress(5, "Topic Modeling", 0.80, "Stage 5: Topic modeling complete.")
    
    # Stage 4: Feature Engineering with Dynamic IMDb Entity Grounding
    update_progress(6, "Feature Engineering", 0.85, "Extracting features with dynamic IMDb entity grounding...")
    df_features = compute_features(
        df_topics,
        sparse_matrix,
        sparse_vec.get_feature_names(),
        film_config["promotional_keywords"],
        film_config["fan_markers"],
        sentiment_col="sentiment_score",
        imdb_entities=imdb_entities
    )
    update_progress(6, "Feature Engineering", 0.90, "Stage 4: Feature engineering complete.")
    
    # Stage 7: Classification & Explainability
    update_progress(7, "Classification & SHAP", 0.93, "Classifying cluster motive and calculating SHAP values...")
    df_classified, shap_data = classify_dataframe_clusters(df_features)
    update_progress(7, "Classification & SHAP", 0.96, "Stage 7: Classification and SHAP calculations complete.")
    
    # Cross-Platform Sentiment & Anomaly Comparison
    update_progress(8, "Cross-Platform Analysis", 0.97, "Analyzing social discourse vs. IMDb rating curves & reviews...")
    cross_platform_report = IMDbCrossPlatformComparator.compare_discourse(df_classified, imdb_data)
    
    # Stage 8: Validation
    update_progress(8, "Validation", 0.98, "Validating flagged clusters against documented case studies...")
    validator = CaseStudyValidator()
    validation_res = validator.validate(df_classified)
    
    # Persist Final Output Labeled Data (both film-specific and default)
    output_dir = "./data/labeled"
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, f"{film_id}_labeled.parquet")
    df_classified.to_parquet(parquet_path, index=False)
    # Also save fallback default
    df_classified.to_parquet(os.path.join(output_dir, "labeled_output.parquet"), index=False)
    
    # Persist SHAP output & Cross-Platform JSON
    import json
    
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return make_serializable(obj.tolist())
        return obj

    shap_path = os.path.join(output_dir, f"{film_id}_shap.json")
    with open(shap_path, "w") as f:
        json.dump(make_serializable(shap_data), f)
    with open(os.path.join(output_dir, "shap_data.json"), "w") as f:
        json.dump(make_serializable(shap_data), f)
        
    cross_platform_path = os.path.join(output_dir, f"{film_id}_cross_platform.json")
    with open(cross_platform_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(cross_platform_report), f, indent=2)
    with open(os.path.join(output_dir, "cross_platform_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_serializable(cross_platform_report), f, indent=2)

    imdb_meta_path = os.path.join(output_dir, f"{film_id}_imdb.json")
    with open(imdb_meta_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(imdb_data.to_dict()), f, indent=2)
    with open(os.path.join(output_dir, "imdb_data.json"), "w", encoding="utf-8") as f:
        json.dump(make_serializable(imdb_data.to_dict()), f, indent=2)
        
    update_progress(8, "Complete", 1.00, f"Pipeline finished. Outputs saved to: {output_dir}")
    
    # Summary report printout
    print(f"\n================ PIPELINE SUMMARY ================")
    print(f"Film: {imdb_data.title} ({imdb_data.imdb_id}) | IMDb Rating: {imdb_data.rating}/10 ({imdb_data.total_votes} votes)")
    print(f"Total social posts processed: {len(df_classified)}")
    print(f"Cross-Platform Status: {cross_platform_report['diagnostic_status']} (Severity: {cross_platform_report['diagnostic_severity']})")
    print(f"Sentiment Divergence Index (SDI): {cross_platform_report['sentiment_divergence_index']:+.3f}")
    print(f"IMDb Polarization Score: {cross_platform_report['rating_distribution_metrics']['bimodal_polarization_score']:.3f}")
    print(f"\nMotive breakdown of tweets:")
    motive_counts = df_classified["motive_label"].value_counts()
    for motive, count in motive_counts.items():
        print(f"  - {motive}: {count} tweets")
        
    print(f"\nCluster details:")
    for c_id, group in df_classified.groupby("cluster_id"):
        topic = group["cluster_topic"].iloc[0] if "cluster_topic" in group.columns else "N/A"
        label = group["motive_label"].iloc[0]
        size = len(group)
        print(f"  * Cluster {c_id:2d} | Size: {size:2d} | Label: {label:15s} | Topic: {topic}")
        
    print(f"==================================================")
    return df_classified

if __name__ == "__main__":
    # Run pipeline on Animal
    os.environ["BEYONDBOTS_MOCK_EMBEDDINGS"] = "1"
    os.environ["BEYONDBOTS_MOCK_SENTIMENT"] = "1"
    run_pipeline("animal_2023")
