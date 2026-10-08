import os
import sys
import pytest
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from feature_engineering.scorers import (
    MarketingCopyScorer,
    FanIdentityScorer,
    CriticismAbsenceScorer,
    SpecificityOfReferenceScorer
)
from sentiment.analyzer import MultilingualSentimentAnalyzer
from topic_modeling.topic_modeler import ClusterTopicModeler
from pipeline import run_pipeline

# Configure testing environment to use mock embeddings & sentiment
os.environ["BEYONDBOTS_MOCK_EMBEDDINGS"] = "1"
os.environ["BEYONDBOTS_MOCK_SENTIMENT"] = "1"

# ----------------- Unit Tests for Stage 4 Scorers -----------------

def test_marketing_copy_scorer():
    promotional_lexicon = ["blockbuster", "masterpiece", "sensational"]
    scorer = MarketingCopyScorer(promotional_lexicon)
    
    feature_names = ["other", "blockbuster", "film", "masterpiece"]
    tfidf_vector = np.array([0.1, 0.4, 0.1, 0.4])
    
    score = scorer.score(tfidf_vector, feature_names)
    # Total promo weight = tfidf[1] (0.4) + tfidf[3] (0.4) = 0.8
    # Total weight = 1.0
    # Expected score = 0.8
    assert abs(score - 0.8) < 1e-5

def test_fan_identity_scorer():
    fan_lexicon = ["srk", "king khan", "badshah"]
    scorer = FanIdentityScorer(fan_lexicon)
    
    tokens = ["this", "is", "srk", "king", "khan", "movie"]
    score = scorer.score(tokens)
    # The lexicon expands "king khan" -> "king", "khan"
    # Matches: "srk", "king", "khan" (3 matches)
    # Total tokens = 6
    # Expected score = 3/6 = 0.5
    assert abs(score - 0.5) < 1e-5

def test_criticism_absence_scorer():
    scorer = CriticismAbsenceScorer()
    
    # Low variance (all positive)
    scores_uniform = [0.95, 0.96, 0.94, 0.95]
    score_uniform = scorer.score_cluster(scores_uniform)
    assert score_uniform > 0.9
    
    # High variance (mixed opinion)
    scores_mixed = [0.1, 0.9, 0.2, 0.8]
    score_mixed = scorer.score_cluster(scores_mixed)
    assert score_mixed < 0.6

def test_specificity_of_reference_scorer():
    scorer = SpecificityOfReferenceScorer()
    
    text_specific = "We went to Galaxy cinema to watch Jawan. Screen 1 was full."
    tokens = ["We", "went", "to", "Galaxy", "cinema", "to", "watch", "Jawan", ".", "Screen", "1", "was", "full"]
    score_spec = scorer.score(text_specific, tokens)
    
    text_generic = "nice movie nice film good watch masterpiece"
    tokens_generic = ["nice", "movie", "nice", "film", "good", "watch", "masterpiece"]
    score_gen = scorer.score(text_generic, tokens_generic)
    
    assert score_spec > score_gen

# ----------------- Unit Tests for Stage 5/6 -----------------

def test_topic_modeling_and_sentiment():
    analyzer = MultilingualSentimentAnalyzer()
    modeler = ClusterTopicModeler()
    
    # Check sentiment analysis
    sent_score = analyzer.score_post("This film was amazing! Pure masterpiece!")
    assert sent_score > 0.7
    
    # Check topic summary
    texts = [
        "Jawan metro hijack scene",
        "Jawan metro hijack scene explanation",
        "Explain metro hijack scene in Jawan",
        "metro scene was good in Jawan",
        "the hijack metro train scene Jawan"
    ]
    df_cluster = pd.DataFrame({"cleaned_text": texts})
    topic = modeler.get_cluster_topic_summary(df_cluster)
    assert "metro" in topic or "hijack" in topic or "scene" in topic

# ----------------- Integration Test -----------------

def test_pipeline_integration():
    # Run the full pipeline
    output_df = run_pipeline("jawan_2023", "config.yaml", override_input_file="")
    
    assert not output_df.empty
    assert "motive_label" in output_df.columns
    assert "cluster_id" in output_df.columns
    
    # Verify the output parquet was saved
    assert os.path.exists("./data/labeled/labeled_output.parquet")
    
    # Assert all three expected label categories appear in the output dataset
    unique_labels = output_df["motive_label"].unique()
    print("Unique motives in output:", unique_labels)
    
    assert "organic" in unique_labels
    assert "fan-coordinated" in unique_labels
    assert "paid-commercial" in unique_labels

# ----------------- Unit Tests for Local File Ingestion Stage -----------------

def test_file_parser_csv(tmp_path):
    from collection.file_parser import parse_input_file
    
    csv_file = tmp_path / "test_reviews.csv"
    df_data = pd.DataFrame({
        "review": ["This movie was a masterpiece!", "Average film, not worth the hype."],
        "author": ["userA", "userB"],
        "created_utc": [1694073600, 1694083600],
        "id": ["t1", "t2"],
        "parent_id": [None, "t1"]
    })
    df_data.to_csv(csv_file, index=False)
    
    config = {
        "films": [
            {
                "id": "jawan_2023",
                "title": "Jawan",
                "release_date": "2023-09-07"
            }
        ]
    }
    
    df_parsed = parse_input_file(str(csv_file), "jawan_2023", config)
    assert len(df_parsed) == 2
    assert df_parsed.iloc[0]["text"] == "This movie was a masterpiece!"
    assert df_parsed.iloc[1]["author"] == "userB"
    assert df_parsed.iloc[1]["parent_id"] == "t1"

def test_file_parser_txt(tmp_path):
    from collection.file_parser import parse_input_file
    
    txt_file = tmp_path / "test_reviews.txt"
    txt_file.write_text("Masterpiece movie!\n\nAverage movie, story was copy.", encoding="utf-8")
    
    config = {
        "films": [
            {
                "id": "jawan_2023",
                "title": "Jawan",
                "release_date": "2023-09-07"
            }
        ]
    }
    
    df_parsed = parse_input_file(str(txt_file), "jawan_2023", config)
    assert len(df_parsed) == 2
    assert df_parsed.iloc[0]["text"] == "Masterpiece movie!"

def test_file_parser_synthetic_datasets():
    from collection.file_parser import parse_input_file
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Test Animal dataset (single film)
    df_animal = parse_input_file("data/Animal_synthetic_x_posts.csv", "animal_2023", config)
    assert not df_animal.empty
    assert (df_animal["film_id"] == "animal_2023").all()
    assert "query" in df_animal.columns
    # Ensure date conversion is numeric
    assert pd.api.types.is_numeric_dtype(df_animal["created_utc"])
    
    # Test all_films dataset (contains Pathaan, Animal, etc.)
    df_pathaan = parse_input_file("data/all_films_synthetic_x_posts.csv", "pathaan_2023", config)
    assert not df_pathaan.empty
    # Every row should be Pathaan
    assert (df_pathaan["film_id"] == "pathaan_2023").all()


