import os
import sys
import pytest
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from collection.imdb_loader import IMDbLoader, IMDbMovieData
from sentiment.imdb_comparator import IMDbCrossPlatformComparator
from feature_engineering.scorers import SpecificityOfReferenceScorer

def test_imdb_loader_benchmark():
    loader = IMDbLoader(data_dir="data/imdb")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    imdb_data = loader.load_movie_data("jawan_2023", config)
    assert imdb_data.film_id == "jawan_2023"
    assert imdb_data.imdb_id == "tt15354916"
    assert imdb_data.rating > 0
    assert imdb_data.total_votes > 0
    assert len(imdb_data.rating_distribution) == 10
    assert isinstance(imdb_data.reviews, pd.DataFrame)
    assert not imdb_data.reviews.empty

def test_imdb_entity_lexicon():
    loader = IMDbLoader(data_dir="data/imdb")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    imdb_data = loader.load_movie_data("animal_2023", config)
    entities = imdb_data.get_entity_lexicon()
    
    # Cast/director checks (lowercase normalized)
    assert "ranbir" in entities or "kapoor" in entities
    assert "vanga" in entities or "sandeep" in entities
    assert len(entities) > 5

def test_dynamic_specificity_with_imdb_entities():
    entities = {"ranbir", "vanga", "bobby deol", "machine gun"}
    scorer = SpecificityOfReferenceScorer(imdb_entities=entities)
    
    text_grounded = "Ranbir Kapoor with that machine gun sequence directed by Vanga."
    tokens_grounded = text_grounded.split()
    score_grounded = scorer.score(text_grounded, tokens_grounded)
    
    text_generic = "good movie very nice must watch blockbuster film."
    tokens_generic = text_generic.split()
    score_generic = scorer.score(text_generic, tokens_generic)
    
    assert score_grounded > score_generic

def test_rating_polarization_metrics():
    # Polarized distribution (high 1-star and 10-star spikes)
    polarized_dist = {1: 20000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 20000}
    metrics_pol = IMDbCrossPlatformComparator.compute_rating_distribution_metrics(polarized_dist)
    
    # Bell curve / organic distribution
    organic_dist = {1: 500, 2: 800, 3: 1500, 4: 3000, 5: 6000, 6: 10000, 7: 15000, 8: 8000, 9: 3000, 10: 1000}
    metrics_org = IMDbCrossPlatformComparator.compute_rating_distribution_metrics(organic_dist)
    
    assert metrics_pol["bimodal_polarization_score"] > metrics_org["bimodal_polarization_score"]
    assert metrics_pol["extreme_ratio"] > 0.70

def test_cross_platform_discourse_comparator():
    imdb_data = IMDbMovieData(
        film_id="animal_2023",
        imdb_id="tt13751694",
        title="Animal",
        director="Sandeep Reddy Vanga",
        cast=["Ranbir Kapoor", "Bobby Deol"],
        rating=6.5,
        total_votes=10000,
        rating_distribution={1: 2500, 2: 500, 3: 500, 4: 500, 5: 500, 6: 500, 7: 1000, 8: 1000, 9: 500, 10: 2500}
    )
    
    # Synthetic social media posts with inflated positive sentiment (astroturfing pattern)
    df_social = pd.DataFrame({
        "id": ["s1", "s2", "s3"],
        "sentiment_score": [0.95, 0.90, 0.92],
        "motive_label": ["paid-commercial", "paid-commercial", "organic"]
    })
    
    res = IMDbCrossPlatformComparator.compare_discourse(df_social, imdb_data)
    assert res["social_mean_sentiment"] > 0.90
    assert res["sentiment_divergence_index"] > 0.20  # 0.92 - 0.65 = +0.27
    assert res["diagnostic_severity"] in ["Medium", "High"]
