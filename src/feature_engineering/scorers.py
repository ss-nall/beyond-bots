import numpy as np
import pandas as pd
import re
from typing import List, Dict, Set, Any, Optional

# Load spaCy for NER. Standard English model is great for Romanized text.
try:
    import spacy
    try:
        nlp_ner = spacy.load("en_core_web_sm")
    except Exception:
        try:
            nlp_ner = spacy.load("xx_ent_wiki_sm")
        except Exception:
            nlp_ner = None
except ImportError:
    spacy = None
    nlp_ner = None

class MarketingCopyScorer:
    """Computes promotional vocabulary density using TF-IDF feature weights."""
    def __init__(self, promotional_lexicon: List[str]):
        self.promotional_lexicon = [w.lower() for w in promotional_lexicon]

    def score(self, tfidf_vector: np.ndarray, feature_names: List[str]) -> float:
        """
        Calculates the ratio of TF-IDF weight of promotional vocabulary 
        to total TF-IDF weight.
        """
        if tfidf_vector.sum() == 0:
            return 0.0
            
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        promo_weight = 0.0
        
        for word in self.promotional_lexicon:
            # Look up word or sub-spans in feature names
            if word in feature_to_idx:
                promo_weight += tfidf_vector[feature_to_idx[word]]
            else:
                # Fallback for sub-words
                for f_name, f_idx in feature_to_idx.items():
                    if word in f_name:
                        promo_weight += tfidf_vector[f_idx] * 0.5  # half weight for partial match
                        
        total_weight = tfidf_vector.sum()
        score_val = promo_weight / total_weight if total_weight > 0 else 0.0
        return float(np.clip(score_val, 0.0, 1.0))

class FanIdentityScorer:
    """Lexicon matcher against fan honorifics/slogans on stopword-retained tokens."""
    def __init__(self, fan_lexicon: List[str]):
        # Split multi-word phrases to individual tokens for word-level matching
        self.fan_lexicon = set()
        for phrase in fan_lexicon:
            for word in phrase.lower().split():
                self.fan_lexicon.add(word)

    def score(self, tokens_retained: List[str]) -> float:
        """Computes ratio of fan identity tokens to total retained tokens."""
        if not tokens_retained:
            return 0.0
        match_count = sum(1 for t in tokens_retained if t.lower() in self.fan_lexicon)
        return float(np.clip(match_count / len(tokens_retained), 0.0, 1.0))

class CriticismAbsenceScorer:
    """Computes cluster sentiment uniformity. Homogeneous sentiment = high criticism absence."""
    def score_cluster(self, sentiment_scores: List[float]) -> float:
        """
        Computes 1.0 - 4.0 * variance (since max variance of values in [0, 1] is 0.25).
        Returns a float between 0.0 and 1.0.
        """
        if len(sentiment_scores) <= 1:
            # Single post or empty: high absence of criticism by default
            return 1.0
        var = np.var(sentiment_scores)
        absence = 1.0 - (4.0 * var)
        return float(np.clip(absence, 0.0, 1.0))

class SpecificityOfReferenceScorer:
    """Computes named entity and specific reference density using spaCy NER and dynamic IMDb entities."""
    def __init__(self, imdb_entities: Optional[Set[str]] = None):
        self.imdb_entities = {e.lower() for e in imdb_entities} if imdb_entities else set()

    def score(self, text: str, tokens: List[str], imdb_entities: Optional[Set[str]] = None) -> float:
        """
        Counts named entities, scene-specific markers, and matched IMDb entities
        divided by token length.
        """
        if not tokens:
            return 0.0
            
        entity_count = 0
        if nlp_ner is not None:
            try:
                doc = nlp_ner(text)
                entity_count = len(doc.ents)
            except Exception:
                pass
                
        # Dynamic IMDb entity matches (cast, director, plot keywords)
        entities_to_check = imdb_entities if imdb_entities is not None else self.imdb_entities
        imdb_matches = 0
        if entities_to_check:
            text_lower = text.lower()
            for entity in entities_to_check:
                if entity in text_lower:
                    imdb_matches += 1

        # Heuristic count of specific markers (e.g. numbers, specific times, uppercase words)
        num_count = len(re.findall(r'\b\d+\b', text))
        
        # Proper noun heuristic: uppercase word tokens (excluding sentence starts)
        prop_noun_count = 0
        for i, tok in enumerate(tokens):
            if i > 0 and tok[0].isupper() and not tokens[i-1] in [".", "!", "?"]:
                prop_noun_count += 1
                
        specificity_signals = entity_count + (imdb_matches * 1.0) + (num_count * 0.5) + (prop_noun_count * 0.2)
        score_val = specificity_signals / len(tokens)
        return float(np.clip(score_val, 0.0, 1.0))

def compute_features(
    df: pd.DataFrame, 
    tfidf_matrix: np.ndarray, 
    feature_names: List[str],
    promotional_lexicon: List[str],
    fan_lexicon: List[str],
    sentiment_col: str = "sentiment_score",
    imdb_entities: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Computes features for each post and adds the 4 scores to the DataFrame.
    """
    df_out = df.copy()
    
    # Initialize scorers
    marketing_scorer = MarketingCopyScorer(promotional_lexicon)
    fan_scorer = FanIdentityScorer(fan_lexicon)
    specificity_scorer = SpecificityOfReferenceScorer(imdb_entities=imdb_entities)
    criticism_scorer = CriticismAbsenceScorer()
    
    # Post-level scores
    mkt_scores = []
    fan_scores = []
    spec_scores = []
    
    for idx, row in df_out.iterrows():
        # Marketing score (Stage 2 TF-IDF matrix row)
        tfidf_vec = tfidf_matrix[idx]
        mkt_scores.append(marketing_scorer.score(tfidf_vec, feature_names))
        
        # Fan identity score (Stage 1 tokens_retained)
        tokens_retained = row.get("tokens_retained", [])
        fan_scores.append(fan_scorer.score(tokens_retained))
        
        # Specificity score (Stage 1 cleaned_text and tokens_clean)
        cleaned_text = row.get("cleaned_text", "")
        tokens_clean = row.get("tokens_clean", [])
        spec_scores.append(specificity_scorer.score(cleaned_text, tokens_clean))
        
    df_out["score_marketing_copy"] = mkt_scores
    df_out["score_fan_identity"] = fan_scores
    df_out["score_specificity"] = spec_scores
    
    # Cluster-level Criticism Absence Score (Stage 6 sentiment variance)
    # Group by cluster_id, compute variance of sentiment, map back
    if sentiment_col not in df_out.columns:
        # Mock sentiment score if not present
        df_out[sentiment_col] = 0.5
        
    cluster_scores = {}
    for c_id, group in df_out.groupby("cluster_id"):
        sent_scores = group[sentiment_col].tolist()
        cluster_scores[c_id] = criticism_scorer.score_cluster(sent_scores)
        
    df_out["score_criticism_absence"] = df_out["cluster_id"].map(cluster_scores)
    
    return df_out

if __name__ == "__main__":
    # Self-test block
    promo = ["blockbuster", "masterpiece", "must watch"]
    fans = ["srk", "king khan", "thalapathy"]
    
    # Sample data
    df = pd.DataFrame({
        "id": ["p0", "p1"],
        "cleaned_text": ["This Jawan movie is a blockbuster masterpiece! Must watch!", "srk is the badshah of bollywood."],
        "tokens_clean": [["jawan", "movie", "blockbuster", "masterpiece", "must", "watch"], ["srk", "badshah", "bollywood"]],
        "tokens_retained": [["this", "jawan", "movie", "is", "a", "blockbuster", "masterpiece", "must", "watch"], ["srk", "is", "the", "badshah", "of", "bollywood"]],
        "cluster_id": [1, 1],
        "sentiment_score": [0.9, 0.8]  # stage 6 output
    })
    
    tfidf_matrix = np.array([
        [0.0, 0.0, 0.4, 0.4, 0.4, 0.4], # p0
        [0.7, 0.7, 0.0, 0.0, 0.0, 0.0]  # p1
    ])
    feat_names = ["srk", "badshah", "blockbuster", "masterpiece", "must", "watch"]
    
    res = compute_features(df, tfidf_matrix, feat_names, promo, fans)
    print("Features Computed:")
    print(res[[
        "id", "score_marketing_copy", "score_fan_identity", 
        "score_specificity", "score_criticism_absence"
    ]])
