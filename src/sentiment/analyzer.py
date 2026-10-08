import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import List, Dict, Any

class MultilingualSentimentAnalyzer:
    """Multilingual sentiment scoring per post using cardiffnlp/twitter-xlm-roberta-base-sentiment."""
    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Lazy load the sentiment classification model."""
        if self.model is None or self.tokenizer is None:
            if os.environ.get("BEYONDBOTS_MOCK_SENTIMENT") == "1":
                print("BEYONDBOTS_MOCK_SENTIMENT=1 detected. Using mockup sentiment scorer.")
                self.tokenizer = "mock"
                self.model = "mock"
                return
            try:
                print(f"Loading sentiment model '{self.model_name}' on {self.device}...")
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                print("Sentiment model loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load sentiment model: {e}")
                print("Falling back to a rule-based mock sentiment scorer.")
                self.tokenizer = "mock"
                self.model = "mock"

    def score_post(self, text: str) -> float:
        """
        Computes sentiment score for a single post.
        Returns a float between 0.0 (negative) and 1.0 (positive).
        """
        self.load_model()
        
        # Heuristic mock fallback
        if self.tokenizer == "mock":
            # Simple lexicon/rule-based mock
            text_lower = text.lower()
            pos_words = ["love", "great", "masterpiece", "blockbuster", "best", "good", "amazing", "superb", "goosebumps", "hit"]
            neg_words = ["hate", "bad", "worst", "flop", "boring", "waste", "rubbish", "crap", "terrible", "disaster"]
            
            pos_count = sum(1 for w in pos_words if w in text_lower)
            neg_count = sum(1 for w in neg_words if w in text_lower)
            
            if pos_count > neg_count:
                return 0.8 + (0.2 * (pos_count / (pos_count + 1)))
            elif neg_count > pos_count:
                return 0.2 - (0.2 * (neg_count / (neg_count + 1)))
            else:
                # Default neutral with a bit of deterministic variance based on text hashing
                seed = sum(ord(c) for c in text) % 100
                return 0.4 + 0.2 * (seed / 100.0)

        # Real model inference
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Cardiff NLP model has 3 outputs: 0 -> Negative, 1 -> Neutral, 2 -> Positive
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            # Map probabilities to a single score from 0.0 to 1.0
            # score = prob(positive) * 1.0 + prob(neutral) * 0.5 + prob(negative) * 0.0
            score = float(probs[2] * 1.0 + probs[1] * 0.5 + probs[0] * 0.0)
            return score
        except Exception as e:
            print(f"Error in model scoring: {e}. Returning default neutral score.")
            return 0.5

    def score_posts(self, texts: List[str]) -> List[float]:
        """Scores a list of posts."""
        return [self.score_post(t) for t in texts]

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sentiment scores for all posts in the DataFrame and returns it.
    Adds `sentiment_score` column.
    """
    df_out = df.copy()
    analyzer = MultilingualSentimentAnalyzer()
    texts = df_out["cleaned_text"].tolist()
    
    # Fast mock for bulk runs if configured
    df_out["sentiment_score"] = analyzer.score_posts(texts)
    return df_out

if __name__ == "__main__":
    # Self-test block
    texts = [
        "This film is a complete masterpiece, I absolutely loved it!",
        "It was okay, nothing special.",
        "What a terrible and boring movie, complete waste of money."
    ]
    
    df = pd.DataFrame({"cleaned_text": texts})
    
    # Force mock for fast check
    os.environ["BEYONDBOTS_MOCK_SENTIMENT"] = "1"
    res = analyze_sentiment(df)
    print("Sentiment scores:")
    print(res)
