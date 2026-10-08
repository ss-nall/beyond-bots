import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Any
import os

# Try importing BERTopic and Gensim
try:
    from bertopic import BERTopic
    has_bertopic = True
except ImportError:
    has_bertopic = False

try:
    from gensim import corpora
    from gensim.models import LdaModel
    has_gensim = True
except ImportError:
    has_gensim = False

class ClusterTopicModeler:
    """Topic modeler that runs per-cluster to characterize what each group is discussing."""
    
    def extract_fallback_keywords(self, texts: List[str], top_n: int = 5) -> List[str]:
        """Extracts top keywords using standard TF-IDF as a robust fallback."""
        if not texts or len(texts) == 0:
            return ["no_content"]
            
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
            X = vectorizer.fit_transform(texts)
            scores = np.asarray(X.sum(axis=0)).flatten()
            words = vectorizer.get_feature_names_out()
            top_indices = np.argsort(scores)[::-1][:top_n]
            return [words[i] for i in top_indices]
        except Exception:
            # Simplest fallback: split words and find unique
            all_words = []
            for t in texts:
                all_words.extend([w.lower() for w in t.split() if len(w) > 3])
            # Count frequency
            from collections import Counter
            counts = Counter(all_words)
            return [w for w, _ in counts.most_common(top_n)]

    def get_cluster_topic_summary(
        self, 
        cluster_df: pd.DataFrame, 
        embeddings: np.ndarray = None, 
        top_n: int = 5
    ) -> str:
        """
        Determines the main topic of a cluster of posts.
        Reuses dense embeddings for BERTopic if possible; falls back to Gensim LDA or TF-IDF.
        """
        texts = cluster_df["cleaned_text"].dropna().tolist()
        if not texts:
            return "empty_cluster"

        # If too few documents for actual topic models, use TF-IDF fallback directly
        if len(texts) < 5:
            keywords = self.extract_fallback_keywords(texts, top_n)
            return ", ".join(keywords)

        # 1. Attempt BERTopic (if embeddings are provided)
        if has_bertopic and embeddings is not None and len(texts) >= 10:
            try:
                # Disable embedding step in BERTopic since we already have them
                # For small clusters, keep it simple
                topic_model = BERTopic(min_topic_size=max(2, len(texts)//3))
                topics, _ = topic_model.fit_transform(texts, embeddings)
                # Get info of top topic (excluding noise -1 if possible)
                freq = topic_model.get_topic_info()
                first_topic = freq.iloc[0]["Topic"]
                if first_topic == -1 and len(freq) > 1:
                    first_topic = freq.iloc[1]["Topic"]
                words = [w[0] for w in topic_model.get_topic(first_topic)[:top_n]]
                return ", ".join(words)
            except Exception as e:
                print(f"BERTopic failed: {e}. Trying LDA fallback.")

        # 2. Attempt Gensim LDA
        if has_gensim and len(texts) >= 5:
            try:
                tokenized_texts = [t.lower().split() for t in texts]
                dictionary = corpora.Dictionary(tokenized_texts)
                corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
                
                # Fit simple LDA with 1 topic
                lda = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=5, random_state=42)
                topic_terms = lda.show_topic(0, topn=top_n)
                words = [term for term, _ in topic_terms]
                return ", ".join(words)
            except Exception as e:
                print(f"Gensim LDA failed: {e}. Using TF-IDF keywords.")

        # 3. Final fallback to TF-IDF keywords
        keywords = self.extract_fallback_keywords(texts, top_n)
        return ", ".join(keywords)

def compute_cluster_topics(
    df: pd.DataFrame, 
    embeddings: np.ndarray = None
) -> pd.DataFrame:
    """
    Computes topic summaries per cluster and appends a `cluster_topic` column.
    """
    df_out = df.copy()
    modeler = ClusterTopicModeler()
    
    # Store mappings of cluster_id -> topic string
    topic_map = {}
    
    # Group by cluster_id and run topic modeling
    for c_id, group in df_out.groupby("cluster_id"):
        # Map cluster embeddings if provided
        group_embeddings = None
        if embeddings is not None:
            # Select subset of embeddings corresponding to group indices
            indices = group.index.tolist()
            group_embeddings = embeddings[indices]
            
        topic_str = modeler.get_cluster_topic_summary(group, group_embeddings)
        topic_map[c_id] = topic_str
        
    df_out["cluster_topic"] = df_out["cluster_id"].map(topic_map)
    return df_out

if __name__ == "__main__":
    # Self-test block
    texts = [
        "Jawan trailer is fantastic, SRK entry is massive!",
        "SRK Jawan movie release date is next week, can't wait.",
        "Anirudh background score in Jawan trailer is goosebumps.",
        "Vijay cameo in Jawan is confirmed by director Atlee.",
        "Jawan advanced bookings are breaking all previous records!"
    ]
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(len(texts))],
        "cleaned_text": texts,
        "cluster_id": [1, 1, 1, 1, 1]
    })
    
    res = compute_cluster_topics(df)
    print("Cluster Topics:")
    print(res[["cluster_id", "cluster_topic"]].drop_duplicates())
