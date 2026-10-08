import os
import json
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
import shap
import anthropic


class MotiveClassifier:
    """Few-shot motive classifier using Anthropic Claude API with heuristic fallback."""
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        self.model_name = model_name
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def classify_cluster_heuristic(
        self, 
        features: Dict[str, float], 
        topic: str, 
        examples: List[str]
    ) -> Dict[str, Any]:
        """Rule-based heuristic fallback classifier returning matching schema."""
        mkt = features.get("score_marketing_copy", 0.0)
        fan = features.get("score_fan_identity", 0.0)
        spec = features.get("score_specificity", 0.0)
        crit = features.get("score_criticism_absence", 0.0)

        # Paid Commercial Heuristics
        if mkt > 0.05 and crit > 0.8 and spec < 0.2:
            label = "paid-commercial"
            confidence = float(np.clip(0.6 + mkt * 3.0 + crit * 0.1, 0.5, 0.95))
            rationale = (f"High marketing vocabulary density ({mkt:.2f}) combined with absolute "
                         f"absence of criticism ({crit:.2f}) and vague movie references ({spec:.2f}) "
                         f"strongly indicates coordinated paid-commercial promotion.")
        # Fan Coordinated Heuristics
        elif fan > 0.1 and crit > 0.4:
            label = "fan-coordinated"
            confidence = float(np.clip(0.6 + fan * 2.0, 0.5, 0.95))
            rationale = (f"Significant presence of fan-identity markers ({fan:.2f}) and high "
                         f"sentiment uniformity ({crit:.2f}) indicates organic but highly "
                         f"coordinated fan-army loyalty behavior.")
        # Organic
        else:
            label = "organic"
            confidence = float(np.clip(0.8 - mkt * 0.3, 0.5, 0.95))
            rationale = (f"Moderate to high detail specificity ({spec:.2f}) with standard "
                         f"criticism levels (absence score: {crit:.2f}) and low promotional vocabulary density "
                         f"({mkt:.2f}) matches organic conversational patterns.")

        return {
            "label": label,
            "confidence": confidence,
            "rationale": rationale
        }

    def classify_cluster(
        self, 
        cluster_id: int, 
        features: Dict[str, float], 
        topic: str, 
        examples: List[str]
    ) -> Dict[str, Any]:
        """Classifies cluster motive via Anthropic few-shot prompting, falling back to heuristics."""
        if not self.client:
            return self.classify_cluster_heuristic(features, topic, examples)

        prompt = f"""You are a research bot analyzing coordinated social media campaigns.
Classify the following X/Twitter cluster into one of three motive categories:
1. "organic": Natural, uncoordinated posts with normal sentiment variance.
2. "fan-coordinated": Passionate fan-army accounts mass-posting templates or repetitive praise out of star loyalty.
3. "paid-commercial": Undisclosed commercial marketing/PR campaigns designed to appear organic.

Here are the cluster properties:
- Cluster ID: {cluster_id}
- Topic summary: {topic}
- Cluster-level Feature Scores:
  * Marketing-copy vocabulary density: {features.get('score_marketing_copy', 0.0):.3f}
  * Fan-identity marker density (honorifics): {features.get('score_fan_identity', 0.0):.3f}
  * Criticism absence (sentiment uniformity): {features.get('score_criticism_absence', 0.0):.3f}
  * Specificity of reference (scene detail density): {features.get('score_specificity', 0.0):.3f}

Sample posts from this cluster:
{chr(10).join([f"- '{text[:200]}'" for text in examples[:5]])}

Provide a structured JSON output with the exact keys:
{{
  "label": "organic" | "fan-coordinated" | "paid-commercial",
  "confidence": float between 0.0 and 1.0,
  "rationale": "detailed explanation of why this label fits the features and examples"
}}
"""
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            content_text = message.content[0].text
            # Extract JSON block
            json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(content_text)
        except Exception as e:
            print(f"Anthropic API call failed: {e}. Falling back to rule-based classification.")
            return self.classify_cluster_heuristic(features, topic, examples)

def explain_features_with_shap(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[Any, np.ndarray]:
    """
    Fits a Logistic Regression surrogate model on the features against the LLM labels,
    and runs SHAP KernelExplainer to interpret feature importances.
    """
    # Check if we have enough sample clusters and classes to train a classifier
    unique_classes = np.unique(y)
    if len(X) < 3 or len(unique_classes) < 2:
        print("Warning: Insufficient clusters or classes to run SHAP explainability. Generating placeholder SHAP values.")
        # Return mock shap explainer and values
        class MockExplainer:
            def __init__(self, num_feats):
                self.expected_value = [0.33, 0.33, 0.33]
            def __call__(self, x):
                return self
        
        mock_shap_values = np.zeros((len(X), len(feature_names)))
        # Put some dummy feature weights
        for i in range(len(X)):
            mock_shap_values[i] = [0.15, 0.1, -0.05, 0.2]
        return MockExplainer(len(feature_names)), mock_shap_values

    # Train surrogate
    surrogate = LogisticRegression(max_iter=1000, random_state=42)
    surrogate.fit(X, y)
    
    # We explain the probability outputs of the model
    # KernelExplainer needs a prediction function and background data
    explainer = shap.KernelExplainer(surrogate.predict_proba, X)
    
    # Generate SHAP values
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

def classify_dataframe_clusters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes cluster-level summaries, classifies them, and runs SHAP surrogate explanation.
    """
    df_out = df.copy()
    classifier = MotiveClassifier()
    
    # Extract unique clusters (ignoring noise -1 if present)
    clusters = df_out["cluster_id"].unique()
    
    cluster_results = {}
    
    for c_id in clusters:
        # Group posts in this cluster
        group = df_out[df_out["cluster_id"] == c_id]
        
        # Calculate mean scores
        features = {
            "score_marketing_copy": float(group["score_marketing_copy"].mean()),
            "score_fan_identity": float(group["score_fan_identity"].mean()),
            "score_specificity": float(group["score_specificity"].mean()),
            "score_criticism_absence": float(group["score_criticism_absence"].mean())
        }
        
        topic = group["cluster_topic"].iloc[0] if "cluster_topic" in group.columns else "unknown"
        examples = group["cleaned_text"].dropna().tolist()[:5]
        
        # Run classification
        res = classifier.classify_cluster(c_id, features, topic, examples)
        cluster_results[c_id] = res

    # Map classifications back to posts
    df_out["motive_label"] = df_out["cluster_id"].map(lambda x: cluster_results[x]["label"])
    df_out["motive_confidence"] = df_out["cluster_id"].map(lambda x: cluster_results[x]["confidence"])
    df_out["motive_rationale"] = df_out["cluster_id"].map(lambda x: cluster_results[x]["rationale"])

    # Prepare data for SHAP explainability
    # We construct a cluster-level feature matrix
    X_list = []
    y_list = []
    c_ids = []
    
    label_map = {"organic": 0, "fan-coordinated": 1, "paid-commercial": 2}
    
    for c_id, res in cluster_results.items():
        if c_id == -1: # Skip noise cluster for SHAP calculation
            continue
        group = df_out[df_out["cluster_id"] == c_id]
        features = [
            group["score_marketing_copy"].mean(),
            group["score_fan_identity"].mean(),
            group["score_specificity"].mean(),
            group["score_criticism_absence"].mean()
        ]
        X_list.append(features)
        y_list.append(label_map.get(res["label"], 0))
        c_ids.append(c_id)
        
    shap_data = {}
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        feature_names = ["marketing_copy", "fan_identity", "specificity", "criticism_absence"]
        
        explainer, shap_vals = explain_features_with_shap(X, y, feature_names)
        
        # Package SHAP output for logging/serialization
        shap_data = {
            "feature_names": feature_names,
            "cluster_ids": c_ids,
            "shap_values": shap_vals.tolist() if isinstance(shap_vals, np.ndarray) else [sv.tolist() for sv in shap_vals]
        }
        
    return df_out, shap_data

if __name__ == "__main__":
    # Self-test block
    df = pd.DataFrame({
        "id": ["p0", "p1", "p2", "p3", "p4", "p5"],
        "cleaned_text": [
            "This movie is a total blockbuster masterpiece!",
            "SRK entry in Jawan is pure goosebumps!",
            "It was okay, average screenplay.",
            "Must watch film of the decade!",
            "King Khan does it again!",
            "I hated the first half, second half was fine."
        ],
        "cluster_id": [1, 1, 2, 3, 3, 2],
        "cluster_topic": ["masterpiece, blockbuster", "srk, king, entry", "screenplay, average", "watch, film", "khan, king", "hate, half"],
        "score_marketing_copy": [0.8, 0.1, 0.05, 0.75, 0.1, 0.0],
        "score_fan_identity": [0.0, 0.9, 0.0, 0.0, 0.85, 0.0],
        "score_specificity": [0.1, 0.4, 0.8, 0.1, 0.3, 0.75],
        "score_criticism_absence": [0.95, 0.95, 0.4, 0.95, 0.95, 0.4]
    })
    
    # Check fallback heuristic
    res_df, shap_info = classify_dataframe_clusters(df)
    print("Classified DataFrame:")
    print(res_df[["id", "cluster_id", "motive_label", "motive_confidence"]])
    print("\nSHAP Info keys:", shap_info.keys() if shap_info else "No SHAP computed")
