# BeyondBots: Detecting Coordinated Non-Organic Social Media Discourse around Indian Film Releases

**Author**: BeyondBots Research Group  
**Date**: August 2026  

---

## Abstract
Social media has become a primary driver of film marketing and audience reception. However, the authenticity of online discourse is frequently compromised by coordinate behaviors: passionate fan-armies mass-posting templated praise, and commercial agencies running undisclosed campaigns (astroturfing). Conventional bot-detection tools look for automated markers, missing human-driven coordination. This paper introduces BeyondBots, a text mining pipeline that processes code-mixed Indic-English text through seven stages to classify discourse into organic, fan-coordinated, or paid-commercial. We show that linguistic markers, sentiment variance, and specificity can distinguish these classes.

---

## 1. Introduction
Modern spam-detection algorithms and bot-detection platforms focus on identifying automated API access, suspicious request timing, or empty user profiles. While effective against standard automated bots, they are blind to two key categories in cultural discourse:
1. **Fan-Armies**: Passionate human communities coordinate to boost their favorite star's film using templated praise.
2. **Astroturfing (Paid-Commercial)**: PR agencies deploy semi-automated or human accounts to mimic organic enthusiasm or coordinate smear campaigns.

BeyondBots is a measurement pipeline designed to identify these clusters by analyzing the text structures, language style, reference specificity, and sentiment uniformity of posts on Reddit.

---

## 2. Methodology & Text Mining Pipeline
The pipeline operates in seven stages:
1. **Preprocessing**: Cleans raw markdown/URLs, normalizes emojis to text, splits tokens, and maintains clean-vs-retained token pipelines to retain honorifics.
2. **Text Representation**: Encodes text using baseline sparse TF-IDF (1,2-grams) and dense embeddings via the multilingual model `google/muril-base-cased`.
3. **Clustering**: Applies cosine similarity on dense embeddings followed by HDBSCAN to group near-duplicate posts. Conducts rolling z-score burst detection and Leiden community graph partitioning.
4. **Feature Scoring**: Extracts four scores (0.0 to 1.0) per cluster:
   - *Marketing Copy language score* (promotional vocabulary weight).
   - *Fan-Identity marker score* (lexicon matches on stopword-retained tokens).
   - *Criticism-Absence score* (sentiment homogeneity).
   - *Specificity-of-Reference score* (named entity and scene density).
5. **Topic Modeling**: Runs BERTopic (reusing Stage 2 embeddings) or Gensim LDA per cluster.
6. **Sentiment Analysis**: Sentiment scoring using `cardiffnlp/twitter-xlm-roberta-base-sentiment` to feed Stage 4.
7. **Classification & Explainability**: Few-shot classification via Anthropic's Claude 3.5 Sonnet to output `{label, confidence, rationale}` JSON. Uses `SHAP` KernelExplainer on a logistic regression surrogate model to interpret feature importance.

---

## 3. Preliminary Results & Discussion
Our sanity checks on synthetic datasets verify distinct profiles:
- **Organic**: Characterized by high reference specificity (names of theaters, specific scenes, actors) and low criticism absence (diverse sentiment polarity).
- **Fan-Coordinated**: High fan-identity honorifics, high sentiment uniformity, and lower reference specificity.
- **Paid-Commercial**: Extremely high marketing vocabulary density, low specificity, and near-zero sentiment variance (criticism absence = 1.0).

This demonstrates that feature profiles successfully segment coordinated groups from natural conversation.
