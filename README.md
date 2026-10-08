# BeyondBots: Text Mining Pipeline for Indian Film Discourse on X/Twitter

BeyondBots is a complete text mining pipeline designed to detect coordinated (non-organic) social media discourse around Indian film releases on X/Twitter. The pipeline aggregates tweets/replies (either loaded from local files or fetched via the X API v2), processes them through seven distinct classical text mining stages, extracts coordinated clusters, and classifies them into three motives: **organic**, **fan-coordinated**, or **paid-commercial**.

---

## 1. The Problem
Modern bot-detection tools and spam-filtering algorithms typically target automated API clients, headless browsers, or newly registered automated accounts ("bots"). However, these tools fail to detect **fan-army coordination** because they are operated by genuine, passionate humans acting out of loyalty to a star or franchise. 

Similarly, **sponsored-content detection systems** rely heavily on disclosure tags (e.g., `#ad`, `#sponsored`, `/s`, or flair tags). They fail to capture undisclosed commercial promotions where PR agencies deploy real or semi-automated accounts to create "organic-looking" praise or coordinate smear campaigns against rival releases.

BeyondBots focuses on analyzing the *textual representation, linguistic style, sentiment variance, and temporal coordination* of social media posts to identify these non-organic patterns, regardless of whether the account is a human fan or a paid actor.

---

## 2. Project Scope & Target Beneficiaries
**BeyondBots is a measurement and classification pipeline, not an intervention tool.** It does not ban accounts, report users, or moderate subreddits. It is designed for analysis and reporting.

The beneficiaries of this research project are prioritized as follows:
1. **Researchers (Direct Beneficiaries)**: Receive a citable, labeled benchmark dataset and a reproducible, multi-stage NLP methodology for code-mixed Indian languages.
2. **ASCI (Advertising Standards Council of India) & Platform Trust & Safety (Secondary)**: Obtain diagnostic indicators and investigative leads regarding undisclosed marketing campaigns. Note that this tool provides statistical probability of coordination, not legal proof of monetary transaction.
3. **Journalists (Tertiary)**: Gain a futuro-readable analysis layer to investigate how public opinion around cultural products is manufactured.
4. **General Audience (Indirect)**: Benefit only indirectly from increased transparency regarding the authenticity of online film reviews and hype.

---

## 3. Text Mining Pipeline Map

The pipeline is organized into distinct, sequential text mining stages, incorporating IMDb cross-platform ground-truth data:

```
┌────────────────────────────────────────────────────────┐   ┌───────────────────────────────────┐
│ Stage 0a: Social Media Ingestion                       │   │ Stage 0b: IMDb Ingestion          │
│ - Local Review Files (.txt, .pdf, .csv, .docx, .doc)   │   │ - Movie Metadata & Entity Graph   │
│ - X API v2 Recent Search                               │   │ - 1-10 Star Rating Distribution   │
│                                                        │   │ - Long-Form Verified User Reviews │
└───────────────────────────┬────────────────────────────┘   └─────────────────┬─────────────────┘
                            │                                                  │
                            ▼                                                  │
┌────────────────────────────────────────────────────────┐                     │
│ Stage 1: Preprocessing                                 │                     │
│ - Text cleaning (Markdown, URLs, Emoji-to-Text)        │                     │
│ - Code-mixed Tokenization (IndicNLP / spaCy)           │                     │
│ - Normalization & Transliteration Detection            │                     │
│ - Stopword Dual Handling (Cleaned & Retained lists)    │                     │
└───────────────────────────┬────────────────────────────┘                     │
                            │                                                  │
                            ▼                                                  │
┌────────────────────────────────────────────────────────┐                     │
│ Stage 2: Text Representation                           │                     │
│ - Sparse Vectorization: TF-IDF (NGrams 1-2)            │                     │
│ - Dense Embeddings: google/muril-base-cased            │                     │
└───────────────────────────┬────────────────────────────┘                     │
                            │                                                  │
                            ▼                                                  │
┌────────────────────────────────────────────────────────┐                     │
│ Stage 3: Document & Network Clustering                 │                     │
│ - Near-Duplicate Clustering: HDBSCAN on MuRIL          │                     │
│ - Temporal Coordination: Rolling Z-score burst detect  │                     │
│ - Network Community: Leiden algorithm on reply graphs  │                     │
└───────────────────────────┬────────────────────────────┘                     │
                            │                                                  │
                            ▼                                                  │
┌────────────────────────────────────────────────────────┐                     │
│ Stage 4: Feature Engineering                           │                     │
│ - Marketing-Copy Lang Score (Promotional vocabulary)   │                     │
│ - Fan-Identity Score (Fan honorifics/names match)      │                     │
│ - Criticism-Absence Score (Sentiment homogeneity)       │                     │
│ - Specificity-of-Reference Score (Dynamic IMDb Ground) │◄────────────────────┤ (Entities)
└───────────────────────────┬────────────────────────────┘                     │
                            │                                                  │
                            ▼                                                  │
┌───────────────────────────┴────────────────────────────┐                     │
│ Stage 5: Topic Modeling   │ Stage 6: Sentiment         │                     │
│ - BERTopic (MuRIL base)   │ - CardiffNLP XLM-RoBERTa   │                     │
│ - LDA fallback (Gensim)   │ - Post-level sentiment     │                     │
└───────────────────────────┬─┴──────────────────────────┘                     │
                            │                                                  │
                            ▼                                                  │
┌────────────────────────────────────────────────────────┐                     │
│ Stage 7: Classification & Explainability               │                     │
│ - Anthropic API (Claude 3.5 Sonnet) Motive Classifier  │                     │
│ - SHAP Explainer (Logistic regression surrogate model) │                     │
└───────────────────────────┬────────────────────────────┘                     │
                            │                                                  │
                            ▼                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Stage 8: Cross-Platform Analysis & Validation                                  │
│ - Sentiment Divergence Index (SDI): Social mean vs. IMDb normalized rating     │
│ - Bimodal Rating Polarization Score (BPS): Variance & 1★ vs 10★ ratio          │
│ - Diagnostic classification (Astroturfing inflation vs. Review-bombing wars)   │
│ - Case study validation benchmarking                                           │
└────────────────────────────────────────────────────────────────────────────────┘
```

| Pipeline Stage | Technique / Tool Used | Rationale & Downstream Dependency |
|---|---|---|
| **Stage 0: Ingestion** | Local Files, X API v2 & IMDb Collector | Multi-source ingestion combining fast-burst social media discourse with IMDb ratings, cast graphs, and long-form reviews. |
| **Stage 1: Preprocessing** | `indic-nlp-library` + `spaCy` + `emoji` | Cleans raw markdown/URLs. Normalizes emojis to text. Retains raw word cases for Indic scripts. Generates stopword-removed and stopword-retained tokens to preserve fan honorifics. |
| **Stage 2: Representation** | `TfidfVectorizer` (scikit-learn) + `google/muril-base-cased` (HF) | Baseline sparse TF-IDF (1,2-grams) captures exact templated phrases. MuRIL dense embeddings capture semantic similarity across English-Hindi code-mixed text. |
| **Stage 3: Clustering** | `HDBSCAN` + rolling window Z-scores + Leiden Algorithm | Groups semantic duplicates into coordinated clusters. Rolling Z-scores detect temporal spikes. Leiden community detection identifies conversational groups. |
| **Stage 4: Feature Engineering**| Custom scorer modules + Dynamic IMDb entity grounding | Computes 4 indicators (0.0 to 1.0) quantifying marketing language, fan identity markers, lack of negative sentiment, and detail specificity against verified IMDb entities. |
| **Stage 5: Topic Modeling** | `BERTopic` / `Gensim LDA` | Identifies thematic content per cluster to understand the semantic core of coordinated groups. |
| **Stage 6: Sentiment Analysis** | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Provides multi-lingual sentiment scores per post, which feed the variance-based criticism-absence score. |
| **Stage 7: Classification** | Anthropic `claude-3-5-sonnet` + `SHAP` surrogate | Classifies cluster motive (organic, fan, paid) based on feature scores and topic summaries. SHAP provides explainability over the feature inputs. |
| **Stage 8: Cross-Platform** | `IMDbCrossPlatformComparator` | Calculates Sentiment Divergence Index (SDI) and Bimodal Polarization Score (BPS) to diagnose PR rating inflation and review-bombing. |

---

## 4. Data Sources and Scoping Limitations
- **Data Sources**:
  - **Social Media (X/Twitter)**: Pre-scraped review files (`.txt`, `.pdf`, `.csv`, `.docx`, `.doc`) loaded locally, or recent tweets collected via the X API v2.
  - **IMDb Ground Truth**: Verified metadata (Director, Cast, Synopsis, Keywords), 1–10 star vote distributions, and user reviews cached in `data/imdb/`.
- **Scoping Limitations**:
  - X/Twitter demographics skew towards younger, mobile-first, and highly active fan bases.
  - Local ingestion relies on the quality and formatting of the provided files.
  - Recent Search API restricts collection to tweets from the last 7 days.

---

## 5. Honest Risks and Ambiguities
- **Ground Truth Limitations**: True "paid-commercial" labels are rarely disclosed. Ground truth labels in validation datasets are based on publicly exposed astroturfing campaigns, leaked PR briefs, or highly suspicious clusters with zero criticism and brand-new accounts.
- **Feature Overlap**: Dedicated fans often adopt the same slogans, release dates, and promotional hashtags as commercial PR campaigns. The pipeline relies on the *conjunction* of high sentiment uniformity, low reference specificity, account metadata, and IMDb cross-platform divergence to distinguish the two.

---

## 6. Research Questions
1. **RQ1**: Can linguistic patterns alone distinguish between passionate organic fan-army behavior and paid undisclosed commercial promotions?
2. **RQ2**: What is the correlation between coordinated commercial cluster volume and a film's box office performance? *(Note: A null result—showing no correlation between coordinated online hype and actual box office returns—is a valid and valuable scientific finding.)*
3. **RQ3**: How does script-mixing (Romanized Hindi/Tamil/Telugu vs. native scripts) differ between organic fan groups and official marketing pushes?
4. **RQ4**: What is the divergence pattern between fast-burst social media discourse (X) and verified long-form reviews/rating curves (IMDb), and can multimodal polarization curves diagnose coordinated review-bombing versus organic consensus?

