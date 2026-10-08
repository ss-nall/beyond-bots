import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from collection.imdb_loader import IMDbMovieData

class IMDbCrossPlatformComparator:
    """Computes cross-platform comparative sentiment, polarization, and divergence metrics."""
    
    @staticmethod
    def compute_rating_distribution_metrics(rating_distribution: Dict[int, int]) -> Dict[str, Any]:
        """
        Analyzes the 1-10 star rating distribution from IMDb.
        Detects bimodal polarization (fan-wars / review-bombing) and extreme vote ratios.
        """
        if not rating_distribution:
            return {
                "total_votes": 0,
                "weighted_mean_rating": 7.0,
                "rating_variance": 0.0,
                "one_star_pct": 0.0,
                "ten_star_pct": 0.0,
                "extreme_ratio": 0.0,
                "bimodal_polarization_score": 0.0
            }

        total_votes = sum(rating_distribution.values())
        if total_votes == 0:
            return {
                "total_votes": 0,
                "weighted_mean_rating": 7.0,
                "rating_variance": 0.0,
                "one_star_pct": 0.0,
                "ten_star_pct": 0.0,
                "extreme_ratio": 0.0,
                "bimodal_polarization_score": 0.0
            }

        # Calculate weighted mean and variance
        weighted_sum = sum(star * count for star, count in rating_distribution.items())
        weighted_mean = weighted_sum / total_votes

        weighted_sq_diff = sum(count * ((star - weighted_mean) ** 2) for star, count in rating_distribution.items())
        rating_variance = weighted_sq_diff / total_votes

        v1 = rating_distribution.get(1, 0)
        v10 = rating_distribution.get(10, 0)
        
        one_star_pct = v1 / total_votes
        ten_star_pct = v10 / total_votes
        extreme_ratio = (v1 + v10) / total_votes

        # Bimodal Polarization Score:
        # High when both 1-star and 10-star shares are elevated, or extreme ratio is high with elevated variance
        # Standard unpolarized films have extreme ratio < 0.15 and variance < 3.5.
        polarization_raw = (extreme_ratio * 1.5) + (rating_variance / 10.0)
        bimodal_polarization_score = float(np.clip(polarization_raw, 0.0, 1.0))

        return {
            "total_votes": total_votes,
            "weighted_mean_rating": round(weighted_mean, 2),
            "rating_variance": round(rating_variance, 3),
            "one_star_pct": round(one_star_pct, 3),
            "ten_star_pct": round(ten_star_pct, 3),
            "extreme_ratio": round(extreme_ratio, 3),
            "bimodal_polarization_score": round(bimodal_polarization_score, 3)
        }

    @staticmethod
    def compare_discourse(
        df_social: pd.DataFrame, 
        imdb_data: IMDbMovieData,
        sentiment_col: str = "sentiment_score"
    ) -> Dict[str, Any]:
        """
        Computes the cross-platform comparison between X/Twitter social discourse and IMDb data.
        Returns a comprehensive diagnostic report.
        """
        # Social media metrics
        if df_social.empty or sentiment_col not in df_social.columns:
            social_mean_sent = 0.5
            social_std_sent = 0.0
            social_post_count = 0
            motive_breakdown = {}
        else:
            social_mean_sent = float(df_social[sentiment_col].mean())
            social_std_sent = float(df_social[sentiment_col].std() if len(df_social) > 1 else 0.0)
            social_post_count = len(df_social)
            motive_breakdown = df_social["motive_label"].value_counts().to_dict() if "motive_label" in df_social.columns else {}

        # IMDb Rating metrics
        rating_metrics = IMDbCrossPlatformComparator.compute_rating_distribution_metrics(imdb_data.rating_distribution)
        imdb_normalized_rating = imdb_data.rating / 10.0

        # IMDb Review Text sentiment
        imdb_review_mean_sent = imdb_normalized_rating
        if isinstance(imdb_data.reviews, pd.DataFrame) and not imdb_data.reviews.empty:
            if "rating" in imdb_data.reviews.columns:
                imdb_review_mean_sent = float(imdb_data.reviews["rating"].mean() / 10.0)

        # Sentiment Divergence Index (SDI):
        # Gap between social media post sentiment and verified IMDb rating/reviews
        sdi_signed = round(social_mean_sent - imdb_normalized_rating, 3)
        sdi_abs = round(abs(social_mean_sent - imdb_normalized_rating), 3)

        # Diagnostic classification
        diag_status = "Harmonious Organic Discourse"
        diag_severity = "Low"
        reasons = []

        if sdi_signed > 0.20 and motive_breakdown.get("paid-commercial", 0) > 0:
            diag_status = "Inflated Social Hype (PR Astroturfing Footprint)"
            diag_severity = "High"
            reasons.append(f"Social sentiment (+{social_mean_sent:.2f}) significantly exceeds verified IMDb rating ({imdb_data.rating}/10) alongside identified paid-commercial clusters.")
        
        if rating_metrics["bimodal_polarization_score"] > 0.40:
            if diag_status == "Harmonious Organic Discourse":
                diag_status = "IMDb Rating Polarization / Fan-War"
                diag_severity = "Medium"
            reasons.append(f"High IMDb extreme vote ratio ({rating_metrics['extreme_ratio']*100:.1f}% 1★/10★ votes) indicates intense audience polarization or review-bombing.")

        if rating_metrics["one_star_pct"] > 0.15 and social_mean_sent > 0.70:
            reasons.append(f"Discrepancy detected: Elevated 1-star reviews on IMDb ({rating_metrics['one_star_pct']*100:.1f}%) contrasted by positive social media sentiment.")

        if not reasons:
            reasons.append("Social sentiment variance and IMDb rating curves are aligned organically.")

        return {
            "film_id": imdb_data.film_id,
            "title": imdb_data.title,
            "imdb_id": imdb_data.imdb_id,
            "imdb_rating": imdb_data.rating,
            "imdb_total_votes": imdb_data.total_votes,
            "social_post_count": social_post_count,
            "social_mean_sentiment": round(social_mean_sent, 3),
            "social_sentiment_std": round(social_std_sent, 3),
            "imdb_review_mean_sentiment": round(imdb_review_mean_sent, 3),
            "sentiment_divergence_index": sdi_signed,
            "sentiment_divergence_abs": sdi_abs,
            "rating_distribution_metrics": rating_metrics,
            "motive_breakdown": motive_breakdown,
            "diagnostic_status": diag_status,
            "diagnostic_severity": diag_severity,
            "diagnostic_reasons": reasons
        }
