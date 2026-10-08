import os
import pandas as pd
from typing import List, Dict, Any

class CaseStudyValidator:
    """Validator that cross-checks pipeline-detected clusters against documented case studies."""
    def __init__(self, case_studies_path: str = "./data/labeled/case_studies.csv"):
        self.case_studies_path = case_studies_path
        self.case_studies = None

    def initialize_case_studies(self):
        """Initializes a sample case studies CSV if one doesn't exist, and loads it."""
        if not os.path.exists(self.case_studies_path):
            os.makedirs(os.path.dirname(self.case_studies_path), exist_ok=True)
            # Create a sample case studies file
            data = {
                "film": ["Jawan", "Jawan", "Pushpa 2"],
                "incident_type": ["paid-commercial", "fan-coordinated", "paid-commercial"],
                "source": ["investigative_report_leak", "x_community_notes", "industry_insider"],
                "date": ["2023-09-07", "2023-09-07", "2024-12-05"]
            }
            pd.DataFrame(data).to_csv(self.case_studies_path, index=False)
            print(f"Created sample case studies file at: {self.case_studies_path}")
            
        self.case_studies = pd.read_csv(self.case_studies_path)

    def validate(self, classified_df: pd.DataFrame) -> Dict[str, Any]:
        """Matches flagged clusters to documented incident dates/films."""
        self.initialize_case_studies()
        if classified_df.empty:
            return {"matches": [], "match_count": 0, "accuracy": 0.0}

        matches = []
        # Group by cluster to evaluate cluster level match
        for c_id, group in classified_df.groupby("cluster_id"):
            if c_id == -1: # Ignore noise
                continue
                
            motive = group["motive_label"].iloc[0]
            film_id = group["film_id"].iloc[0]
            # Convert film_id to film title (e.g. jawan_2023 -> Jawan)
            film_title = film_id.split("_")[0].capitalize()
            
            # Find matching case study by film name
            film_studies = self.case_studies[self.case_studies["film"].str.lower() == film_title.lower()]
            
            for _, study in film_studies.iterrows():
                # Check match (e.g. if the detected motive matches the documented incident type)
                if study["incident_type"] == motive:
                    matches.append({
                        "cluster_id": c_id,
                        "film": film_title,
                        "detected_motive": motive,
                        "documented_incident": study["incident_type"],
                        "documented_source": study["source"],
                        "date": study["date"]
                    })
                    
        total_valid = len(matches)
        
        print(f"Validation complete. Found {total_valid} overlaps with documented case studies:")
        for m in matches:
            print(f"  - Cluster {m['cluster_id']}: Matches '{m['film']}' documented '{m['documented_incident']}' via {m['documented_source']} on {m['date']}")
            
        return {
            "matches": matches,
            "match_count": total_valid
        }

if __name__ == "__main__":
    # Self-test block
    df = pd.DataFrame({
        "cluster_id": [1, 2],
        "motive_label": ["paid-commercial", "fan-coordinated"],
        "film_id": ["jawan_2023", "jawan_2023"]
    })
    validator = CaseStudyValidator("./data/labeled_test/case_studies.csv")
    validator.validate(df)
