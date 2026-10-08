import os
import csv
import json
import argparse
import pandas as pd
from typing import List, Dict, Any

ANNOTATIONS_CSV = "./data/labeled/annotations.csv"
SCHEMA_JSON = "./annotation/schema.json"

def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_JSON, "r") as f:
        return json.load(f)

def add_annotation(
    cluster_id: int,
    label: str,
    annotator_id: str,
    confidence: float,
    notes: str = "",
    agreed_label: str = ""
):
    """Saves a cluster annotation row to the labeled data storage."""
    os.makedirs(os.path.dirname(ANNOTATIONS_CSV), exist_ok=True)
    
    # Check if headers need writing
    file_exists = os.path.exists(ANNOTATIONS_CSV)
    
    row = {
        "cluster_id": cluster_id,
        "label": label,
        "annotator_id": annotator_id,
        "confidence": confidence,
        "notes": notes,
        "agreed_label": agreed_label or label  # default agreed to label if empty
    }
    
    # Save using pandas or csv module
    df = pd.DataFrame([row])
    if file_exists:
        # Load existing, append and save to prevent duplicate cluster_ids if desired,
        # but standard append is fine
        df.to_csv(ANNOTATIONS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(ANNOTATIONS_CSV, index=False)
        
    print(f"Annotation added to: {ANNOTATIONS_CSV}")
    print(row)

def main():
    parser = argparse.ArgumentParser(description="BeyondBots Cluster Labeler Tool")
    parser.add_argument("--cluster", type=int, help="Cluster ID to annotate")
    parser.add_argument("--label", type=str, choices=["organic", "fan-coordinated", "paid-commercial"], help="Label choice")
    parser.add_argument("--annotator", type=str, help="Annotator name/ID")
    parser.add_argument("--confidence", type=float, help="Annotator confidence score (0.0 to 1.0)")
    parser.add_argument("--notes", type=str, default="", help="Notes or rationale for label")
    parser.add_argument("--agreed", type=str, default="", choices=["organic", "fan-coordinated", "paid-commercial"], help="Consensus label")
    
    args = parser.parse_args()
    
    # If no arguments provided, run in interactive mode
    if args.cluster is None or args.label is None or args.annotator is None or args.confidence is None:
        print("--- BeyondBots Interactive Cluster Labeler ---")
        try:
            cluster_id = int(input("Enter Cluster ID (integer): "))
            print("Select Label:")
            print("  1. organic")
            print("  2. fan-coordinated")
            print("  3. paid-commercial")
            label_opt = input("Choice (1-3): ")
            label_map = {"1": "organic", "2": "fan-coordinated", "3": "paid-commercial"}
            label = label_map.get(label_opt.strip())
            if not label:
                print("Invalid choice. Exiting.")
                return
                
            annotator_id = input("Enter Annotator ID: ").strip()
            confidence = float(input("Enter Confidence (0.0 - 1.0): "))
            notes = input("Enter Notes (optional): ").strip()
            agreed_label = input("Enter Agreed Label (optional, press Enter to default): ").strip()
            
            add_annotation(cluster_id, label, annotator_id, confidence, notes, agreed_label)
        except ValueError as e:
            print(f"Input error: {e}. Please enter valid types.")
    else:
        add_annotation(
            args.cluster, 
            args.label, 
            args.annotator, 
            args.confidence, 
            args.notes, 
            args.agreed
        )

if __name__ == "__main__":
    main()
