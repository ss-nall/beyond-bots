import os
import sys

# Ensure src directory and project root are in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline import run_pipeline

def precompute_all_films():
    os.environ["BEYONDBOTS_MOCK_EMBEDDINGS"] = "1"
    os.environ["BEYONDBOTS_MOCK_SENTIMENT"] = "1"

    films = [
        ("jawan_2023", "data/jawan_2023_posts.csv"),
        ("animal_2023", "data/animal_2023_posts.csv"),
        ("pushpa2_2024", "data/pushpa2_2024_posts.csv"),
        ("jailer_2023", "data/jailer_2023_posts.csv"),
        ("kalki_2024", "data/kalki_2024_posts.csv"),
        ("rrr_2022", "data/rrr_2022_posts.csv"),
        ("pathaan_2023", "data/pathaan_2023_posts.csv")
    ]

    for film_id, path in films:
        print(f"\n==========================================")
        print(f"Precomputing intelligence for: {film_id}")
        print(f"==========================================")
        if os.path.exists(path):
            run_pipeline(film_id, config_path="config.yaml", override_input_file=path)
        else:
            print(f"Skipping {film_id}, file not found: {path}")

    print("\nAll films precomputed successfully!")

if __name__ == "__main__":
    precompute_all_films()
