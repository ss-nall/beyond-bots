import os
import sys
import yaml
import json
import time
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collection.real_data_scraper import scrape_real_reddit_discourse, scrape_real_imdb_metadata_and_reviews
from collection.imdb_loader import IMDbLoader, IMDbMovieData

def build_all_real_datasets(config_path: str = "config.yaml"):
    """
    Builds real high-volume social media datasets and IMDb profiles
    for all films configured in config.yaml.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = "data"
    imdb_dir = "data/imdb"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(imdb_dir, exist_ok=True)

    imdb_loader = IMDbLoader(data_dir=imdb_dir)

    for film in config.get("films", []):
        film_id = film["id"]
        film_title = film["title"]
        imdb_id = film.get("imdb_id", "tt0000000")
        keywords = film.get("keywords", [film_title.lower()])

        print(f"\n==========================================")
        print(f"Collecting real data for: {film_title} ({film_id})")
        print(f"==========================================")

        # 1. Scrape real Reddit social media discourse across multiple keywords
        all_real_posts = []
        for kw in keywords[:3]:
            posts = scrape_real_reddit_discourse(kw, limit=100)
            all_real_posts.extend(posts)

        # Remove duplicate post texts
        df_real = pd.DataFrame(all_real_posts)
        if not df_real.empty:
            df_real.drop_duplicates(subset=["text"], inplace=True)
            df_real["film_id"] = film_id
        
        # Save to CSV
        output_csv = os.path.join(data_dir, f"{film_id}_real_posts.csv")
        df_real.to_csv(output_csv, index=False)
        print(f"Saved {len(df_real)} real social posts to {output_csv}")

        # 2. Scrape real IMDb metadata and user reviews
        real_imdb_dict = scrape_real_imdb_metadata_and_reviews(imdb_id, film_title)
        
        # Load or update IMDb movie data
        existing_imdb = imdb_loader.load_movie_data(film_id, config)
        
        if real_imdb_dict["reviews"]:
            df_reviews = pd.DataFrame(real_imdb_dict["reviews"])
            existing_imdb.reviews = df_reviews
        if real_imdb_dict["rating"] > 0:
            existing_imdb.rating = real_imdb_dict["rating"]
        if real_imdb_dict["total_votes"] > 0:
            existing_imdb.total_votes = real_imdb_dict["total_votes"]
        if real_imdb_dict["genres"]:
            existing_imdb.genres = real_imdb_dict["genres"]

        imdb_loader.save_movie_data(existing_imdb)
        print(f"Updated real IMDb profile for {film_title} (Rating: {existing_imdb.rating}, Reviews: {len(existing_imdb.reviews)})")

    print("\nAll real datasets built successfully!")

if __name__ == "__main__":
    build_all_real_datasets()
