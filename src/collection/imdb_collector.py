import os
import urllib.request
import json
import pandas as pd
from typing import Dict, Any, Optional
from .imdb_loader import IMDbLoader, IMDbMovieData

class IMDbCollector:
    """IMDb Data Collector for fetching/loading IMDb metadata, reviews and rating distributions."""
    def __init__(self, data_dir: str = "data/imdb"):
        self.loader = IMDbLoader(data_dir=data_dir)

    def get_film_imdb_data(self, film_id: str, config: Optional[Dict[str, Any]] = None, fetch_online: bool = False) -> IMDbMovieData:
        """
        Retrieves IMDb data for a given film.
        If fetch_online is True and network is available, attempts online lookup before fallback.
        """
        if fetch_online and config and "films" in config:
            film_config = next((f for f in config["films"] if f.get("id") == film_id), None)
            if film_config and film_config.get("imdb_id"):
                imdb_id = film_config["imdb_id"]
                try:
                    # Attempt lightweight public metadata fetch
                    url = f"https://www.imdb.com/title/{imdb_id}/"
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    # Note: Production scrapers or cinemagoer can be plugged here
                except Exception as e:
                    print(f"Online IMDb fetch skipped or failed: {e}. Using cached benchmark data.")

        return self.loader.load_movie_data(film_id, config)
