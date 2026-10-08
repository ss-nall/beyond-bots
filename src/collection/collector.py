import os
import time
import pandas as pd
import yaml
from typing import List, Dict, Any
from .file_parser import parse_input_file

class XCollector:
    """X/Twitter collector with support for local file parsing and X API v2 recent search fetching."""
    def __init__(self, config_path_or_dict: Any = "config.yaml"):
        if isinstance(config_path_or_dict, dict):
            self.config = config_path_or_dict
        else:
            with open(config_path_or_dict, "r") as f:
                self.config = yaml.safe_load(f)
            
        # Load local .env file if present
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")

        self.queries = self.config.get("queries", ["#bollywood", "#movies"])
        self.bearer_token = os.environ.get("X_BEARER_TOKEN") or os.environ.get("TWITTER_BEARER_TOKEN") or self.config.get("x_credentials", {}).get("bearer_token")
        
        self.x_client = None
        if self.bearer_token:
            self.x_client = True

    def generate_synthetic_data(self, film_id: str, film_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generates realistic, code-mixed social media posts containing coordinated commercial,
        coordinated fan, and organic review properties for testing.
        """
        title = film_config.get("title", "Film")
        release_date = pd.to_datetime(film_config.get("release_date"))
        start_time = int(time.mktime(release_date.timetuple()))
        
        posts = []
        
        # 1. Paid-commercial templated clusters (high marketing vocabulary, low specificity, uniform sentiment)
        # We construct a cluster of 6 posts that look like templates from a PR agency
        commercial_templates = [
            "Guys, {title} is an absolute masterpiece! Sensational work! Box office rampage starts today! Must watch! #blockbuster",
            "{title} review: Unprecedented blockbuster! pure cinema. goosebumps entry scene. records broken on day 1! #masterpiece",
            "Watched {title} today. Sensational blockbuster of the year! Atlee magic is visible. Must watch, go with family! #atlee",
            "OMG, {title} is a blockbuster! Pure masterpiece! Mass film of the year, records broken already! Sensational! #epic",
            "{title} is pure cinema, unprecedented box office rampage! Atlee magic at its peak. Blockbuster of the decade! #hype",
            "No words for {title}, pure masterpiece and blockbuster. Unprecedented records broken! Must watch guys! #mustwatch"
        ]
        for idx, template in enumerate(commercial_templates):
            posts.append({
                "id": f"comm_{film_id}_{idx}",
                "text": template.format(title=title),
                "author": f"pr_bot_{idx}",
                "created_utc": start_time + 3600 * idx,  # bursty timeline
                "query": self.queries[0] if self.queries else "film",
                "parent_id": None,
                "film_id": film_id
            })

        # 2. Fan-coordinated posts (fan honorifics, high sentiment uniformity, star-loyal keywords)
        fan_honorifics = film_config.get("fan_markers", ["star"])
        fan_templates = [
            "All hail {honorific}! {title} theater response is crazy. King Khan rule starts. {honorific} badshah for a reason! 🔥",
            "{title} first day response is insane! Only {honorific} badshah can do this! King Khan badshah showing who is boss.",
            "Historical opening for {title}! SRK badshah did it again. {honorific} fan-club response at Gaiety Galaxy is nuts! #fans",
            "Pure goosebumps entry of {honorific} in {title}! Best movie ever. King Khan badshah for a reason. Fan-club celebrating!",
            "Craziest response for {honorific} in {title} trailer. badshah returning to show who is the real king khan! 🔥 #srk"
        ]
        for idx, template in enumerate(fan_templates):
            honorific = fan_honorifics[idx % len(fan_honorifics)].upper()
            posts.append({
                "id": f"fan_{film_id}_{idx}",
                "text": template.format(honorific=honorific, title=title),
                "author": f"fan_army_{idx}",
                "created_utc": start_time + 7200 * idx,
                "query": self.queries[idx % len(self.queries)] if self.queries else "film",
                "parent_id": f"comm_{film_id}_0" if idx % 2 == 0 else None,  # reply connections
                "film_id": film_id
            })

        # 3. Organic posts (detailed reviews, mixed sentiment, scene specific references)
        organic_templates = [
            "Watched {title}. The first half was excellent but the second half felt dragged. Screenplay could be better.",
            "{title} review: Action scenes choreographed by Atlee were good. However, the logic was missing in many scenes. 6/10.",
            "I went to PVR Forum Mall to watch {title}. The background music by Anirudh was nice, but story was copy of standard mass movies.",
            "Honestly {title} was just average. Too much hype. It will do good box office but it's not a great film. Nayanthara role was weak.",
            "Can someone explain the metro hijack scene in {title}? It made no sense at all. How did police allow him to escape?"
        ]
        for idx, template in enumerate(organic_templates):
            posts.append({
                "id": f"org_{film_id}_{idx}",
                "text": template.format(title=title),
                "author": f"critic_user_{idx}",
                "created_utc": start_time + 14400 * idx,
                "query": self.queries[1] if len(self.queries) > 1 else (self.queries[0] if self.queries else "film"),
                "parent_id": None,
                "film_id": film_id
            })

        return pd.DataFrame(posts)

    def collect_data(self, film_id: str) -> pd.DataFrame:
        """
        Collects submissions and comments matching the keywords for the given film.
        Prioritizes:
          1. Loading from local input file (if configured and exists).
          2. Fetching from X API v2 (if Bearer Token exists).
          3. Generating simulated X/Twitter data (as fallback).
        """
        film_config = next((f for f in self.config.get("films", []) if f["id"] == film_id), None)
        if not film_config:
            raise ValueError(f"Film ID '{film_id}' not found in configuration.")

        # 1. Ingestion stage: Local File Input
        input_file = self.config.get("collection_settings", {}).get("input_file")
        if input_file and os.path.exists(input_file):
            print(f"[Stage 0: Ingestion] Loading pre-scraped reviews from file: {input_file}...")
            try:
                df_loaded = parse_input_file(input_file, film_id, self.config)
                if not df_loaded.empty:
                    print(f"[Stage 0: Ingestion] Successfully loaded {len(df_loaded)} records from local file.")
                    return df_loaded
                else:
                    print(f"[Stage 0: Ingestion] Local file contains 0 records matching '{film_config['title']}'. Falling back to API/synthetic.")
            except Exception as e:
                print(f"[Stage 0: Ingestion] Error reading file: {e}. Falling back to API/synthetic.")

        # 2. X API Collection
        if not self.x_client:
            print("No X API credentials found (X_BEARER_TOKEN or TWITTER_BEARER_TOKEN).")
            print("Generating realistic synthetic X/Twitter data for testing...")
            return self.generate_synthetic_data(film_id, film_config)

        print(f"X API Bearer token found. Sourcing tweets for '{film_config['title']}'...")
        import requests
        
        keywords = film_config.get("keywords", [])
        collected = []
        limit = self.config.get("collection_settings", {}).get("max_posts_per_query", 50)
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        
        query = " OR ".join([f'"{kw}"' for kw in keywords])
        params = {
            "query": query,
            "max_results": min(100, max(10, limit)),
            "tweet.fields": "author_id,created_at,referenced_tweets",
            "expansions": "author_id"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                res_data = response.json()
                tweets = res_data.get("data", [])
                users = {u["id"]: u["username"] for u in res_data.get("includes", {}).get("users", [])}
                
                for tweet in tweets:
                    author_id = tweet.get("author_id")
                    username = users.get(author_id, author_id)
                    
                    # Convert created_at ISO 8601 string to unix epoch
                    created_at_str = tweet.get("created_at")
                    try:
                        dt_str = created_at_str.replace("Z", "")
                        if "." in dt_str:
                            dt_str = dt_str.split(".")[0]
                        timestamp = int(time.mktime(time.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")))
                    except Exception:
                        timestamp = int(time.time())
                        
                    # Find reply ID
                    parent_id = None
                    ref_tweets = tweet.get("referenced_tweets", [])
                    for ref in ref_tweets:
                        if ref.get("type") == "replied_to":
                            parent_id = ref.get("id")
                            break
                            
                    collected.append({
                        "id": tweet.get("id"),
                        "text": tweet.get("text"),
                        "author": username,
                        "created_utc": timestamp,
                        "query": self.queries[0] if self.queries else "film",
                        "parent_id": parent_id,
                        "film_id": film_id
                    })
            else:
                print(f"X API search request failed with status: {response.status_code} - {response.text}")
        except Exception as api_err:
            print(f"X API search request error: {api_err}")
            
        if not collected:
            print("No real tweets found or API error. Returning synthetic dataset.")
            return self.generate_synthetic_data(film_id, film_config)
            
        return pd.DataFrame(collected)

def save_raw_data(df: pd.DataFrame, output_dir: str = "./data/raw") -> str:
    """Saves raw data to a parquet file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raw_posts.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Raw data saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Self-test block
    collector = XCollector()
    df = collector.collect_data("jawan_2023")
    print(f"Collected {len(df)} posts.")
    print("Sample posts:")
    print(df[["id", "author", "text"]].head())
    save_raw_data(df, "./data/raw_test")
