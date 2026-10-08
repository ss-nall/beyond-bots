import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Any

def parse_input_file(file_path: str, film_id: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Parses a local file (.txt, .pdf, .csv, .docx, .doc) containing pre-scraped reviews
    and formats the extracted text/data into the standardized BeyondBots DataFrame schema.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    
    # Resolve film config for default temporal boundaries
    film_config = next((f for f in config.get("films", []) if f["id"] == film_id), {})
    film_title = film_config.get("title", film_id.split("_")[0])
    release_date_str = film_config.get("release_date", "2023-09-07")
    
    try:
        release_time = int(time.mktime(time.strptime(release_date_str, "%Y-%m-%d")))
    except Exception:
        release_time = int(time.time())
        
    texts = []
    authors = []
    ids = []
    created_utcs = []
    parent_ids = []
    
    if ext == ".csv":
        df = pd.read_csv(file_path)
        
        # Filter by film if column is present in the CSV
        if "film" in df.columns:
            df = df[df["film"].str.lower() == film_title.lower()]
            
        # Identify text column
        text_col = None
        for col in ["text", "body", "tweet", "content", "review"]:
            if col in df.columns:
                text_col = col
                break
        if not text_col:
            # Fallback to first string/object column
            str_cols = df.select_dtypes(include=[object]).columns
            if len(str_cols) > 0:
                text_col = str_cols[0]
            else:
                raise ValueError(f"No text column found in CSV file: {file_path}")
                
        # Parse rows
        for idx, row in df.reset_index(drop=True).iterrows():
            texts.append(str(row[text_col]))
            
            author_col = next((c for c in ["username", "author", "user"] if c in df.columns), None)
            author = str(row[author_col]) if author_col and pd.notna(row[author_col]) else f"user_{idx}"
            authors.append(author)
            
            id_col = next((c for c in ["post_id", "id", "tweet_id"] if c in df.columns), None)
            id_val = str(row[id_col]) if id_col and pd.notna(row[id_col]) else f"tweet_{idx}"
            ids.append(id_val)
            
            # Timestamp (epoch or ISO format string)
            timestamp_col = next((c for c in ["timestamp", "created_utc", "created_at"] if c in df.columns), None)
            if timestamp_col and pd.notna(row[timestamp_col]):
                val = row[timestamp_col]
                if isinstance(val, str):
                    try:
                        # e.g., '2023-10-31T18:06:57.844558' or '2023-10-31 18:06:57'
                        dt_str = val.replace("Z", "")
                        if "." in dt_str:
                            dt_str = dt_str.split(".")[0]
                        timestamp = int(time.mktime(time.strptime(dt_str.replace(" ", "T"), "%Y-%m-%dT%H:%M:%S")))
                    except Exception:
                        timestamp = release_time + idx * 3600
                else:
                    timestamp = int(val)
            else:
                timestamp = release_time + idx * 3600
            created_utcs.append(timestamp)
            
            parent = str(row["parent_id"]) if "parent_id" in df.columns and pd.notna(row["parent_id"]) else None
            parent_ids.append(parent)
            
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # Split by double newline or blank lines
        raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        if len(raw_chunks) <= 1:
            # Fallback to single line split
            raw_chunks = [line.strip() for line in content.split("\n") if line.strip()]
            
        for idx, chunk in enumerate(raw_chunks):
            texts.append(chunk)
            authors.append(f"user_{idx}")
            ids.append(f"tweet_{idx}")
            created_utcs.append(release_time + idx * 3600)
            # Simulate a reply network structure (every 5th post replies to the previous one)
            parent_ids.append(f"tweet_{idx-1}" if idx > 0 and idx % 5 == 0 else None)
            
    elif ext in [".docx", ".doc"]:
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for idx, p in enumerate(paragraphs):
                texts.append(p)
                authors.append(f"user_{idx}")
                ids.append(f"tweet_{idx}")
                created_utcs.append(release_time + idx * 3600)
                parent_ids.append(f"tweet_{idx-1}" if idx > 0 and idx % 5 == 0 else None)
        except Exception as docx_err:
            if ext == ".doc":
                # Fallback parser for binary .doc files by stripping non-ASCII characters
                with open(file_path, "rb") as f:
                    content_bytes = f.read()
                text_chars = []
                for b in content_bytes:
                    if 32 <= b <= 126 or b in (10, 13, 9):
                        text_chars.append(chr(b))
                raw_text = "".join(text_chars)
                lines = [l.strip() for l in raw_text.split("\n") if len(l.strip()) > 15]
                if not lines:
                    raise ValueError(f"Failed to extract readable text from legacy binary .doc file: {file_path}")
                for idx, l in enumerate(lines):
                    texts.append(l)
                    authors.append(f"user_{idx}")
                    ids.append(f"tweet_{idx}")
                    created_utcs.append(release_time + idx * 3600)
                    parent_ids.append(f"tweet_{idx-1}" if idx > 0 and idx % 5 == 0 else None)
            else:
                raise ValueError(f"Error reading Word document {file_path}: {docx_err}")
                
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            chunks = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Try splitting by double-newlines first
                    page_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
                    if len(page_chunks) <= 1:
                        # Fallback to splitting by line (only lines with substantive text)
                        page_chunks = [line.strip() for line in text.split("\n") if len(line.strip()) > 15]
                    chunks.extend(page_chunks)
            if not chunks:
                raise ValueError("No text extracted from PDF pages.")
            for idx, chunk in enumerate(chunks):
                texts.append(chunk)
                authors.append(f"user_{idx}")
                ids.append(f"tweet_{idx}")
                created_utcs.append(release_time + idx * 3600)
                parent_ids.append(f"tweet_{idx-1}" if idx > 0 and idx % 5 == 0 else None)
        except Exception as pdf_err:
            raise ValueError(f"Error reading PDF file {file_path}: {pdf_err}")
            
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    df_out = pd.DataFrame({
        "id": ids,
        "text": texts,
        "author": authors,
        "created_utc": created_utcs,
        "query": [film_config.get("title", "Film")] * len(texts),
        "parent_id": parent_ids,
        "film_id": [film_id] * len(texts)
    })
    
    return df_out
