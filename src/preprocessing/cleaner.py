import re
import emoji
import spacy
import pandas as pd
from typing import List, Dict, Tuple, Set

# Initialize spaCy blank multilingual model
try:
    nlp = spacy.blank("xx")
except Exception:
    nlp = spacy.blank("en")

# Indic NLP Library trivial tokenizer fallback wrapper
try:
    from indicnlp.tokenize import indic_tokenize
    has_indic_nlp = True
except ImportError:
    has_indic_nlp = False

# Common Indic words (both Romanized and native script) to help flag transliteration/code-mixing
COMMON_ROMANIZED_INDIC = {
    # Hindi/Urdu
    "hai", "hain", "aur", "ka", "ki", "ke", "mein", "ko", "se", "par", "ek", "yeh", "woh", 
    "bhi", "to", "hi", "ho", "kar", "kya", "tha", "thi", "the", "re", "yaar", "bhai", "accha",
    "achha", "film", "movie", "mast", "hit", "super", "flopo", "flopa", "badiya", "badhiya",
    "dhamaka", "maza", "maja", "dekh", "dekha", "dekho", "aaj", "kal", "ab", "kab", "sab",
    "kuch", "hoga", "gaya", "gayi", "gaye", "karta", "kr", "karne", "lia", "liya", "diya",
    # Tamil/Telugu/Kannada/Malayalam
    "anna", "garu", "ra", "da", "thala", "thalapathy", "superstar", "rebel", "mass", "class",
    "cinema", "hero", "director", "bgm", "teaser", "trailer", "song", "babu", "boss", "mawa",
    "mama", "machas", "machan", "nanba", "konda", "chudu", "bagundi", "baagundi", "chusa",
    "chusaara", "superb", "chitra", "keka", "adurs", "bomma", "padam", "pavam", "nalla",
    "romba", "ila", "ela", "ala", "eppudu", "appudu", "ippudu", "nenu", "nuvvu", "memu"
}

INDIC_SCRIPT_RANGES = [
    (0x0900, 0x097F), # Devanagari
    (0x0980, 0x09FF), # Bengali
    (0x0A00, 0x0A7F), # Gurmukhi
    (0x0A80, 0x0AFF), # Gujarati
    (0x0B00, 0x0B7F), # Oriya
    (0x0B80, 0x0BFF), # Tamil
    (0x0C00, 0x0C7F), # Telugu
    (0x0C80, 0x0CFF), # Kannada
    (0x0D00, 0x0D7F), # Malayalam
]

# Combined English + Indic stopwords
ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

INDIC_STOPWORDS = {
    # Devanagari common stopwords
    "है", "हैं", "और", "का", "की", "के", "में", "को", "से", "पर", "एक", "यह", "वह", "भी", 
    "तो", "ही", "हो", "कर", "था", "थी", "थे", "ने", "इस", "उस", "जो", "करते", "करना",
    # Romanized common Indic stopwords
    "hai", "hain", "aur", "ka", "ki", "ke", "mein", "ko", "se", "par", "ek", "yeh", "woh", 
    "bhi", "to", "hi", "ho", "kar", "tha", "thi", "the", "ne", "is", "us", "jo"
}

ALL_STOPWORDS = ENGLISH_STOPWORDS.union(INDIC_STOPWORDS)

def clean_text(text: str) -> str:
    """
    Strips URLs, markdown links, converts emojis to text representations,
    and returns cleaned string.
    """
    if not isinstance(text, str):
        return ""
    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Strip markdown URL patterns [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\(\S+\)', r'\1', text)
    # Strip basic markdown characters (*, _, #, `, ~)
    text = re.sub(r'[*_#`~]', '', text)
    # Normalize emoji to text
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Replace underscores from demojize with spaces to help tokenizer
    text = text.replace("_", " ")
    # Replace multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_indic_char(char: str) -> bool:
    """Returns True if the char lies in any Indic script unicode range."""
    o = ord(char)
    for start, end in INDIC_SCRIPT_RANGES:
        if start <= o <= end:
            return True
    return False

def is_romanized_indic_token(token: str) -> bool:
    """Flags if a Latin token is likely Romanized Indic text based on a common lexicon."""
    val = token.lower()
    return val in COMMON_ROMANIZED_INDIC

def tokenize_code_mixed(text: str) -> List[str]:
    """
    Tokenizes text. Uses indic-nlp tokenizer for native scripts,
    falls back to spaCy multilingual tokenizer.
    """
    if not text:
        return []
    
    # Check if text contains Indic native characters
    has_native = any(is_indic_char(c) for c in text)
    
    if has_native and has_indic_nlp:
        try:
            return indic_tokenize.trivial_tokenize(text)
        except Exception:
            pass # fallback to spacy
            
    # spaCy blank multilingual tokenization
    doc = nlp(text)
    return [t.text for t in doc if not t.is_space]

def normalize_tokens(tokens: List[str]) -> Tuple[List[str], List[bool]]:
    """
    Lowercases English spans only (does not lowercase Devanagari or other Indic scripts).
    Detectes and flags transliterated tokens.
    Returns (normalized_tokens, transliterated_mask).
    """
    normalized = []
    trans_mask = []
    for token in tokens:
        # Check if token contains native Indic chars
        is_native = any(is_indic_char(c) for c in token)
        if is_native:
            # Keep case for native scripts
            normalized.append(token)
            trans_mask.append(False)
        else:
            # Lowercase English/Latin script
            normalized_token = token.lower()
            normalized.append(normalized_token)
            # Detect transliteration
            trans_mask.append(is_romanized_indic_token(normalized_token))
            
    return normalized, trans_mask

def process_stopwords(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Produces two tokenized versions:
    1. tokens_clean: stopwords removed
    2. tokens_retained: stopwords retained (needed for honorific/identity detection)
    """
    tokens_clean = [t for t in tokens if t.lower() not in ALL_STOPWORDS and re.match(r'^\w+$', t)]
    # Keep punctuation/special tokens out of the clean list too, but keep alphanumeric
    tokens_retained = [t for t in tokens if re.match(r'^\w+$', t)]
    return tokens_clean, tokens_retained

def preprocess_post(text: str) -> Dict:
    """Preprocesses a single text field and returns features."""
    cleaned = clean_text(text)
    raw_tokens = tokenize_code_mixed(cleaned)
    normalized, trans_mask = normalize_tokens(raw_tokens)
    tokens_clean, tokens_retained = process_stopwords(normalized)
    
    # Calculate ratios
    total_latin_tokens = sum(1 for t in normalized if not any(is_indic_char(c) for c in t))
    romanized_indic_count = sum(1 for m in trans_mask if m)
    
    trans_ratio = romanized_indic_count / total_latin_tokens if total_latin_tokens > 0 else 0.0
    
    # Check if native Indic chars exist
    has_native = any(any(is_indic_char(c) for c in t) for t in normalized)
    
    return {
        "cleaned_text": cleaned,
        "tokens_clean": tokens_clean,
        "tokens_retained": tokens_retained,
        "is_transliterated": trans_ratio > 0.1,  # flag if >10% of latin tokens match Romanized Indic lexicon
        "script_mix_ratio": trans_ratio if not has_native else 0.5,
        "has_emoji": "emoji" in cleaned.lower() or emoji.emoji_count(text) > 0
    }

def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Processes a pandas DataFrame through Stage 1 preprocessing."""
    results = []
    for idx, row in df.iterrows():
        text_val = row[text_column] if text_column in df.columns else ""
        results.append(preprocess_post(text_val))
        
    res_df = pd.DataFrame(results)
    # Combine with original df metadata
    out_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
    return out_df

if __name__ == "__main__":
    # Self-test block
    test_text = "This film was amazing! SRK was superb. bilkul paisa vasool movie tha! [EMOJI_HEART_EYES] #Jawan"
    print("Raw text:", test_text)
    res = preprocess_post(test_text)
    print("Cleaned text:", res["cleaned_text"])
    print("Tokens clean:", res["tokens_clean"])
    print("Tokens retained:", res["tokens_retained"])
    print("Is Transliterated:", res["is_transliterated"])
    print("Has Emoji:", res["has_emoji"])
