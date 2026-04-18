"""
============================================================
  EMOTION ANALYSIS ON SOCIAL MEDIA - Part 2/6
  🧹 Data Preprocessing
============================================================
  Steps:
    1.  Expand contractions
    2.  Remove noise (URLs, @mentions, HTML)
    3.  Preserve emotion-carrying punctuation signals
        (!! → very_strong, ? → questioning, ...)
    4.  Lowercase + punctuation removal
    5.  Tokenize
    6.  Remove stopwords (keep negations + intensifiers)
    7.  Lemmatize
    8.  Handle class imbalance stats (for training weights)
    9.  Stratified 70 / 15 / 15 split
  Input:  data/raw/raw_data.csv
  Output: data/processed/  train / val / test .csv
          data/processed/class_weights.json
============================================================
"""

import re
import json
import string
import numpy as np
import pandas as pd
from pathlib import Path

import nltk
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
RAW_FILE      = BASE_DIR / "data" / "raw" / "raw_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_STATE = 42
MIN_TOKENS   = 2     # drop samples with fewer tokens after cleaning

lemmatizer = WordNetLemmatizer()

# Keep negations + intensifiers — critical for emotion detection
STOP_WORDS = set(stopwords.words("english"))
KEEP_WORDS = {
    "no", "not", "nor", "never", "nobody", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "very", "really", "so", "too",
    "extremely", "absolutely", "completely", "totally", "quite",
    "rather", "almost", "just", "only", "even", "still", "already",
}
STOP_WORDS -= KEEP_WORDS

# ── Contraction expansion ─────────────────────────────────
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "that's": "that is", "what's": "what is", "who's": "who is",
    "there's": "there is", "here's": "here is", "let's": "let us",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "could've": "could have", "would've": "would have", "should've": "should have",
}


# ═══════════════════════════════════════════════════════════
#  EMOTION SIGNAL PRESERVATION
#  Before stripping punctuation, capture surface signals
#  that carry strong emotional cues.
# ═══════════════════════════════════════════════════════════

def encode_punctuation_signals(text: str) -> str:
    """
    Replace punctuation patterns with token proxies BEFORE
    lowercasing + punctuation removal, so the signal survives.
    """
    # Repeated exclamation → strong emphasis
    text = re.sub(r"!!+", " very_strong_emphasis ", text)
    text = re.sub(r"!",   " strong_emphasis ", text)
    # Question marks → uncertainty/curiosity
    text = re.sub(r"\?+", " questioning_signal ", text)
    # Ellipsis → trailing emotion / hesitation
    text = re.sub(r"\.{3,}", " trailing_emotion ", text)
    # ALL CAPS word → shouting / strong emotion
    text = re.sub(r"\b([A-Z]{2,})\b", lambda m: m.group(0).lower() + " shouted_word", text)
    return text


# ═══════════════════════════════════════════════════════════
#  CLEANING PIPELINE
# ═══════════════════════════════════════════════════════════

def expand_contractions(text: str) -> str:
    text = text.lower()
    for c, e in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(c) + r"\b", e, text)
    text = re.sub(r"n't\b", " not", text)
    return text


def remove_noise(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+",   "", text)   # URLs
    text = re.sub(r"@\w+",                       "", text)   # mentions
    text = re.sub(r"#(\w+)",              r" \1 ", text)     # hashtags → word
    text = re.sub(r"<[^>]+>",                   "", text)    # HTML
    text = re.sub(r"[^\x00-\x7F]+",            " ", text)    # non-ASCII / emoji
    text = re.sub(r"(.)\1{2,}",           r"\1\1", text)     # loooove → loove
    return text.strip()


def preprocess(text: str) -> str:
    text = encode_punctuation_signals(str(text))
    text = expand_contractions(text)
    text = remove_noise(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and len(t) >= 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════
#  BATCH PROCESSING
# ═══════════════════════════════════════════════════════════

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    print(f"[INFO] Preprocessing {total:,} samples...")
    cleaned = []
    for i, text in enumerate(df["text"]):
        cleaned.append(preprocess(text))
        if (i + 1) % 5000 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:5.1f}%  ({i+1:,}/{total:,})", end="", flush=True)
    print()

    df = df.copy()
    df["clean_text"] = cleaned
    # Drop samples that became empty or too short after cleaning
    df = df[df["clean_text"].str.split().str.len() >= MIN_TOKENS].reset_index(drop=True)
    print(f"[INFO] {len(df):,} samples after cleaning (min {MIN_TOKENS} tokens)")
    return df


# ═══════════════════════════════════════════════════════════
#  CLASS WEIGHTS (for imbalanced 28-class problem)
# ═══════════════════════════════════════════════════════════

def compute_weights(y_train: np.ndarray, classes: list) -> dict:
    """
    Compute balanced class weights for loss / sample_weight.
    GoEmotions is significantly imbalanced — neutral alone is ~34%.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=y_train
    )
    return {str(cls): float(w) for cls, w in zip(classes, weights)}


# ═══════════════════════════════════════════════════════════
#  SPLIT
# ═══════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame):
    X = df.drop(columns=["emotion"])
    y = df["emotion"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO), stratify=y, random_state=RANDOM_STATE
    )
    adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - adj), stratify=y_temp, random_state=RANDOM_STATE
    )

    train_df = X_train.copy(); train_df["emotion"] = y_train.values
    val_df   = X_val.copy();   val_df["emotion"]   = y_val.values
    test_df  = X_test.copy();  test_df["emotion"]  = y_test.values
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ═══════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════

def summarize_splits(train, val, test):
    print("\n" + "═" * 62)
    print("  🧹 PREPROCESSING COMPLETE")
    print("═" * 62)
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        top5 = df["emotion"].value_counts().head(5)
        print(f"\n  {name} ({len(df):,} samples) — top 5 emotions:")
        for em, n in top5.items():
            print(f"    {em:<20}: {n:>5,}  ({n/len(df)*100:.1f}%)")
    print("═" * 62 + "\n")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 62)
    print("  🚀 PART 2: DATA PREPROCESSING")
    print("═" * 62)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing: {RAW_FILE}\n→ Run 1_data_collection.py first.")

    df = pd.read_csv(RAW_FILE)
    # Convert stringified lists back
    print(f"[INFO] Loaded: {len(df):,} rows, {df['emotion'].nunique()} emotions")

    df = preprocess_df(df)

    # ── Encode emotion as integer label ───────────────────
    emotions_sorted = sorted(df["emotion"].unique())
    emotion2id      = {e: i for i, e in enumerate(emotions_sorted)}
    df["label_id"]  = df["emotion"].map(emotion2id)

    # ── Split ─────────────────────────────────────────────
    train_df, val_df, test_df = split_data(df)

    # ── Class weights (from train only!) ──────────────────
    classes = sorted(train_df["emotion"].unique())
    weights = compute_weights(train_df["emotion"].values, classes)

    # ── Save ──────────────────────────────────────────────
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR   / "val.csv",   index=False)
    test_df.to_csv(PROCESSED_DIR  / "test.csv",  index=False)

    with open(PROCESSED_DIR / "class_weights.json", "w") as f:
        json.dump({
            "weights":     weights,
            "emotion2id":  emotion2id,
            "id2emotion":  {str(v): k for k, v in emotion2id.items()},
            "n_classes":   len(emotions_sorted),
        }, f, indent=2)

    summarize_splits(train_df, val_df, test_df)

    print(f"[✓] train.csv       ({len(train_df):,} rows) → {PROCESSED_DIR}")
    print(f"[✓] val.csv         ({len(val_df):,} rows)")
    print(f"[✓] test.csv        ({len(test_df):,} rows)")
    print(f"[✓] class_weights.json  ({len(classes)} classes)")


if __name__ == "__main__":
    main()
