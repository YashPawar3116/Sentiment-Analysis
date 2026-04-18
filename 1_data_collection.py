"""
============================================================
  EMOTION ANALYSIS ON SOCIAL MEDIA - Part 1/6
  📊 Data Collection
============================================================
  Dataset : GoEmotions (Google Research, 2020)
            58,009 Reddit comments
            27 emotion categories + neutral
            Best-in-class for fine-grained emotion NLP
  Source  : Hugging Face  →  "google-research-datasets/go_emotions"
  Output  : data/raw/raw_data.csv
            data/raw/emotion_labels.json
============================================================
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── GoEmotions: 27 emotion classes ────────────────────────
# Organized by Plutchik wheel clusters for later visualization
EMOTION_GROUPS = {
    "positive":  ["admiration", "amusement", "approval", "caring", "curiosity",
                  "desire", "excitement", "gratitude", "joy", "love",
                  "optimism", "pride", "relief"],
    "negative":  ["anger", "annoyance", "disappointment", "disapproval",
                  "disgust", "embarrassment", "fear", "grief", "nervousness",
                  "remorse", "sadness"],
    "ambiguous": ["confusion", "realization", "surprise"],
    "neutral":   ["neutral"],
}

ALL_EMOTIONS = (
    EMOTION_GROUPS["positive"] +
    EMOTION_GROUPS["negative"] +
    EMOTION_GROUPS["ambiguous"] +
    EMOTION_GROUPS["neutral"]
)   # 28 total (27 emotions + neutral)

# Simplified grouping for optional 6-class training
SIMPLIFIED_MAP = {
    "joy": ["joy", "amusement", "excitement", "relief", "gratitude", "pride", "love"],
    "sadness":  ["sadness", "grief", "remorse", "disappointment"],
    "anger":    ["anger", "annoyance", "disapproval", "disgust"],
    "fear":     ["fear", "nervousness"],
    "surprise": ["surprise", "realization", "confusion"],
    "other":    ["admiration", "approval", "caring", "curiosity", "desire",
                 "optimism", "embarrassment", "neutral"],
}


# ═══════════════════════════════════════════════════════════
#  LOADER
# ═══════════════════════════════════════════════════════════

def load_go_emotions():
    """
    Load GoEmotions via Hugging Face datasets.
    The dataset has 3 splits: train / validation / test.
    Each sample has multi-label annotations; we convert to
    the PRIMARY emotion (highest-confidence / first label)
    for single-label classification, and keep all labels
    for multi-label analysis.
    """
    print("[INFO] Downloading GoEmotions from Hugging Face...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

    frames = []
    for split_name in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split_name])
        df["_split"] = split_name
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Raw rows loaded: {len(raw):,}")
    return raw


# ═══════════════════════════════════════════════════════════
#  PROCESSING
# ═══════════════════════════════════════════════════════════

def process(raw: pd.DataFrame) -> pd.DataFrame:
    """
    GoEmotions 'simplified' subset maps to 28 labels (27 + neutral).
    The 'labels' column is a list of integer label IDs.
    We:
      1. Keep the original multi-label list as 'all_labels'
      2. Extract the primary label (first element) as 'label_id'
      3. Map label_id → emotion string → group
    """
    # GoEmotions simplified label list (index → emotion name)
    label_names = dataset_label_names()

    rows = []
    for _, row in raw.iterrows():
        labels = row["labels"]  # list of ints
        if not labels:
            continue

        primary_id = labels[0]
        emotion    = label_names[primary_id]
        group      = emotion_group(emotion)

        rows.append({
            "text":        row["text"],
            "label_id":    primary_id,
            "emotion":     emotion,
            "group":       group,
            "all_labels":  labels,
            "all_emotions":[label_names[l] for l in labels],
            "_split":      row["_split"],
        })

    df = pd.DataFrame(rows)
    df.dropna(subset=["text", "emotion"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df["text"] = df["text"].astype(str)
    return df


def dataset_label_names():
    """
    GoEmotions simplified subset: 28 labels in alphabetical order.
    These are fixed by the dataset — do not reorder.
    """
    return [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ]


def emotion_group(emotion: str) -> str:
    for group, emotions in EMOTION_GROUPS.items():
        if emotion in emotions:
            return group
    return "neutral"


# ═══════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════

def summarize(df: pd.DataFrame):
    print("\n" + "═" * 62)
    print("  📊 DATASET SUMMARY — GoEmotions")
    print("═" * 62)
    print(f"  Total samples     : {len(df):,}")
    print(f"  Unique emotions   : {df['emotion'].nunique()}")
    print(f"  Avg text length   : {df['text'].str.split().str.len().mean():.1f} tokens")
    print(f"  Source splits     : {df['_split'].value_counts().to_dict()}")

    print("\n  Emotion Distribution (by group):")
    for group, emotions in EMOTION_GROUPS.items():
        subset = df[df["group"] == group]
        pct    = len(subset) / len(df) * 100
        bar    = "█" * int(pct / 3)
        print(f"\n  {group.upper():<12} ({pct:.1f}%)")
        for em in emotions:
            n   = (df["emotion"] == em).sum()
            ep  = n / len(df) * 100
            bar = "▪" * int(ep * 1.5)
            print(f"    {em:<18}: {n:>5,}  ({ep:4.1f}%)  {bar}")

    print("\n  Sample texts:")
    for em in ["joy", "anger", "fear", "surprise", "love"]:
        sample = df[df["emotion"] == em].sample(1, random_state=42).iloc[0]
        print(f"    [{em:<12}] {sample['text'][:70]}...")
    print("═" * 62 + "\n")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 62)
    print("  🚀 PART 1: DATA COLLECTION")
    print("  Dataset: GoEmotions (Google Research)")
    print("  28 emotion classes · 58k Reddit comments")
    print("═" * 62)

    # ── Download & process ────────────────────────────────
    raw = load_go_emotions()
    df  = process(raw)

    # ── Save raw CSV ──────────────────────────────────────
    df.to_csv(RAW_DIR / "raw_data.csv", index=False)

    # ── Save label metadata ───────────────────────────────
    meta = {
        "all_emotions":    ALL_EMOTIONS,
        "emotion_groups":  EMOTION_GROUPS,
        "simplified_map":  SIMPLIFIED_MAP,
        "label_names":     dataset_label_names(),
        "n_classes":       len(dataset_label_names()),
        "total_samples":   len(df),
        "class_counts":    df["emotion"].value_counts().to_dict(),
    }
    with open(RAW_DIR / "emotion_labels.json", "w") as f:
        json.dump(meta, f, indent=2)

    summarize(df)

    print(f"[✓] raw_data.csv      → {RAW_DIR / 'raw_data.csv'}  ({len(df):,} rows)")
    print(f"[✓] emotion_labels.json → {RAW_DIR / 'emotion_labels.json'}")


if __name__ == "__main__":
    main()
