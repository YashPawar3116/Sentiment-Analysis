"""
============================================================
  EMOTION ANALYSIS ON SOCIAL MEDIA
  🚀 run_all.py — Full Pipeline Orchestrator
============================================================
  Usage:
    python run_all.py              # Run all 6 parts
    python run_all.py --from 3    # Resume from Part 3
    python run_all.py --only 5    # Run only Part 5
    python run_all.py --skip 7    # Skip Part 7 (e.g. DistilBERT)
============================================================
"""

import sys, time, argparse, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent

PARTS = [
    (1, "📊 Data Collection",      "1_data_collection.py",
     "GoEmotions — 58k Reddit comments, 28 emotion classes"),
    (2, "🧹 Data Preprocessing",   "2_data_preprocessing.py",
     "Clean, lemmatize, preserve emotion signals, 70/15/15 split"),
    (3, "🔢 Feature Extraction",   "3_feature_extraction.py",
     "TF-IDF, BoW, Word2Vec, FastText, LSTM sequences, BERT tokens"),
    (4, "🤖 Model Training",       "4_model_training.py",
     "LR, NB, SVM, RF, XGBoost, BiLSTM, DistilBERT"),
    (5, "📏 Evaluation Metrics",   "5_evaluation_metrics.py",
     "Acc, F1, MCC, Kappa, ROC-AUC, per-emotion analysis, winner"),
    (6, "📊 Output Visualization", "6_output_visualization.py",
     "12 charts: radar, emotion wheel, t-SNE, word clouds, ROC"),
]


def run_part(num, name, script, desc):
    print("\n╔" + "═"*60 + "╗")
    print(f"║  PART {num}/6 — {name:<51}║")
    print(f"║  {desc:<59}║")
    print("╚" + "═"*60 + "╝\n")

    start = time.time()
    result = subprocess.run([sys.executable, str(BASE_DIR / script)], cwd=str(BASE_DIR))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n✗ Part {num} FAILED (exit {result.returncode})")
        print(f"  Fix errors above and resume with: python run_all.py --from {num}")
        sys.exit(result.returncode)

    print(f"\n  ✓ Part {num} completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", type=int, dest="from_part", default=1)
    parser.add_argument("--only", type=int, dest="only_part", default=None)
    args = parser.parse_args()

    to_run = [(n,nm,sc,d) for n,nm,sc,d in PARTS if n >= args.from_part] \
             if not args.only_part else \
             [(n,nm,sc,d) for n,nm,sc,d in PARTS if n == args.only_part]

    if not to_run:
        print("No valid parts specified."); sys.exit(1)

    print("\n" + "═"*62)
    print("  🎯 EMOTION ANALYSIS — FULL PIPELINE")
    print("  Dataset: GoEmotions (28 classes) · 7 models")
    print("═"*62)

    t0 = time.time(); timings = {}
    for num, name, script, desc in to_run:
        timings[num] = run_part(num, name, script, desc)

    total = time.time() - t0
    print("\n╔" + "═"*60 + "╗")
    print("║" + "  ✅ PIPELINE COMPLETE — Emotion Analysis".center(60) + "║")
    print("╚" + "═"*60 + "╝")
    print("\n  Timing:")
    for num, name, _, _ in to_run:
        if num in timings:
            print(f"    Part {num} — {name:<32}: {timings[num]:>6.1f}s")
    print(f"\n  Total: {total:.0f}s ({total/60:.1f} min)\n")
    print("  Key outputs:")
    print("    data/results/comparative_report.csv   ← model leaderboard")
    print("    data/results/per_emotion_f1.csv       ← per-emotion F1 (all models)")
    print("    data/results/best_model.json          ← winner declaration")
    print("    data/results/plots/  (12 charts)")


if __name__ == "__main__":
    main()
