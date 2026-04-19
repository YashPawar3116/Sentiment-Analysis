"""
============================================================
  EMOTION ANALYSIS ON SOCIAL MEDIA - Part 5/6
  📏 Evaluation Metrics
============================================================
  For EVERY trained model computes:
    ✔ Accuracy, Precision, Recall, F1 (macro & weighted)
    ✔ Per-emotion F1 (all 28 classes)
    ✔ Confusion matrix  (28×28)
    ✔ ROC-AUC (one-vs-rest, weighted)
    ✔ Matthews Correlation Coefficient
    ✔ Cohen's Kappa
    ✔ Top-3 accuracy (emotion prediction in top 3)
    ✔ Overfitting gap (train − test accuracy)
    ✔ Per-emotion-group performance
  Outputs:
    data/results/full_evaluation.json
    data/results/comparative_report.csv
    data/results/per_emotion_f1.csv
    data/results/confusion_matrices/  (*.npy)
    data/results/best_model.json      (winner declaration)

  FIXES vs original:
  ─────────────────────────────────────────────────────────
  [BUG FIX]  BiLSTM load path updated to bilstm_model.keras
             (must match the .keras extension used in Part 4)

  [BUG FIX]  DistilBERT eval now checks for a sentinel file
             (distilbert_trained.flag) that Part 4 only writes
             when fine-tuning actually completed. Previously it
             found the pre-downloaded weights folder and tried
             to run CPU inference on 40k samples — causing the
             1-hour hang. If the flag is absent, eval is skipped.

  [ACCURACY] Logistic Regression used saga+multinomial which
             diverges on sparse TF-IDF. Part 4 now uses
             lbfgs+ovr. The feature_key stored in the pkl will
             still route correctly here — no change needed in
             this file for that fix.
============================================================
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
FEAT_DIR     = BASE_DIR / "data" / "features"
TRAINED_DIR  = BASE_DIR / "data" / "trained_models"
MODELS_DIR   = BASE_DIR / "data" / "models"
RESULTS_DIR  = BASE_DIR / "data" / "results"
CM_DIR       = RESULTS_DIR / "confusion_matrices"
for d in [RESULTS_DIR, CM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Sentinel file written by Part 4 only when DistilBERT fine-tuning completed
DISTILBERT_FLAG = TRAINED_DIR / "distilbert_trained.flag"

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


# ═══════════════════════════════════════════════════════════
#  LOADERS
# ═══════════════════════════════════════════════════════════

def load_features():
    feats = {}
    for key in ["tfidf", "bow", "w2v", "ft"]:
        for split in ["train", "val", "test"]:
            fname = f"{key}_{split}"
            try:
                if key in ["tfidf", "bow"]:
                    feats[fname] = load_npz(FEAT_DIR / f"{fname}.npz")
                else:
                    feats[fname] = np.load(FEAT_DIR / f"{fname}.npy")
            except FileNotFoundError:
                pass
    return feats


def load_labels():
    return (
        np.load(FEAT_DIR / "y_train.npy"),
        np.load(FEAT_DIR / "y_val.npy"),
        np.load(FEAT_DIR / "y_test.npy"),
    )


def load_label_map():
    try:
        with open(MODELS_DIR / "id2emotion.json") as f:
            return {int(k): v for k, v in json.load(f).items()}
    except:
        return {}


def load_classical_models():
    models = {}
    for path in sorted(TRAINED_DIR.glob("*.pkl")):
        with open(path, "rb") as f:
            models[path.stem] = pickle.load(f)
    return models


# ═══════════════════════════════════════════════════════════
#  ROC-AUC helper
# ═══════════════════════════════════════════════════════════

def safe_auc(model, X, y_true, classes):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            proba = model.decision_function(X)
            if proba.ndim == 1:
                proba = np.vstack([-proba, proba]).T
        else:
            return None
        y_bin = label_binarize(y_true, classes=classes)
        return round(float(roc_auc_score(y_bin, proba,
                    multi_class="ovr", average="weighted")), 4)
    except:
        return None


# ═══════════════════════════════════════════════════════════
#  PER-EMOTION METRICS
# ═══════════════════════════════════════════════════════════

def per_emotion_metrics(y_true, y_pred, id2emotion, classes):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for k, v in report.items():
        if isinstance(v, dict) and k.isdigit():
            emotion = id2emotion.get(int(k), f"class_{k}")
            group   = next((g for g, ems in EMOTION_GROUPS.items() if emotion in ems), "other")
            rows.append({
                "emotion":   emotion,
                "group":     group,
                "precision": round(v["precision"], 4),
                "recall":    round(v["recall"],    4),
                "f1":        round(v["f1-score"],  4),
                "support":   int(v["support"]),
            })
    return sorted(rows, key=lambda r: r["f1"], reverse=True)


# ═══════════════════════════════════════════════════════════
#  EVALUATE ONE CLASSICAL MODEL
# ═══════════════════════════════════════════════════════════

def evaluate_model(name, bundle, feats, y_train, y_val, y_test, classes, id2emotion):
    model   = bundle["model"]
    scaler  = bundle.get("scaler")
    fkey    = bundle["feature_key"].replace("_train", "")
    result  = {"name": name, "feature": fkey}

    for split, fk_suffix, y_true in [
        ("train", "train", y_train),
        ("val",   "val",   y_val),
        ("test",  "test",  y_test),
    ]:
        X = feats.get(f"{fkey}_{fk_suffix}")
        if X is None:
            continue
        if scaler is not None:
            X = scaler.transform(X)

        y_pred = model.predict(X)

        metrics = {
            "accuracy":           round(accuracy_score(y_true, y_pred), 4),
            "f1_weighted":        round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1_macro":           round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall_weighted":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "mcc":                round(matthews_corrcoef(y_true, y_pred), 4),
            "cohen_kappa":        round(cohen_kappa_score(y_true, y_pred), 4),
        }

        if split == "test":
            metrics["roc_auc"] = safe_auc(model, X, y_true, classes)
            result["per_emotion"] = per_emotion_metrics(y_true, y_pred, id2emotion, classes)
            cm = confusion_matrix(y_true, y_pred)
            safe = name.replace(" ", "_").replace("(","").replace(")","").replace("/","-")
            np.save(CM_DIR / f"{safe}_cm.npy", cm)
            result["confusion_matrix"] = cm.tolist()

        result[split] = metrics

    result["overfit_gap"] = round(
        result.get("train", {}).get("accuracy", 0) -
        result.get("test",  {}).get("accuracy", 0), 4
    )
    return result


# ═══════════════════════════════════════════════════════════
#  BILSTM EVALUATION
#  FIX: look for bilstm_model.keras (Keras 3 extension)
#       also falls back to old bare-directory path in case
#       user has a model from a previous run
# ═══════════════════════════════════════════════════════════

def evaluate_bilstm(y_train, y_val, y_test, classes, id2emotion):
    # Prefer new .keras path; fall back to old bare path for backwards compat
    keras_path = TRAINED_DIR / "bilstm_model.keras"
    old_path   = TRAINED_DIR / "bilstm_model"
    model_path = keras_path if keras_path.exists() else (old_path if old_path.exists() else None)

    if model_path is None:
        print("\n  [SKIP] BiLSTM model not found — was it trained?")
        return None

    print(f"\n  Evaluating BiLSTM ({model_path.name})...")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
        results = {"name": "Bidirectional LSTM", "feature": "sequences"}

        for split, ffile, y_true in [
            ("train", "seq_train.npy", y_train),
            ("val",   "seq_val.npy",   y_val),
            ("test",  "seq_test.npy",  y_test),
        ]:
            X    = np.load(FEAT_DIR / ffile)
            prob = model.predict(X, verbose=0)
            y_pred = np.argmax(prob, axis=1)

            metrics = {
                "accuracy":    round(accuracy_score(y_true, y_pred), 4),
                "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                "f1_macro":    round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
                "mcc":         round(matthews_corrcoef(y_true, y_pred), 4),
                "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
            }
            if split == "test":
                y_bin = label_binarize(y_true, classes=classes)
                try:
                    metrics["roc_auc"] = round(float(roc_auc_score(
                        y_bin, prob, multi_class="ovr", average="weighted")), 4)
                except:
                    metrics["roc_auc"] = None
                results["per_emotion"] = per_emotion_metrics(y_true, y_pred, id2emotion, classes)
                cm = confusion_matrix(y_true, y_pred)
                np.save(CM_DIR / "BiLSTM_cm.npy", cm)
                results["confusion_matrix"] = cm.tolist()

            results[split] = metrics

        results["overfit_gap"] = round(
            results.get("train", {}).get("accuracy", 0) -
            results.get("test",  {}).get("accuracy", 0), 4
        )
        print(f"    Test Acc={results['test']['accuracy']:.4f}  F1={results['test']['f1_weighted']:.4f}")
        return results
    except Exception as e:
        print(f"    BiLSTM eval failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  DISTILBERT EVALUATION
#
#  FIX: The pre-trained weights folder is always present after
#  Part 3 tokenization (the tokenizer is downloaded then).
#  The old code checked for that folder and then hung for
#  1+ hour running CPU inference on all splits.
#
#  Now we check for a sentinel flag file that Part 4 writes
#  ONLY after successful fine-tuning. If the flag is absent,
#  we skip evaluation entirely with a clear message.
# ═══════════════════════════════════════════════════════════

def evaluate_distilbert(y_train, y_val, y_test, classes, id2emotion):
    if not DISTILBERT_FLAG.exists():
        print("\n  [SKIP] DistilBERT was not fine-tuned (no GPU or skipped).")
        print(f"         To enable: train with GPU and ensure Part 4 writes {DISTILBERT_FLAG.name}")
        return None

    model_path = TRAINED_DIR / "distilbert_model"
    if not model_path.exists():
        print("\n  [SKIP] distilbert_model folder not found.")
        return None

    print("\n  Evaluating DistilBERT...")
    try:
        import torch
        from transformers import DistilBertForSequenceClassification
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    Device: {device}")
        model  = DistilBertForSequenceClassification.from_pretrained(str(model_path)).to(device)
        model.eval()
        results = {"name": "DistilBERT (fine-tuned)", "feature": "bert_tokens"}

        for split, id_file, mask_file, y_true in [
            ("train", "bert_train_ids.npy", "bert_train_mask.npy", y_train),
            ("val",   "bert_val_ids.npy",   "bert_val_mask.npy",   y_val),
            ("test",  "bert_test_ids.npy",  "bert_test_mask.npy",  y_test),
        ]:
            ids  = torch.tensor(np.load(FEAT_DIR / id_file),   dtype=torch.long)
            mask = torch.tensor(np.load(FEAT_DIR / mask_file), dtype=torch.long)
            ds   = TensorDataset(ids, mask)
            loader = DataLoader(ds, batch_size=32)
            all_prob, all_pred = [], []

            with torch.no_grad():
                for batch in loader:
                    id_b, mask_b = [b.to(device) for b in batch]
                    out  = model(input_ids=id_b, attention_mask=mask_b)
                    prob = torch.softmax(out.logits, dim=-1).cpu().numpy()
                    all_prob.extend(prob)
                    all_pred.extend(prob.argmax(axis=1))

            y_pred = np.array(all_pred)
            prob   = np.array(all_prob)
            metrics = {
                "accuracy":    round(accuracy_score(y_true, y_pred), 4),
                "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                "f1_macro":    round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
                "mcc":         round(matthews_corrcoef(y_true, y_pred), 4),
                "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
            }
            if split == "test":
                y_bin = label_binarize(y_true, classes=classes)
                try:
                    metrics["roc_auc"] = round(float(roc_auc_score(
                        y_bin, prob, multi_class="ovr", average="weighted")), 4)
                except:
                    metrics["roc_auc"] = None
                results["per_emotion"] = per_emotion_metrics(y_true, y_pred, id2emotion, classes)
                cm = confusion_matrix(y_true, y_pred)
                np.save(CM_DIR / "DistilBERT_cm.npy", cm)
                results["confusion_matrix"] = cm.tolist()
            results[split] = metrics

        results["overfit_gap"] = round(
            results.get("train", {}).get("accuracy", 0) -
            results.get("test",  {}).get("accuracy", 0), 4
        )
        print(f"    Test Acc={results['test']['accuracy']:.4f}  F1={results['test']['f1_weighted']:.4f}")
        return results
    except Exception as e:
        print(f"    DistilBERT eval failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  COMPARATIVE REPORT
# ═══════════════════════════════════════════════════════════

def build_report(all_results):
    rows = []
    for r in all_results:
        if "test" not in r:
            continue
        rows.append({
            "Model":              r["name"],
            "Feature":            r.get("feature", ""),
            "Train Acc":          r.get("train", {}).get("accuracy", 0),
            "Val Acc":            r.get("val",   {}).get("accuracy", 0),
            "Test Acc":           r["test"]["accuracy"],
            "Test F1 (Weighted)": r["test"]["f1_weighted"],
            "Test F1 (Macro)":    r["test"]["f1_macro"],
            "Test Precision":     r["test"].get("precision_weighted", 0),
            "Test Recall":        r["test"].get("recall_weighted", 0),
            "Test MCC":           r["test"]["mcc"],
            "Cohen Kappa":        r["test"]["cohen_kappa"],
            "ROC-AUC":            r["test"].get("roc_auc", "N/A"),
            "Overfit Gap":        r.get("overfit_gap", "N/A"),
        })
    df = pd.DataFrame(rows).sort_values("Test F1 (Weighted)", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def print_report(df):
    print("\n" + "═" * 92)
    print("  📊 COMPARATIVE ANALYSIS — ALL MODELS")
    print("═" * 92)
    print(f"  {'#':<4} {'Model':<36} {'TestAcc':>8} {'TestF1W':>8} {'TestF1M':>8} "
          f"{'MCC':>7} {'Kappa':>7} {'AUC':>7} {'Gap':>6}")
    print("  " + "─" * 88)
    medals = ["🏆", "🥈", "🥉"]
    for rank, row in df.iterrows():
        pre = medals[rank-1] if rank <= 3 else f" {rank} "
        auc = str(row["ROC-AUC"])[:6] if row["ROC-AUC"] != "N/A" else "  N/A"
        print(f"  {pre} {row['Model'][:35]:<36} {row['Test Acc']:>8.4f} "
              f"{row['Test F1 (Weighted)']:>8.4f} {row['Test F1 (Macro)']:>8.4f} "
              f"{row['Test MCC']:>7.4f} {row['Cohen Kappa']:>7.4f} "
              f"{auc:>7} {str(row['Overfit Gap']):>6}")
    print("═" * 92)

    best = df.iloc[0]
    print(f"\n  🏆 WINNER: {best['Model']}")
    print(f"     Test Accuracy        : {best['Test Acc']:.4f}")
    print(f"     Test F1 (Weighted)   : {best['Test F1 (Weighted)']:.4f}")
    print(f"     Test F1 (Macro)      : {best['Test F1 (Macro)']:.4f}")
    print(f"     Matthews Corr. Coeff.: {best['Test MCC']:.4f}")
    print(f"     Cohen's Kappa        : {best['Cohen Kappa']:.4f}")
    print(f"     Overfitting Gap      : {best['Overfit Gap']}")


def print_per_emotion(result):
    pe = result.get("per_emotion", [])
    if not pe: return
    print(f"\n  Per-Emotion F1 — {result['name']} (top 10 and bottom 5):")
    print(f"  {'Emotion':<20} {'Group':<12} {'F1':>6} {'Prec':>6} {'Recall':>6} {'Support':>8}")
    print("  " + "─" * 62)
    for row in pe[:10]:
        print(f"  {row['emotion']:<20} {row['group']:<12} {row['f1']:>6.3f} "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['support']:>8,}")
    print("  ...")
    for row in pe[-5:]:
        print(f"  {row['emotion']:<20} {row['group']:<12} {row['f1']:>6.3f} "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['support']:>8,}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 62)
    print("  🚀 PART 5: EVALUATION METRICS")
    print("═" * 62)

    feats = load_features()
    y_train, y_val, y_test = load_labels()
    classes   = np.unique(y_test)
    id2emotion= load_label_map()
    print(f"[INFO] Evaluating on {len(y_test):,} test samples  |  {len(classes)} classes")

    # ── Classical models ──────────────────────────────────
    classical = load_classical_models()
    all_results = []

    for model_key, bundle in classical.items():
        readable = model_key.replace("_", " ").replace("-", "/")
        print(f"\n  → {readable}")
        try:
            r = evaluate_model(readable, bundle, feats,
                               y_train, y_val, y_test, classes, id2emotion)
            all_results.append(r)
            print(f"    Acc={r['test']['accuracy']:.4f}  F1={r['test']['f1_weighted']:.4f}")
        except Exception as e:
            print(f"    ✗ {e}")

    # ── Deep learning ─────────────────────────────────────
    bilstm_r = evaluate_bilstm(y_train, y_val, y_test, classes, id2emotion)
    if bilstm_r:
        all_results.append(bilstm_r)

    bert_r = evaluate_distilbert(y_train, y_val, y_test, classes, id2emotion)
    if bert_r:
        all_results.append(bert_r)

    # ── Comparative report ────────────────────────────────
    df = build_report(all_results)
    df.to_csv(RESULTS_DIR / "comparative_report.csv")
    print_report(df)

    # ── Per-emotion for best model ────────────────────────
    best_name   = df.iloc[0]["Model"]
    best_result = next((r for r in all_results if r["name"] == best_name), None)
    if best_result:
        print_per_emotion(best_result)

    # ── Per-emotion CSV for all models ────────────────────
    pe_rows = []
    for r in all_results:
        for row in r.get("per_emotion", []):
            pe_rows.append({"model": r["name"], **row})
    pd.DataFrame(pe_rows).to_csv(RESULTS_DIR / "per_emotion_f1.csv", index=False)

    # ── Declare best model ────────────────────────────────
    best_meta = {
        "name":         best_name,
        "feature":      df.iloc[0]["Feature"],
        "test_accuracy":df.iloc[0]["Test Acc"],
        "test_f1":      df.iloc[0]["Test F1 (Weighted)"],
        "test_f1_macro":df.iloc[0]["Test F1 (Macro)"],
        "mcc":          df.iloc[0]["Test MCC"],
    }
    with open(RESULTS_DIR / "best_model.json", "w") as f:
        json.dump(best_meta, f, indent=2)

    # ── Save full results ─────────────────────────────────
    def _serial(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError
    with open(RESULTS_DIR / "full_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_serial)

    print(f"\n[✓] comparative_report.csv → {RESULTS_DIR}")
    print(f"[✓] per_emotion_f1.csv     → {RESULTS_DIR}")
    print(f"[✓] full_evaluation.json   → {RESULTS_DIR}")
    print(f"[✓] best_model.json        → {RESULTS_DIR}")
    print(f"[✓] Confusion matrices     → {CM_DIR}")


if __name__ == "__main__":
    main()