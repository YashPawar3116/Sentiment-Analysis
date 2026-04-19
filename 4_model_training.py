"""
============================================================
  EMOTION ANALYSIS ON SOCIAL MEDIA - Part 4/6
  🤖 Model Training
============================================================
  7 models trained:
  ┌──────────────────────────┬───────────────┬────────────┐
  │ Model                    │ Features      │ Type       │
  ├──────────────────────────┼───────────────┼────────────┤
  │ 1. Logistic Regression   │ TF-IDF        │ Classical  │
  │ 2. Multinomial NB        │ TF-IDF        │ Classical  │
  │ 3. Bernoulli NB          │ BoW (binary)  │ Classical  │
  │ 4. Linear SVM            │ TF-IDF        │ Classical  │
  │ 5. Random Forest         │ TF-IDF        │ Ensemble   │
  │ 6. XGBoost               │ TF-IDF        │ Ensemble   │
  │ 7. Bidirectional LSTM    │ Sequences     │ Deep Learn │
  │ 8. DistilBERT (finetune) │ BERT tokens   │ Transformer│
  └──────────────────────────┴───────────────┴────────────┘

  FIXES vs original:
  ─────────────────────────────────────────────────────────
  [BUG FIX]  BiLSTM crash: save path now uses .keras extension
             (Keras 3 requires it — was saving bare directory path)

  [OVERFIT]  BiLSTM severely overfit (train 77% vs val 36%):
             • Added SpatialDropout1D after Embedding (0.3)
             • Reduced LSTM units: 128/64 → 64/32
             • Increased Dropout: 0.4/0.3 → 0.5/0.4
             • Removed second Dense layer (was causing overfit)
             • Reduced max_epochs: 30 → 15
             • EarlyStopping patience: 5 → 3

  [SPEED]    RF n_estimators: 300 → 100
             XGBoost n_estimators: 300 → 100
             LR max_iter: 500 → 300

  [SPEED]    DistilBERT: added CPU guard — prints a clear warning
             and skips on CPU (would take 3+ hrs). Remove the guard
             if you have a GPU (cuda) available.
             N_EPOCHS: 5 → 3 (saves ~40% time on GPU)
============================================================
"""

import os, json, time, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes   import MultinomialNB, BernoulliNB
from sklearn.svm           import LinearSVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import accuracy_score, f1_score
from sklearn.preprocessing import MaxAbsScaler
from xgboost               import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
FEAT_DIR     = BASE_DIR / "data" / "features"
MODELS_DIR   = BASE_DIR / "data" / "models"
TRAINED_DIR  = BASE_DIR / "data" / "trained_models"
RESULTS_DIR  = BASE_DIR / "data" / "results"
for d in [TRAINED_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_JOBS       = -1


# ═══════════════════════════════════════════════════════════
#  LOADERS
# ═══════════════════════════════════════════════════════════

def load_features():
    return {
        "tfidf_train": load_npz(FEAT_DIR / "tfidf_train.npz"),
        "tfidf_val":   load_npz(FEAT_DIR / "tfidf_val.npz"),
        "tfidf_test":  load_npz(FEAT_DIR / "tfidf_test.npz"),
        "bow_train":   load_npz(FEAT_DIR / "bow_train.npz"),
        "bow_val":     load_npz(FEAT_DIR / "bow_val.npz"),
        "bow_test":    load_npz(FEAT_DIR / "bow_test.npz"),
        "w2v_train":   np.load(FEAT_DIR / "w2v_train.npy"),
        "w2v_val":     np.load(FEAT_DIR / "w2v_val.npy"),
        "w2v_test":    np.load(FEAT_DIR / "w2v_test.npy"),
        "y_train":     np.load(FEAT_DIR / "y_train.npy"),
        "y_val":       np.load(FEAT_DIR / "y_val.npy"),
        "y_test":      np.load(FEAT_DIR / "y_test.npy"),
    }


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ═══════════════════════════════════════════════════════════
#  CLASSICAL MODEL TRAINING
# ═══════════════════════════════════════════════════════════

def train_classical(name, model, X_train, y_train, X_val, y_val, scaler_needed=False):
    print(f"\n  ┌─ {name}")
    start = time.time()
    scaler = None

    if scaler_needed:
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print(f"  │  Time: {elapsed:.1f}s  |  Val Acc: {acc:.4f}  |  Val F1: {f1:.4f}")
    print(f"  └─ Done")
    return model, scaler, acc, f1, elapsed


def run_classical_models(data):
    y_train, y_val = data["y_train"], data["y_val"]
    results        = []

    print("\n" + "─" * 55)
    print("  CLASSICAL MODELS")
    print("─" * 55)

    configs = [
        # (name, model, tr_key, val_key, needs_scaler)
        ("Logistic Regression (TF-IDF)",
            LogisticRegression(
                C=1.0, solver="saga", max_iter=300,   # was 500
                multi_class="multinomial", n_jobs=N_JOBS,
                class_weight="balanced", random_state=RANDOM_STATE),
            "tfidf_train", "tfidf_val", False),

        ("Multinomial NB (TF-IDF)",
            MultinomialNB(alpha=0.1),
            "tfidf_train", "tfidf_val", True),

        ("Bernoulli NB (BoW)",
            BernoulliNB(alpha=0.3),
            "bow_train", "bow_val", False),

        ("Linear SVM (TF-IDF)",
            LinearSVC(C=0.5, max_iter=3000, class_weight="balanced",
                multi_class="ovr", random_state=RANDOM_STATE),
            "tfidf_train", "tfidf_val", False),

        ("Random Forest (TF-IDF)",
            RandomForestClassifier(
                n_estimators=100,       # was 300 — 3x faster, ~1% accuracy drop
                max_depth=30,
                class_weight="balanced",
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE),
            "tfidf_train", "tfidf_val", False),

        ("XGBoost (TF-IDF)",
            XGBClassifier(
                n_estimators=100,       # was 300
                learning_rate=0.1,
                max_depth=6,
                eval_metric="mlogloss",
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                use_label_encoder=False),
            "tfidf_train", "tfidf_val", False),
    ]

    for name, model, tr_key, val_key, scaler_needed in configs:
        try:
            trained, scaler, acc, f1, elapsed = train_classical(
                name, model, data[tr_key], y_train, data[val_key], y_val, scaler_needed
            )
            safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
            save_pkl({"model": trained, "scaler": scaler, "feature_key": tr_key},
                     TRAINED_DIR / f"{safe}.pkl")
            results.append({
                "name": name, "type": "classical",
                "feature": tr_key.replace("_train", ""),
                "val_accuracy": round(acc, 4),
                "val_f1":       round(f1, 4),
                "train_time_s": round(elapsed, 2),
            })
        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            results.append({"name": name, "error": str(e)})

    return results


# ═══════════════════════════════════════════════════════════
#  BIDIRECTIONAL LSTM
#
#  Root cause of original overfitting (train 77% / val 36%):
#    • Model was too large for the task (128+64 LSTM units,
#      two dense layers) with insufficient regularisation.
#    • recurrent_dropout slows CPU training significantly.
#  Fixes applied:
#    • Smaller LSTM units (64/32)
#    • SpatialDropout1D after Embedding (drops entire feature
#      maps rather than individual tokens — better for NLP)
#    • Higher dropout (0.5/0.4)
#    • Single dense layer
#    • Shorter training (15 epochs, patience=3)
#    • Correct .keras save extension (Keras 3 requirement)
# ═══════════════════════════════════════════════════════════

def train_bilstm(y_train, y_val):
    print("\n" + "─" * 55)
    print("  BIDIRECTIONAL LSTM")
    print("─" * 55)

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, Bidirectional, LSTM, Dense,
            Dropout, SpatialDropout1D, GlobalMaxPooling1D,
        )
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        import json

        with open(MODELS_DIR / "lstm_config.json") as f:
            cfg = json.load(f)

        vocab_size  = cfg["vocab_size"]
        max_seq_len = cfg["max_seq_len"]
        embed_dim   = cfg["embed_dim"]
        n_classes   = len(np.unique(y_train))

        X_train = np.load(FEAT_DIR / "seq_train.npy")
        X_val   = np.load(FEAT_DIR / "seq_val.npy")

        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", y_train)

        print(f"\n  Building BiLSTM: vocab={vocab_size:,}  seq_len={max_seq_len}  classes={n_classes}")

        model = Sequential([
            # SpatialDropout1D drops whole embedding dims — better than
            # regular Dropout for sequential/NLP data
            Embedding(vocab_size, embed_dim, input_length=max_seq_len, mask_zero=True),
            SpatialDropout1D(0.3),

            # Smaller LSTM units to reduce overfitting
            # Removed recurrent_dropout — it's correct but very slow on CPU
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
            Bidirectional(LSTM(32, return_sequences=True, dropout=0.3)),
            GlobalMaxPooling1D(),

            # Single dense layer (was two — extra capacity caused overfit)
            Dense(128, activation="relu"),
            Dropout(0.5),

            Dense(n_classes, activation="softmax"),
        ])

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )
        model.summary(print_fn=lambda x: print(f"    {x}"))

        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=3,                # was 5 — stop faster once plateaued
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,                # was 3
                verbose=1,
            ),
        ]

        start = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs          = 15,           # was 30
            batch_size      = 128,
            sample_weight   = sample_weights,
            callbacks       = callbacks,
            verbose         = 1,
        )
        elapsed = time.time() - start

        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        # FIX: Keras 3 requires .keras extension — bare directory path crashed
        save_path = str(TRAINED_DIR / "bilstm_model.keras")
        model.save(save_path)
        print(f"\n  BiLSTM — Val Acc: {acc:.4f}  Val F1: {f1:.4f}  Time: {elapsed:.0f}s")

        return {
            "name": "Bidirectional LSTM",
            "type": "deep_learning",
            "feature": "sequences",
            "val_accuracy": round(acc, 4),
            "val_f1":       round(f1, 4),
            "train_time_s": round(elapsed, 1),
            "epochs_run":   len(history.history["accuracy"]),
        }

    except ImportError:
        print("  [SKIP] TensorFlow not installed: pip install tensorflow")
        return {"name": "Bidirectional LSTM", "error": "TensorFlow not installed"}
    except Exception as e:
        print(f"  ✗ BiLSTM FAILED: {e}")
        return {"name": "Bidirectional LSTM", "error": str(e)}


# ═══════════════════════════════════════════════════════════
#  DISTILBERT FINE-TUNING
#
#  CPU GUARD: DistilBERT on CPU takes 3+ hours for 28 classes
#  on 40k samples. The guard below skips it when no GPU is
#  found. To force CPU training anyway (e.g. overnight run),
#  set FORCE_DISTILBERT_CPU = True below.
#
#  N_EPOCHS reduced 5 → 3: the log showed val_f1=0.375 at
#  epoch 1 already — 3 epochs is plenty with early stopping.
# ═══════════════════════════════════════════════════════════

FORCE_DISTILBERT_CPU = False   # set True to run on CPU (slow!)

def train_distilbert(y_train, y_val):
    print("\n" + "─" * 55)
    print("  DISTILBERT FINE-TUNING")
    print("─" * 55)

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import (DistilBertForSequenceClassification,
                                  get_linear_schedule_with_warmup)
        from torch.optim import AdamW

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CPU guard — remove or set FORCE_DISTILBERT_CPU=True to bypass
        if str(device) == "cpu" and not FORCE_DISTILBERT_CPU:
            print("\n  [SKIP] No GPU detected.")
            print("  DistilBERT on CPU takes 3+ hours for 28 classes.")
            print("  Options:")
            print("    1. Run on a machine with a CUDA GPU")
            print("    2. Set FORCE_DISTILBERT_CPU = True at the top of this file")
            print("    3. Use Google Colab (free T4 GPU)")
            return {
                "name": "DistilBERT (fine-tuned)",
                "type": "transformer",
                "feature": "bert_tokens",
                "error": "skipped — no GPU (set FORCE_DISTILBERT_CPU=True to override)",
            }

        bert_tr_ids  = np.load(FEAT_DIR / "bert_train_ids.npy")
        bert_tr_mask = np.load(FEAT_DIR / "bert_train_mask.npy")
        bert_va_ids  = np.load(FEAT_DIR / "bert_val_ids.npy")
        bert_va_mask = np.load(FEAT_DIR / "bert_val_mask.npy")

        n_classes    = len(np.unique(y_train))
        BATCH_SIZE   = 32 if str(device) == "cuda" else 16
        N_EPOCHS     = 3    # was 5 — epoch 1 already gave val_f1=0.375; 3 is enough
        LR           = 2e-5
        PATIENCE     = 2

        print(f"\n  Device: {device}  |  Classes: {n_classes}  |  Batch: {BATCH_SIZE}  |  Epochs: {N_EPOCHS}")

        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(cw, dtype=torch.float).to(device)

        train_ds = TensorDataset(
            torch.tensor(bert_tr_ids,  dtype=torch.long),
            torch.tensor(bert_tr_mask, dtype=torch.long),
            torch.tensor(y_train,      dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(bert_va_ids,  dtype=torch.long),
            torch.tensor(bert_va_mask, dtype=torch.long),
            torch.tensor(y_val,        dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=(str(device)=="cuda"))
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  num_workers=0, pin_memory=(str(device)=="cuda"))

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=n_classes
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * N_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = total_steps // 10,
            num_training_steps = total_steps,
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        best_val_f1  = 0
        best_acc     = 0
        patience_cnt = 0
        start        = time.time()

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            tr_losses = []
            for batch in train_loader:
                ids, mask, labels = [b.to(device) for b in batch]
                optimizer.zero_grad()
                out  = model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(out.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                tr_losses.append(loss.item())

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    ids, mask, labels = [b.to(device) for b in batch]
                    out   = model(input_ids=ids, attention_mask=mask)
                    preds = out.logits.argmax(-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
            print(f"  Epoch {epoch}/{N_EPOCHS}  loss={np.mean(tr_losses):.4f}"
                  f"  val_acc={acc:.4f}  val_f1={f1:.4f}")

            if f1 > best_val_f1:
                best_val_f1  = f1
                best_acc     = acc
                patience_cnt = 0
                model.save_pretrained(str(TRAINED_DIR / "distilbert_model"))
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        elapsed = time.time() - start
        print(f"\n  DistilBERT — Best Val Acc: {best_acc:.4f}  Best Val F1: {best_val_f1:.4f}"
              f"  Time: {elapsed:.0f}s")

        return {
            "name": "DistilBERT (fine-tuned)",
            "type": "transformer",
            "feature": "bert_tokens",
            "val_accuracy": round(best_acc, 4),
            "val_f1":       round(best_val_f1, 4),
            "train_time_s": round(elapsed, 1),
        }

    except ImportError:
        print("  [SKIP] transformers/torch not installed.")
        print("         Run: pip install transformers torch")
        return {"name": "DistilBERT (fine-tuned)", "error": "transformers/torch not installed"}
    except Exception as e:
        print(f"  ✗ DistilBERT FAILED: {e}")
        return {"name": "DistilBERT (fine-tuned)", "error": str(e)}


# ═══════════════════════════════════════════════════════════
#  LEADERBOARD
# ═══════════════════════════════════════════════════════════

def print_leaderboard(results):
    valid = [r for r in results if "val_f1" in r and "error" not in r]
    valid.sort(key=lambda r: r["val_f1"], reverse=True)

    print("\n" + "═" * 70)
    print("  📊 VALIDATION LEADERBOARD")
    print("═" * 70)
    print(f"  {'#':<3} {'Model':<38} {'Feature':<14} {'Acc':>6} {'F1':>6} {'Time':>7}")
    print("  " + "─" * 68)
    medals = ["🏆", "🥈", "🥉"]
    for i, r in enumerate(valid):
        prefix = medals[i] if i < 3 else f"{i+1:>2} "
        print(f"  {prefix} {r['name']:<38} {r.get('feature',''):<14} "
              f"{r.get('val_accuracy',0):>6.4f} {r.get('val_f1',0):>6.4f} "
              f"{r.get('train_time_s',0):>6.1f}s")

    # Also show skipped/failed models
    skipped = [r for r in results if "error" in r]
    if skipped:
        print("\n  Skipped / failed:")
        for r in skipped:
            print(f"    ✗ {r['name']}: {r['error']}")

    if valid:
        best = valid[0]
        print(f"\n  Best: {best['name']}  (F1={best['val_f1']:.4f})")
    print("═" * 70)
    return valid


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print("  🚀 PART 4: MODEL TRAINING")
    print("  7 models · classical + deep learning + transformer")
    print("═" * 70)

    data = load_features()
    y_train, y_val = data["y_train"], data["y_val"]
    n_classes = len(np.unique(y_train))
    print(f"\n[INFO] Classes: {n_classes}  |  Train: {len(y_train):,}  |  Val: {len(y_val):,}")

    # ── Classical models ──────────────────────────────────
    results = run_classical_models(data)

    # ── BiLSTM ────────────────────────────────────────────
    results.append(train_bilstm(y_train, y_val))

    # ── DistilBERT ────────────────────────────────────────
    results.append(train_distilbert(y_train, y_val))

    # ── Save scores ───────────────────────────────────────
    with open(RESULTS_DIR / "val_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Leaderboard ───────────────────────────────────────
    ranked = print_leaderboard(results)

    print(f"\n[✓] Models saved  → {TRAINED_DIR}")
    print(f"[✓] Val scores    → {RESULTS_DIR / 'val_scores.json'}")


if __name__ == "__main__":
    main()