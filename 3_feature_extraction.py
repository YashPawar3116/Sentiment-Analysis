"""
============================================================
  EMOTION ANALYSIS ON SOCIAL MEDIA - Part 3/6
  🔢 Feature Extraction
============================================================
  Produces 5 feature sets for 7 different models:
  ┌─────────────────┬────────────────────────────────────┐
  │ Feature Set     │ Used by                            │
  ├─────────────────┼────────────────────────────────────┤
  │ TF-IDF          │ Logistic Reg, NB, SVM, RF, XGBoost│
  │ BoW             │ Bernoulli NB                       │
  │ Word2Vec (avg)  │ Logistic Reg (dense), KNN          │
  │ FastText (avg)  │ Logistic Reg (dense)               │
  │ Sequences + pad │ BiLSTM                             │
  │ BERT tokens     │ DistilBERT fine-tune               │
  └─────────────────┴────────────────────────────────────┘
  Input:  data/processed/  train/val/test.csv
  Output: data/features/   *.npz / *.npy
          data/models/      vectorizers + tokenizer

  CHANGES vs original:
    - TFIDF_MAX_FEATURES  40k → 20k  (faster, negligible accuracy loss)
    - NGRAM_RANGE         (1,3) → (1,2)  (trigrams add little for emotion)
    - W2V_EPOCHS / FT_EPOCHS  15 → 5  (biggest time saver in this part)
    - W2V_VECTOR_SIZE / FT_VECTOR_SIZE  200 → 100  (half the RAM + time)
============================================================
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import save_npz

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

import gensim
from gensim.models import Word2Vec, FastText

# ── Paths ─────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR  = BASE_DIR / "data" / "features"
MODELS_DIR    = BASE_DIR / "data" / "models"
for d in [FEATURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────
TFIDF_MAX_FEATURES  = 20_000   # was 40_000 — halved, minimal accuracy impact
BOW_MAX_FEATURES    = 15_000   # was 25_000
NGRAM_RANGE         = (1, 2)   # was (1, 3) — trigrams slow TF-IDF with little gain
W2V_VECTOR_SIZE     = 100      # was 200 — halved: less RAM, same quality
W2V_WINDOW          = 5        # was 6
W2V_MIN_COUNT       = 2
W2V_EPOCHS          = 5        # was 15 — biggest speedup in this file
FT_VECTOR_SIZE      = 100      # was 200
FT_EPOCHS           = 5        # was 15
LSTM_MAX_VOCAB      = 20_000   # was 30_000
LSTM_MAX_SEQ_LEN    = 100
BERT_MODEL_NAME     = "distilbert-base-uncased"
BERT_MAX_LEN        = 128
RANDOM_STATE        = 42


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def load_splits():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val   = pd.read_csv(PROCESSED_DIR / "val.csv")
    test  = pd.read_csv(PROCESSED_DIR / "test.csv")
    for df in [train, val, test]:
        df["clean_text"] = df["clean_text"].fillna("").astype(str)
    return train, val, test


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    [✓] {path.name}")


def tokenize_series(series):
    return [t.split() for t in series]


# ═══════════════════════════════════════════════════════════
#  1. LABEL ENCODING
# ═══════════════════════════════════════════════════════════

def encode_labels(train, val, test):
    print("\n[INFO] Encoding labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(train["emotion"])
    y_val   = le.transform(val["emotion"])
    y_test  = le.transform(test["emotion"])

    np.save(FEATURES_DIR / "y_train.npy", y_train)
    np.save(FEATURES_DIR / "y_val.npy",   y_val)
    np.save(FEATURES_DIR / "y_test.npy",  y_test)
    save_pickle(le, MODELS_DIR / "label_encoder.pkl")

    id2emotion = {int(i): str(c) for i, c in enumerate(le.classes_)}
    with open(MODELS_DIR / "id2emotion.json", "w") as f:
        json.dump(id2emotion, f, indent=2)

    print(f"    Classes ({len(le.classes_)}): {list(le.classes_)}")
    return y_train, y_val, y_test, le


# ═══════════════════════════════════════════════════════════
#  2. TF-IDF
# ═══════════════════════════════════════════════════════════

def extract_tfidf(train, val, test):
    print("\n[INFO] TF-IDF extraction...")
    vec = TfidfVectorizer(
        max_features  = TFIDF_MAX_FEATURES,
        ngram_range   = NGRAM_RANGE,        # (1,2) — bigrams only
        sublinear_tf  = True,
        min_df        = 2,
        max_df        = 0.95,
        strip_accents = "unicode",
        analyzer      = "word",
    )
    Xtr = vec.fit_transform(train["clean_text"])
    Xv  = vec.transform(val["clean_text"])
    Xte = vec.transform(test["clean_text"])

    save_npz(FEATURES_DIR / "tfidf_train.npz", Xtr)
    save_npz(FEATURES_DIR / "tfidf_val.npz",   Xv)
    save_npz(FEATURES_DIR / "tfidf_test.npz",  Xte)
    save_pickle(vec, MODELS_DIR / "tfidf_vectorizer.pkl")
    print(f"    Shape: {Xtr.shape}")
    return Xtr, Xv, Xte, vec


# ═══════════════════════════════════════════════════════════
#  3. Bag of Words
# ═══════════════════════════════════════════════════════════

def extract_bow(train, val, test):
    print("\n[INFO] Bag-of-Words extraction...")
    vec = CountVectorizer(
        max_features = BOW_MAX_FEATURES,
        ngram_range  = (1, 2),
        min_df       = 2,
        max_df       = 0.95,
        binary       = True,
    )
    Xtr = vec.fit_transform(train["clean_text"])
    Xv  = vec.transform(val["clean_text"])
    Xte = vec.transform(test["clean_text"])

    save_npz(FEATURES_DIR / "bow_train.npz", Xtr)
    save_npz(FEATURES_DIR / "bow_val.npz",   Xv)
    save_npz(FEATURES_DIR / "bow_test.npz",  Xte)
    save_pickle(vec, MODELS_DIR / "bow_vectorizer.pkl")
    print(f"    Shape: {Xtr.shape}")
    return Xtr, Xv, Xte


# ═══════════════════════════════════════════════════════════
#  4. Word2Vec
# ═══════════════════════════════════════════════════════════

def _mean_vec(model, tokens, size):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(size)


def extract_word2vec(train, val, test):
    print("\n[INFO] Word2Vec training + extraction...")
    tr_tok = tokenize_series(train["clean_text"])
    va_tok = tokenize_series(val["clean_text"])
    te_tok = tokenize_series(test["clean_text"])

    model = Word2Vec(
        tr_tok,
        vector_size = W2V_VECTOR_SIZE,  # 100 (was 200)
        window      = W2V_WINDOW,
        min_count   = W2V_MIN_COUNT,
        workers     = 4,
        epochs      = W2V_EPOCHS,       # 5 (was 15)
        seed        = RANDOM_STATE,
    )
    model.save(str(MODELS_DIR / "word2vec.model"))
    print(f"    Vocab: {len(model.wv):,} words")

    Xtr = np.array([_mean_vec(model, t, W2V_VECTOR_SIZE) for t in tr_tok])
    Xv  = np.array([_mean_vec(model, t, W2V_VECTOR_SIZE) for t in va_tok])
    Xte = np.array([_mean_vec(model, t, W2V_VECTOR_SIZE) for t in te_tok])

    np.save(FEATURES_DIR / "w2v_train.npy", Xtr)
    np.save(FEATURES_DIR / "w2v_val.npy",   Xv)
    np.save(FEATURES_DIR / "w2v_test.npy",  Xte)
    print(f"    Shape: {Xtr.shape}")
    return Xtr, Xv, Xte


# ═══════════════════════════════════════════════════════════
#  5. FastText
# ═══════════════════════════════════════════════════════════

def extract_fasttext(train, val, test):
    print("\n[INFO] FastText training + extraction...")
    tr_tok = tokenize_series(train["clean_text"])
    va_tok = tokenize_series(val["clean_text"])
    te_tok = tokenize_series(test["clean_text"])

    model = FastText(
        tr_tok,
        vector_size = FT_VECTOR_SIZE,   # 100 (was 200)
        window      = W2V_WINDOW,
        min_count   = W2V_MIN_COUNT,
        workers     = 4,
        epochs      = FT_EPOCHS,        # 5 (was 15)
        seed        = RANDOM_STATE,
    )
    model.save(str(MODELS_DIR / "fasttext.model"))
    print(f"    Vocab: {len(model.wv):,} words (+ char n-grams for OOV)")

    Xtr = np.array([model.wv.get_mean_vector(t) if t else np.zeros(FT_VECTOR_SIZE) for t in tr_tok])
    Xv  = np.array([model.wv.get_mean_vector(t) if t else np.zeros(FT_VECTOR_SIZE) for t in va_tok])
    Xte = np.array([model.wv.get_mean_vector(t) if t else np.zeros(FT_VECTOR_SIZE) for t in te_tok])

    np.save(FEATURES_DIR / "ft_train.npy", Xtr)
    np.save(FEATURES_DIR / "ft_val.npy",   Xv)
    np.save(FEATURES_DIR / "ft_test.npy",  Xte)
    print(f"    Shape: {Xtr.shape}")
    return Xtr, Xv, Xte


# ═══════════════════════════════════════════════════════════
#  6. Integer Sequences (for BiLSTM)
# ═══════════════════════════════════════════════════════════

def extract_sequences(train, val, test):
    print("\n[INFO] Building integer sequences for LSTM...")
    from collections import Counter

    counter = Counter()
    for text in train["clean_text"]:
        counter.update(text.split())

    vocab = {w: i+2 for i, (w, _) in enumerate(counter.most_common(LSTM_MAX_VOCAB))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    def encode(text):
        tokens = text.split()[:LSTM_MAX_SEQ_LEN]
        ids    = [vocab.get(t, 1) for t in tokens]
        ids   += [0] * (LSTM_MAX_SEQ_LEN - len(ids))
        return ids

    Xtr = np.array([encode(t) for t in train["clean_text"]], dtype=np.int32)
    Xv  = np.array([encode(t) for t in val["clean_text"]],   dtype=np.int32)
    Xte = np.array([encode(t) for t in test["clean_text"]],  dtype=np.int32)

    np.save(FEATURES_DIR / "seq_train.npy", Xtr)
    np.save(FEATURES_DIR / "seq_val.npy",   Xv)
    np.save(FEATURES_DIR / "seq_test.npy",  Xte)
    save_pickle(vocab, MODELS_DIR / "lstm_vocab.pkl")

    with open(MODELS_DIR / "lstm_config.json", "w") as f:
        json.dump({"vocab_size": len(vocab), "max_seq_len": LSTM_MAX_SEQ_LEN,
                   "embed_dim": 128}, f, indent=2)

    print(f"    Vocab size: {len(vocab):,}   Sequence shape: {Xtr.shape}")
    return Xtr, Xv, Xte, vocab


# ═══════════════════════════════════════════════════════════
#  7. BERT Tokenization (for DistilBERT)
# ═══════════════════════════════════════════════════════════

def extract_bert_tokens(train, val, test):
    print("\n[INFO] BERT tokenization (DistilBERT)...")
    try:
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    except ImportError:
        print("    [SKIP] transformers not installed. Run: pip install transformers")
        return None, None, None, None

    def tokenize_batch(texts, desc=""):
        enc = tokenizer(
            texts.tolist(),
            max_length      = BERT_MAX_LEN,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "np",
        )
        return enc["input_ids"], enc["attention_mask"]

    print("    Tokenizing train...")
    tr_ids, tr_mask = tokenize_batch(train["clean_text"])
    print("    Tokenizing val...")
    va_ids, va_mask = tokenize_batch(val["clean_text"])
    print("    Tokenizing test...")
    te_ids, te_mask = tokenize_batch(test["clean_text"])

    np.save(FEATURES_DIR / "bert_train_ids.npy",   tr_ids)
    np.save(FEATURES_DIR / "bert_train_mask.npy",  tr_mask)
    np.save(FEATURES_DIR / "bert_val_ids.npy",     va_ids)
    np.save(FEATURES_DIR / "bert_val_mask.npy",    va_mask)
    np.save(FEATURES_DIR / "bert_test_ids.npy",    te_ids)
    np.save(FEATURES_DIR / "bert_test_mask.npy",   te_mask)
    tokenizer.save_pretrained(str(MODELS_DIR / "bert_tokenizer"))

    print(f"    BERT input shape: {tr_ids.shape}")
    return tr_ids, va_ids, te_ids, tokenizer


# ═══════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════

def summary(records):
    print("\n" + "═" * 62)
    print("  🔢 FEATURE EXTRACTION COMPLETE")
    print("═" * 62)
    print(f"  {'Feature':<18} {'Train shape':<22} {'Val shape':<22}")
    print("  " + "─" * 58)
    for name, tr_shape, va_shape in records:
        if tr_shape is None: continue
        print(f"  {name:<18} {str(tr_shape):<22} {str(va_shape):<22}")
    print("═" * 62 + "\n")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 62)
    print("  🚀 PART 3: FEATURE EXTRACTION")
    print("═" * 62)

    train, val, test = load_splits()
    print(f"[INFO] Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    y_train, y_val, y_test, le = encode_labels(train, val, test)

    tfidf_tr, tfidf_v, tfidf_te, _ = extract_tfidf(train, val, test)
    bow_tr,   bow_v,   bow_te      = extract_bow(train, val, test)
    w2v_tr,   w2v_v,   w2v_te     = extract_word2vec(train, val, test)
    ft_tr,    ft_v,    ft_te      = extract_fasttext(train, val, test)
    seq_tr,   seq_v,   seq_te, _  = extract_sequences(train, val, test)
    bert_tr,  bert_v,  bert_te, _ = extract_bert_tokens(train, val, test)

    summary([
        ("TF-IDF",    tfidf_tr.shape, tfidf_v.shape),
        ("BoW",       bow_tr.shape,   bow_v.shape),
        ("Word2Vec",  w2v_tr.shape,   w2v_v.shape),
        ("FastText",  ft_tr.shape,    ft_v.shape),
        ("LSTM seqs", seq_tr.shape,   seq_v.shape),
        ("BERT ids",  bert_tr.shape if bert_tr is not None else None,
                      bert_v.shape  if bert_v  is not None else None),
    ])

    print(f"[✓] All features → {FEATURES_DIR}")
    print(f"[✓] Vectorizers  → {MODELS_DIR}")


if __name__ == "__main__":
    main()