"""
============================================================
  EMOTION ANALYSIS ON SOCIAL MEDIA - Part 6/6
  📊 Output Visualization
============================================================
  12 publication-ready plots:
   01. Model accuracy comparison (grouped bar)
   02. Model F1 comparison (weighted + macro)
   03. Radar chart — top 5 models, 6 metrics
   04. Confusion matrix heatmap — best model (28×28)
   05. Per-emotion F1 heatmap — all models
   06. Emotion wheel / polar bar chart
   07. ROC curves — all classifiers
   08. Overfitting gap analysis
   09. Sentiment distribution (pie, by group)
   10. Feature method comparison
   11. t-SNE embedding visualization (TF-IDF → 2D)
   12. Word clouds — per emotion group
  Output: data/results/plots/
============================================================
"""

import json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import seaborn as sns
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
FEAT_DIR      = BASE_DIR / "data" / "features"
RESULTS_DIR   = BASE_DIR / "data" / "results"
TRAINED_DIR   = BASE_DIR / "data" / "trained_models"
MODELS_DIR    = BASE_DIR / "data" / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR     = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
})

EMOTION_PALETTE = {
    "positive": "#4CAF50", "negative": "#F44336",
    "ambiguous": "#FF9800", "neutral":  "#9E9E9E",
}
MODEL_COLORS = ["#2196F3","#4CAF50","#F44336","#FF9800","#9C27B0","#00BCD4","#E91E63","#8BC34A"]

EMOTION_GROUPS = {
    "positive":  ["admiration","amusement","approval","caring","curiosity",
                  "desire","excitement","gratitude","joy","love","optimism","pride","relief"],
    "negative":  ["anger","annoyance","disappointment","disapproval",
                  "disgust","embarrassment","fear","grief","nervousness","remorse","sadness"],
    "ambiguous": ["confusion","realization","surprise"],
    "neutral":   ["neutral"],
}


def emotion_color(emotion):
    for group, ems in EMOTION_GROUPS.items():
        if emotion in ems:
            return EMOTION_PALETTE[group]
    return "#9E9E9E"


def short(name):
    return (name.replace("Logistic Regression","LR")
                .replace("Multinomial NB","MNB")
                .replace("Bernoulli NB","BNB")
                .replace("Random Forest","RF")
                .replace("Linear SVM","LinearSVM")
                .replace("Bidirectional LSTM","BiLSTM")
                .replace("DistilBERT (fine-tuned)","DistilBERT")
                .replace("XGBoost","XGB")
                .replace(" (TF-IDF)","").replace(" (BoW)","")
                .replace(" (Word2Vec)","").replace(" (FastText)",""))


def save(fig, name):
    fig.savefig(PLOTS_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [✓] {name}")


# ── Loaders ───────────────────────────────────────────────
def load_report():
    return pd.read_csv(RESULTS_DIR / "comparative_report.csv", index_col=0)

def load_eval():
    with open(RESULTS_DIR / "full_evaluation.json") as f:
        return json.load(f)

def load_labels():
    return np.load(FEAT_DIR / "y_test.npy")

def load_id2emotion():
    try:
        with open(MODELS_DIR / "id2emotion.json") as f:
            return {int(k): v for k, v in json.load(f).items()}
    except:
        return {}


# ═══════════════════════════════════════════════════════════
#  PLOT 01 — Accuracy comparison
# ═══════════════════════════════════════════════════════════
def plot_01_accuracy(df):
    names = [short(n) for n in df["Model"]]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (col, label) in enumerate([("Train Acc","Train"),("Val Acc","Val"),("Test Acc","Test")]):
        bars = ax.bar(x + (i-1)*w, df[col], w, label=label, color=MODEL_COLORS[i], alpha=0.85)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("Model Accuracy: Train vs Val vs Test", fontsize=14, fontweight="bold")
    ax.legend(); fig.tight_layout()
    save(fig, "01_accuracy_comparison.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 02 — F1 comparison
# ═══════════════════════════════════════════════════════════
def plot_02_f1(df):
    names = [short(n) for n in df["Model"]]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (col, label, color) in enumerate([
        ("Test F1 (Weighted)","F1 Weighted","#FF9800"),
        ("Test F1 (Macro)","F1 Macro","#9C27B0"),
    ]):
        bars = ax.bar(x+(i-0.5)*w/2, df[col], w*0.9, label=label, color=color, alpha=0.85)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.1)
    ax.set_title("F1 Score Comparison (Weighted vs Macro)", fontsize=14, fontweight="bold")
    ax.legend(); fig.tight_layout()
    save(fig, "02_f1_comparison.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 03 — Radar chart (top 5 models)
# ═══════════════════════════════════════════════════════════
def plot_03_radar(df):
    metrics = ["Test Acc","Test F1 (Weighted)","Test F1 (Macro)",
               "Test Precision","Test Recall","Test MCC"]
    labels  = ["Accuracy","F1 Weighted","F1 Macro","Precision","Recall","MCC"]
    top5 = df.head(min(5, len(df)))
    N = len(metrics); angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [float(row.get(m,0)) for m in metrics]; vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, color=MODEL_COLORS[i], label=short(row["Model"]))
        ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[i])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0,1)
    ax.set_title("Top-5 Models — Metric Radar", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=9)
    save(fig, "03_radar_chart.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 04 — Confusion matrix heatmap (best model)
# ═══════════════════════════════════════════════════════════
def plot_04_confusion_matrix(full_eval, id2emotion):
    best = full_eval[0]
    if "confusion_matrix" not in best: return
    cm   = np.array(best["confusion_matrix"])
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n    = cm.shape[0]
    labels = [id2emotion.get(i, str(i)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(14, n*0.5), max(12, n*0.45)))
    sns.heatmap(norm, ax=ax, cmap="YlOrRd", xticklabels=labels, yticklabels=labels,
                annot=(n <= 15), fmt=".2f", linewidths=0.3, cbar_kws={"shrink":0.7})
    ax.set_xlabel("Predicted Emotion", fontsize=11)
    ax.set_ylabel("True Emotion", fontsize=11)
    ax.set_title(f"Confusion Matrix (Normalized) — {short(best['name'])}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    save(fig, "04_confusion_matrix_best.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 05 — Per-emotion F1 heatmap
# ═══════════════════════════════════════════════════════════
def plot_05_per_emotion_heatmap():
    pe_path = RESULTS_DIR / "per_emotion_f1.csv"
    if not pe_path.exists(): return
    pe = pd.read_csv(pe_path)
    pivot = pe.pivot_table(index="emotion", columns="model", values="f1")
    pivot.columns = [short(c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns)*1.5), max(10, len(pivot)*0.4)))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f",
                linewidths=0.3, vmin=0, vmax=1,
                cbar_kws={"label":"F1 Score", "shrink":0.7})
    ax.set_title("Per-Emotion F1 — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Emotion")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    save(fig, "05_per_emotion_f1_heatmap.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 06 — Emotion Wheel (polar bar chart)
# ═══════════════════════════════════════════════════════════
def plot_06_emotion_wheel():
    test_path = PROCESSED_DIR / "test.csv"
    if not test_path.exists(): return
    test = pd.read_csv(test_path)
    counts = test["emotion"].value_counts()
    emotions = counts.index.tolist()
    values   = counts.values.tolist()
    colors   = [emotion_color(e) for e in emotions]

    N      = len(emotions)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    vals   = np.array(values, dtype=float)
    vals   = vals / vals.max()    # normalize

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    bars = ax.bar(angles, vals, width=2*np.pi/N*0.85, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(angles)
    ax.set_xticklabels(emotions, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Emotion Distribution Wheel\n(test set)", fontsize=14, fontweight="bold", y=1.06)

    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in EMOTION_PALETTE.items()]
    ax.legend(handles=legend_patches, loc="lower left", bbox_to_anchor=(-0.1, -0.05))
    fig.tight_layout()
    save(fig, "06_emotion_wheel.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 07 — ROC Curves
# ═══════════════════════════════════════════════════════════
def plot_07_roc_curves(trained_models, y_test, classes):
    tfidf_test = load_npz(FEAT_DIR / "tfidf_test.npz")
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (key, bundle) in enumerate(trained_models.items()):
        model  = bundle["model"]
        scaler = bundle.get("scaler")
        fkey   = bundle["feature_key"].replace("_train","")
        if fkey not in ["tfidf"]: continue

        X = tfidf_test
        if scaler: X = scaler.transform(X)
        try:
            if hasattr(model,"predict_proba"):
                proba = model.predict_proba(X)
            elif hasattr(model,"decision_function"):
                proba = model.decision_function(X)
            else: continue

            y_bin = label_binarize(y_test, classes=classes)
            fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
            roc_auc_val = auc(fpr, tpr)
            name = key.replace("_"," ").replace("-","/")
            ax.plot(fpr, tpr, color=MODEL_COLORS[i%len(MODEL_COLORS)], lw=1.5,
                    label=f"{short(name)} (AUC={roc_auc_val:.3f})")
        except: continue

    ax.plot([0,1],[0,1],"k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Classical Models (micro-average)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    fig.tight_layout()
    save(fig, "07_roc_curves.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 08 — Overfitting analysis
# ═══════════════════════════════════════════════════════════
def plot_08_overfit(df):
    names = [short(n) for n in df["Model"]]
    gaps  = df["Overfit Gap"].astype(float)
    colors = ["#F44336" if g > 0.05 else "#4CAF50" for g in gaps]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, gaps, color=colors, alpha=0.85, edgecolor="white")
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{g:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.05, color="orange", linestyle="--", alpha=0.7, label="Threshold (0.05)")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Train Acc − Test Acc"); ax.set_title("Overfitting Analysis", fontsize=14, fontweight="bold")
    r = mpatches.Patch(color="#F44336", alpha=0.85, label="Overfitting")
    g = mpatches.Patch(color="#4CAF50", alpha=0.85, label="Generalising")
    ax.legend(handles=[r, g, mpatches.Patch(color="orange", alpha=0.7, label="Threshold")])
    fig.tight_layout()
    save(fig, "08_overfit_analysis.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 09 — Emotion group distribution (pie)
# ═══════════════════════════════════════════════════════════
def plot_09_distribution():
    splits = {}
    for name in ["train","val","test"]:
        p = PROCESSED_DIR / f"{name}.csv"
        if p.exists():
            splits[name] = pd.read_csv(p)
    if not splits: return

    fig, axes = plt.subplots(1, len(splits), figsize=(5*len(splits), 5))
    if len(splits)==1: axes=[axes]
    for ax, (split, df) in zip(axes, splits.items()):
        df["group"] = df["emotion"].apply(
            lambda e: next((g for g, ems in EMOTION_GROUPS.items() if e in ems), "neutral"))
        counts  = df["group"].value_counts()
        colors  = [EMOTION_PALETTE.get(g, "#999") for g in counts.index]
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors,
               wedgeprops=dict(edgecolor="white", linewidth=2), startangle=90)
        ax.set_title(f"{split.capitalize()} Set\n(n={len(df):,})", fontsize=12, fontweight="bold")
    fig.suptitle("Emotion Group Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "09_emotion_distribution.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 10 — Feature method comparison
# ═══════════════════════════════════════════════════════════
def plot_10_feature_comparison(df):
    df2 = df.copy()
    df2["feat"] = df2["Feature"].str.replace("_train","")
    grp = df2.groupby("feat")["Test F1 (Weighted)"].agg(["mean","max","min"])

    fig, ax = plt.subplots(figsize=(9,5))
    x = np.arange(len(grp))
    for i, (col, label, c) in enumerate([("max","Max","#2196F3"),("mean","Mean","#4CAF50"),("min","Min","#F44336")]):
        ax.bar(x+i*0.25-0.25, grp[col], 0.22, label=label, color=c, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(grp.index, fontsize=11)
    ax.set_ylabel("Test F1 (Weighted)"); ax.set_ylim(0,1.1)
    ax.set_title("Feature Extraction Method Comparison", fontsize=14, fontweight="bold")
    ax.legend(); fig.tight_layout()
    save(fig, "10_feature_comparison.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 11 — t-SNE embedding visualisation
# ═══════════════════════════════════════════════════════════
def plot_11_tsne(y_test, id2emotion):
    tfidf_test = load_npz(FEAT_DIR / "tfidf_test.npz")
    n_samples  = min(3000, tfidf_test.shape[0])
    idx        = np.random.choice(tfidf_test.shape[0], n_samples, replace=False)
    X_sub      = tfidf_test[idx]
    y_sub      = y_test[idx]

    print("  [INFO] Running t-SNE (may take ~1 min)...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    X50 = svd.fit_transform(X_sub)
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=800)
    X2d  = tsne.fit_transform(X50)

    emotions_present = sorted(set(y_sub))
    cmap = get_cmap("tab20", len(emotions_present))
    color_map = {e: cmap(i) for i, e in enumerate(emotions_present)}

    fig, ax = plt.subplots(figsize=(12, 10))
    for eid in emotions_present:
        mask = y_sub == eid
        ename = id2emotion.get(eid, str(eid))
        ax.scatter(X2d[mask,0], X2d[mask,1], c=[color_map[eid]],
                   s=12, alpha=0.6, label=ename)
    ax.set_title(f"t-SNE of TF-IDF features ({n_samples:,} samples)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="center left", bbox_to_anchor=(1,0.5), fontsize=8, ncol=2, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, "11_tsne_embeddings.png")


# ═══════════════════════════════════════════════════════════
#  PLOT 12 — Word clouds per emotion group
# ═══════════════════════════════════════════════════════════
def plot_12_wordclouds():
    if not HAS_WC:
        print("  [SKIP] wordcloud not installed: pip install wordcloud")
        return
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    train_path = PROCESSED_DIR / "train.csv"
    if not (tfidf_path.exists() and train_path.exists()): return

    with open(tfidf_path,"rb") as f: vec = pickle.load(f)
    train = pd.read_csv(train_path)
    train["group"] = train["emotion"].apply(
        lambda e: next((g for g, ems in EMOTION_GROUPS.items() if e in ems), "neutral"))
    vocab = vec.vocabulary_; idf = vec.idf_

    groups = list(EMOTION_GROUPS.keys())
    fig, axes = plt.subplots(1, len(groups), figsize=(7*len(groups), 5))
    cmaps = {"positive":"Greens","negative":"Reds","ambiguous":"Oranges","neutral":"Blues"}

    for ax, group in zip(axes, groups):
        text = " ".join(train[train["group"]==group]["clean_text"].fillna(""))
        freqs = {}
        for word, cnt in zip(*np.unique(text.split(), return_counts=True)):
            if word in vocab:
                freqs[word] = cnt * idf[vocab[word]]
        if not freqs: ax.axis("off"); continue
        wc = WordCloud(width=600, height=400, background_color="white",
                       colormap=cmaps.get(group,"viridis"), max_words=80
                       ).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"{group.capitalize()}\nTop keywords", fontsize=12, fontweight="bold")

    fig.suptitle("Word Clouds by Emotion Group (TF-IDF weighted)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "12_wordclouds.png")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 55)
    print("  🚀 PART 6: OUTPUT VISUALIZATION")
    print(f"  Generating 12 plots → {PLOTS_DIR}")
    print("═" * 55 + "\n")

    df        = load_report()
    full_eval = load_eval()
    y_test    = load_labels()
    id2emotion= load_id2emotion()
    classes   = np.unique(y_test)

    trained = {}
    for p in sorted(TRAINED_DIR.glob("*.pkl")):
        with open(p,"rb") as f:
            trained[p.stem] = pickle.load(f)

    plot_01_accuracy(df)
    plot_02_f1(df)
    plot_03_radar(df)
    plot_04_confusion_matrix(full_eval, id2emotion)
    plot_05_per_emotion_heatmap()
    plot_06_emotion_wheel()
    plot_07_roc_curves(trained, y_test, classes)
    plot_08_overfit(df)
    plot_09_distribution()
    plot_10_feature_comparison(df)
    plot_11_tsne(y_test, id2emotion)
    plot_12_wordclouds()

    print("\n" + "═" * 55)
    print("  ✅ ALL 12 VISUALIZATIONS COMPLETE")
    print("═" * 55)
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"    📊 {p.name}")


if __name__ == "__main__":
    main()
