import os
import re
import tarfile
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import joblib


DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DEFAULT_DATA_DIR = "/content/aclImdb"
DEFAULT_TAR_PATH = "/content/aclImdb_v1.tar.gz"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def download_and_extract_imdb(
    data_dir: str = DEFAULT_DATA_DIR,
    tar_path: str = DEFAULT_TAR_PATH,
    url: str = DATA_URL,
) -> None:
    """
    Downloads and extracts aclImdb dataset if not present.
    """
    if os.path.exists(data_dir) and os.listdir(data_dir):
        return

    os.makedirs(os.path.dirname(tar_path), exist_ok=True)

    if not os.path.exists(data_dir):
        print("Downloading IMDB dataset...")
        download_ok = os.system(f'wget -q -O "{tar_path}" "{url}"') == 0
        if not download_ok:
            urllib.request.urlretrieve(url, tar_path)
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(os.path.dirname(data_dir))


def _read_imdb_text_file(fp: str) -> str:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_imdb_binary(
    split_name: str,
    max_per_class: int,
    data_dir: str = DEFAULT_DATA_DIR,
) -> Tuple[List[str], np.ndarray]:
    """
    Loads IMDB reviews for binary sentiment.
    Returns: (texts, labels) where 1=Positive, 0=Negative.
    """
    pos_dir = os.path.join(data_dir, split_name, "pos")
    neg_dir = os.path.join(data_dir, split_name, "neg")

    pos_files = sorted([os.path.join(pos_dir, f) for f in os.listdir(pos_dir)])[:max_per_class]
    neg_files = sorted([os.path.join(neg_dir, f) for f in os.listdir(neg_dir)])[:max_per_class]

    texts: List[str] = []
    labels: List[int] = []

    for fp in tqdm(pos_files, desc=f"Loading {split_name} pos"):
        texts.append(_read_imdb_text_file(fp))
        labels.append(1)

    for fp in tqdm(neg_files, desc=f"Loading {split_name} neg"):
        texts.append(_read_imdb_text_file(fp))
        labels.append(0)

    return texts, np.array(labels, dtype=np.int64)


# ----------------------------
# Text preprocessing (stage 2)
# ----------------------------
STOPWORDS = set(
    """
a an the and or but if while with of at by for from into during including until
against among through despite towards upon about before after above below to in out on off
over under again further then once here there when where why how all any both each few more
most other some such no nor not only own same so than too very s t can will just don should now
""".split()
)

HTML_TAG_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Basic preprocessing:
    - lowercase
    - remove HTML tags
    - keep letters/numbers
    - collapse whitespace
    - remove stopwords + short tokens
    """
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def make_demo_preprocessing_df(
    raw_texts: List[str],
    labels: np.ndarray,
    n: int = 5,
) -> pd.DataFrame:
    rows = []
    for i in range(min(n, len(raw_texts))):
        before = raw_texts[i].replace("\n", " ")
        after = clean_text(raw_texts[i])
        rows.append(
            {
                "Original (first 200 chars)": before[:200] + ("..." if len(before) > 200 else ""),
                "Cleaned (first 200 chars)": after[:200] + ("..." if len(after) > 200 else ""),
                "Label": "Positive" if int(labels[i]) == 1 else "Negative",
            }
        )
    return pd.DataFrame(rows)


@dataclass
class TrainConfig:
    seed: int = 42
    limit_per_class: int = 2500  # per split & per class
    test_size: float = 0.2
    max_features: int = 60000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2


def build_models(seed: int) -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1),
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(),
        "SGD_LogLoss": SGDClassifier(loss="log_loss", max_iter=2500, tol=1e-3, random_state=seed),
    }


def run_training_and_report(
    model_out_path: str = "/content/models/best_sentiment_pipeline.joblib",
    data_dir: str = DEFAULT_DATA_DIR,
    tar_path: str = DEFAULT_TAR_PATH,
    url: str = DATA_URL,
    limit_per_class: int = 2500,
    seed: int = 42,
    max_features: int = 60000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    show_plots: bool = True,
) -> pd.DataFrame:
    """
    End-to-end pipeline: load dataset -> preprocess -> vectorize -> train multiple models
    -> compare -> pick best -> confusion matrix -> save model.

    Returns: comparison_df
    """
    set_seed(seed)

    # Ensure dataset exists
    download_and_extract_imdb(data_dir=data_dir, tar_path=tar_path, url=url)

    # Load (train split)
    print("Loading IMDB training split...")
    raw_texts, y = load_imdb_binary(
        split_name="train",
        max_per_class=limit_per_class,
        data_dir=data_dir,
    )

    print("\nDataset exploration:")
    print("Total samples:", len(raw_texts))
    print("Positive count:", int((y == 1).sum()))
    print("Negative count:", int((y == 0).sum()))

    # Preprocessing demo (for screenshot requirement)
    demo_df = make_demo_preprocessing_df(raw_texts, y, n=5)
    print("\n=== Text Preprocessing Output (before/after) ===")
    try:
        display(demo_df)  # type: ignore[name-defined]
    except Exception:
        print(demo_df)

    # Preprocess all text
    print("Cleaning all text...")
    X_clean = [clean_text(t) for t in tqdm(raw_texts, desc="Preprocessing")]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    # Feature extraction (stage 3)
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    models = build_models(seed=seed)
    trained: Dict[str, Pipeline] = {}
    rows = []

    # Training multiple models (stage 4)
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", pos_label=1, zero_division=0
        )

        rows.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision(Positive)": prec,
                "Recall(Positive)": rec,
                "F1(Positive)": f1,
            }
        )
        trained[name] = pipe

    # Model comparison (stage 5)
    comparison_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False)
    print("\n=== Model Comparison Table ===")
    try:
        display(comparison_df)  # type: ignore[name-defined]
    except Exception:
        print(comparison_df.to_string(index=False))

    if show_plots:
        plt.figure(figsize=(10, 4))
        sns.barplot(data=comparison_df, x="Model", y="Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        plt.title("Model Accuracy Comparison")
        plt.tight_layout()
        plt.show()

    # Best model selection (stage 6)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipe = trained[best_model_name]

    print("\n=== Best Model Selection ===")
    print("Best Model:", best_model_name)
    print("Best Accuracy:", float(comparison_df.iloc[0]["Accuracy"]))

    # Confusion matrix (screenshot requirement)
    best_preds = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)

    if show_plots:
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative(0)", "Positive(1)"],
            yticklabels=["Negative(0)", "Positive(1)"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {best_model_name}")
        plt.tight_layout()
        plt.show()

    # Retrain best on all cleaned data (so GUI uses max data)
    print("\nRetraining best model on full dataset...")
    best_pipe.fit(X_clean, y)

    # Save model file (submission requirement)
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(best_pipe, model_out_path)
    print("\nSaved best model to:", model_out_path)

    return comparison_df


if __name__ == "__main__":
    # Simple CLI usage (optional)
    run_training_and_report()

