import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TrainConfig:
    dataset_name: str = "yahuqiao/fake-real-news"
    text_col: str = "text"
    label_col: str = "label"
    test_size: float = 0.2
    random_state: int = 42

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize_embeddings: bool = True

    cache_dir: str = "artifacts"
    embeddings_cache_prefix: str = "embeddings"

    models_dir: str = "models"
    best_model_filename: str = "best_model.joblib"
    best_model_meta_filename: str = "best_model.meta.json"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cache_paths(cfg: TrainConfig):
    base = f"{cfg.embeddings_cache_prefix}-{cfg.dataset_name.replace('/', '__')}-{cfg.embedding_model.replace('/', '__')}"
    x_path = os.path.join(cfg.cache_dir, f"{base}.X.npy")
    y_path = os.path.join(cfg.cache_dir, f"{base}.y.npy")
    return x_path, y_path


def load_text_label(cfg: TrainConfig):
    ds = load_dataset(cfg.dataset_name)
    if "train" not in ds:
        raise KeyError(f"Dataset has no 'train' split. Splits: {list(ds.keys())}")
    train = ds["train"]

    if cfg.text_col not in train.column_names:
        raise KeyError(f"Missing text column {cfg.text_col!r}. Columns: {train.column_names}")
    if cfg.label_col not in train.column_names:
        raise KeyError(f"Missing label column {cfg.label_col!r}. Columns: {train.column_names}")

    texts = train[cfg.text_col]
    labels = train[cfg.label_col]

    # HF boolean column can come through as python bool already; enforce 0/1 ints for sklearn.
    y = np.asarray([1 if bool(v) else 0 for v in labels], dtype=np.int64)
    return texts, y


def embed_texts(cfg: TrainConfig, texts, *, force_recompute: bool = False):
    _ensure_dir(cfg.cache_dir)
    x_cache_path, y_cache_path = _cache_paths(cfg)

    if (not force_recompute) and os.path.exists(x_cache_path) and os.path.exists(y_cache_path):
        X = np.load(x_cache_path)
        y = np.load(y_cache_path)
        return X, y, True

    model = SentenceTransformer(cfg.embedding_model)
    X = model.encode(
        texts,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.normalize_embeddings,
    ).astype(np.float32)

    np.save(x_cache_path, X)
    # y is returned separately by caller; this function caches only X
    return X, None, False


def train_and_eval(cfg: TrainConfig, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=cfg.random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred))
        results[name] = {"model": clf, "accuracy": acc, "f1": f1}

    return results


def pick_best(results: dict):
    # Primary: F1, tie-breaker: Accuracy
    best_name = max(results.keys(), key=lambda k: (results[k]["f1"], results[k]["accuracy"]))
    return best_name, results[best_name]


def save_best(cfg: TrainConfig, best_name: str, best_entry: dict):
    _ensure_dir(cfg.models_dir)
    model_path = os.path.join(cfg.models_dir, cfg.best_model_filename)
    meta_path = os.path.join(cfg.models_dir, cfg.best_model_meta_filename)

    payload = {
        "model_name": best_name,
        "metrics": {"accuracy": best_entry["accuracy"], "f1": best_entry["f1"]},
        "config": asdict(cfg),
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }

    # Save sklearn model; note: sentence-transformers is not saved here (we only use it to embed).
    joblib.dump(best_entry["model"], model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved best model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")


def main():
    cfg = TrainConfig()
    _ensure_dir(cfg.cache_dir)

    print(f"Loading dataset: {cfg.dataset_name}")
    texts, y = load_text_label(cfg)
    print(f"Loaded rows: {len(texts)}")

    x_cache_path, y_cache_path = _cache_paths(cfg)
    if os.path.exists(y_cache_path):
        y_cached = np.load(y_cache_path)
        # quick sanity: ensure same length
        if len(y_cached) == len(y):
            y = y_cached

    print(f"Embedding with: {cfg.embedding_model}")
    X, _, used_cache = embed_texts(cfg, texts)

    # cache y alongside X (kept separate so we can load X/y fast next run)
    if not os.path.exists(y_cache_path):
        np.save(y_cache_path, y)

    print(f"Embeddings shape: {X.shape} (cache_used={used_cache})")

    results = train_and_eval(cfg, X, y)
    for name, r in results.items():
        print(f"{name}: Accuracy={r['accuracy']:.4f}  F1={r['f1']:.4f}")

    best_name, best_entry = pick_best(results)
    print(f"\nBest model: {best_name} (F1={best_entry['f1']:.4f}, Acc={best_entry['accuracy']:.4f})")
    save_best(cfg, best_name, best_entry)


if __name__ == "__main__":
    main()
