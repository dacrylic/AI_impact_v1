from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


CLASS_NAMES = ["E0", "E3", "E2", "E1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a zoo of classical NLP baselines.")
    parser.add_argument("--input", required=True, help="Path to the source .xlsx workbook.")
    parser.add_argument("--output-dir", default="results/classical_zoo", help="Directory to write artifacts.")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-teacher-score",
        action="store_true",
        help="Include ai_impact_score as a teacher-aided diagnostic. Disabled by default because it maps directly to the label.",
    )
    return parser.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _text_join(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    text = frame[columns[0]].astype(str)
    for column in columns[1:]:
        text = text + " [SEP] " + frame[column].astype(str)
    return text


def _pair_text(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    return (
        frame[left].fillna("").astype(str).str.strip()
        + " [SEP] "
        + frame[right].fillna("").astype(str).str.strip()
    ).str.strip()


def _best_bias_from_scores(y_true: np.ndarray, score_matrix: np.ndarray) -> np.ndarray:
    best_bias = np.zeros(score_matrix.shape[1], dtype=float)
    best_score = -1.0
    grid = np.linspace(-1.5, 1.5, 13)
    for b1 in grid:
        for b2 in grid:
            for b3 in grid:
                bias = np.array([0.0, b1, b2, b3], dtype=float)
                pred = np.argmax(score_matrix + bias, axis=1)
                score = f1_score(y_true, pred, average="macro")
                if score > best_score:
                    best_score = score
                    best_bias = bias
    return best_bias


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class OrderedLogitModel:
    coef_: np.ndarray
    cuts_: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        eta = x @ self.coef_
        logits = self.cuts_[None, :] - eta[:, None]
        cum = expit(logits)
        probs = np.empty((x.shape[0], len(self.cuts_) + 1), dtype=float)
        probs[:, 0] = cum[:, 0]
        for idx in range(1, len(self.cuts_)):
            probs[:, idx] = cum[:, idx] - cum[:, idx - 1]
        probs[:, -1] = 1.0 - cum[:, -1]
        probs = np.clip(probs, 1e-12, None)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


@dataclass
class OrderedLogitPipeline:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    model: OrderedLogitModel

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        x = self.svd.transform(self.vectorizer.transform(texts))
        return self.model.predict_proba(x)

    def predict(self, texts: pd.Series) -> np.ndarray:
        return self.model.predict(self.svd.transform(self.vectorizer.transform(texts)))


def _fit_ordered_logit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    l2: float = 1.0,
    max_iter: int = 250,
) -> OrderedLogitModel:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    n_samples, n_features = x_train.shape
    n_classes = int(y_train.max()) + 1
    if n_classes < 3:
        raise ValueError("ordered logit requires at least 3 classes")

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    y_init = np.quantile(y_train.astype(float), np.linspace(0.2, 0.8, n_classes - 1))
    initial_cuts = np.linspace(-1.0, 1.0, n_classes - 1)
    initial_deltas = np.log(np.maximum(np.diff(np.r_[[-5.0], initial_cuts]), 1e-2))
    initial = np.r_[np.zeros(n_features, dtype=float), initial_cuts[0], np.zeros(n_classes - 2, dtype=float)]

    def unpack(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coef = theta[:n_features]
        raw = theta[n_features:]
        cuts = np.empty(n_classes - 1, dtype=float)
        cuts[0] = raw[0]
        for i in range(1, len(cuts)):
            cuts[i] = cuts[i - 1] + np.exp(raw[i])
        return coef, cuts

    def objective(theta: np.ndarray) -> float:
        coef, cuts = unpack(theta)
        eta = x_train @ coef
        logits = cuts[None, :] - eta[:, None]
        cum = expit(logits)
        probs = np.empty((n_samples, n_classes), dtype=float)
        probs[:, 0] = cum[:, 0]
        for idx in range(1, n_classes - 1):
            probs[:, idx] = cum[:, idx] - cum[:, idx - 1]
        probs[:, -1] = 1.0 - cum[:, -1]
        probs = np.clip(probs, 1e-12, None)
        ll = np.log(probs[np.arange(n_samples), y_train])
        loss = -float(np.sum(sample_weight * ll))
        loss += 0.5 * l2 * float(np.dot(coef, coef))
        return loss

    result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": max_iter})
    coef, cuts = unpack(result.x)
    return OrderedLogitModel(coef_=coef, cuts_=cuts)


def _search_weighted_blend(
    y_true: np.ndarray,
    val_mats: Sequence[np.ndarray],
    test_mats: Sequence[np.ndarray],
    *,
    n_trials: int = 2000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    best_val = None
    best_test = None
    best_weights = None
    best_score = -1.0
    num_models = len(val_mats)
    for _ in range(n_trials):
        weights = rng.dirichlet(np.ones(num_models))
        blended_val = np.zeros_like(val_mats[0], dtype=float)
        blended_test = np.zeros_like(test_mats[0], dtype=float)
        for weight, val_mat, test_mat in zip(weights, val_mats, test_mats):
            blended_val += weight * val_mat
            blended_test += weight * test_mat
        score = f1_score(y_true, np.argmax(blended_val, axis=1), average="macro")
        if score > best_score:
            best_score = score
            best_val = blended_val
            best_test = blended_test
            best_weights = weights
    assert best_val is not None and best_test is not None and best_weights is not None
    return best_val, best_test, best_weights


def _train_svm(
    train_text: pd.Series,
    train_y: np.ndarray,
    *,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 300_000,
    class_weight: Dict[int, float] | str = "balanced",
    c_value: float = 1.0,
) -> make_pipeline:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_features=max_features),
        LinearSVC(class_weight=class_weight, C=c_value),
    )
    model.fit(train_text, train_y)
    return model


def _train_lr(
    train_text: pd.Series,
    train_y: np.ndarray,
    *,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 300_000,
    class_weight: Dict[int, float] | str = "balanced",
) -> make_pipeline:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_features=max_features),
        LogisticRegression(max_iter=5000, class_weight=class_weight, n_jobs=1),
    )
    model.fit(train_text, train_y)
    return model


def _train_score_aided_lr(
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    *,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 300_000,
    class_weight: Dict[int, float] | str = "balanced",
) -> make_pipeline:
    features = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_features=max_features),
                "combo2",
            ),
            (
                "score",
                StandardScaler(with_mean=False),
                ["ai_impact_score"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = make_pipeline(
        features,
        LogisticRegression(max_iter=5000, class_weight=class_weight, n_jobs=1),
    )
    model.fit(train_frame, train_y)
    return model


def _ordinal_threshold_probs(
    train_text: pd.Series,
    train_y: np.ndarray,
    eval_text: pd.Series,
    *,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 300_000,
    class_weight: Dict[int, float] | str = "balanced",
) -> np.ndarray:
    thresholds: List[np.ndarray] = []
    for threshold in range(len(CLASS_NAMES) - 1):
        binary_y = (train_y > threshold).astype(int)
        model = make_pipeline(
            TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_features=max_features),
            LogisticRegression(max_iter=5000, class_weight=class_weight, n_jobs=1),
        )
        model.fit(train_text, binary_y)
        thresholds.append(model.predict_proba(eval_text)[:, 1])

    probs_gt = np.column_stack(thresholds)
    # Enforce monotonicity of cumulative probabilities.
    for idx in range(1, probs_gt.shape[1]):
        probs_gt[:, idx] = np.minimum(probs_gt[:, idx], probs_gt[:, idx - 1])

    p0 = 1.0 - probs_gt[:, 0]
    p1 = probs_gt[:, 0] - probs_gt[:, 1]
    p2 = probs_gt[:, 1] - probs_gt[:, 2]
    p3 = probs_gt[:, 2]
    probs = np.column_stack([p0, p1, p2, p3])
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums


def _train_char_word_svm(
    train_text: pd.Series,
    train_y: np.ndarray,
    *,
    class_weight: Dict[int, float] | str = "balanced",
    c_value: float = 1.0,
) -> make_pipeline:
    model = make_pipeline(
        FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=250_000,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=250_000,
                    ),
                ),
            ]
        ),
        LinearSVC(class_weight=class_weight, C=c_value),
    )
    model.fit(train_text, train_y)
    return model


def _column_text_vectorizer(
    *,
    analyzer: str = "word",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int = 250_000,
) -> make_pipeline:
    return make_pipeline(
        FunctionTransformer(lambda x: np.asarray(x).ravel().astype(str), validate=False),
        TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
        ),
    )


def _train_multiview_svm(
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    *,
    class_weight: Dict[int, float] | str = "balanced",
    c_value: float = 0.75,
) -> make_pipeline:
    features = ColumnTransformer(
        [
            ("jobrole_title", _column_text_vectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000), "jobrole_title"),
            ("sector_title", _column_text_vectorizer(ngram_range=(1, 2), min_df=2, max_features=40_000), "sector_title"),
            ("ssoc_title", _column_text_vectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000), "ssoc_title"),
            ("keytask_word", _column_text_vectorizer(ngram_range=(1, 2), min_df=2, max_features=220_000), "keytask_content"),
            ("keytask_char", _column_text_vectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=180_000), "keytask_content"),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = make_pipeline(features, LinearSVC(class_weight=class_weight, C=c_value))
    model.fit(train_frame, train_y)
    return model


def _train_ordered_logit(
    train_text: pd.Series,
    train_y: np.ndarray,
    *,
    class_weight: Dict[int, float] | str = "balanced",
    n_components: int = 180,
    l2: float = 1.0,
) -> OrderedLogitPipeline:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300_000)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_x = svd.fit_transform(tfidf.fit_transform(train_text))
    if class_weight == "balanced":
        counts = np.bincount(train_y, minlength=len(CLASS_NAMES)).astype(float)
        weights = (counts.sum() / np.maximum(counts, 1.0))[train_y]
    elif isinstance(class_weight, dict):
        weights = np.array([float(class_weight.get(int(y), 1.0)) for y in train_y], dtype=float)
    else:
        weights = np.ones_like(train_y, dtype=float)
    model = _fit_ordered_logit(train_x, train_y, sample_weight=weights, l2=l2)
    return OrderedLogitPipeline(vectorizer=tfidf, svd=svd, model=model)


def _train_nb(train_text: pd.Series, train_y: np.ndarray) -> make_pipeline:
    model = make_pipeline(
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=250_000,
            use_idf=False,
            norm=None,
        ),
        MultinomialNB(alpha=0.1),
    )
    model.fit(train_text, train_y)
    return model


def _hierarchical_predict(
    train_text: pd.Series,
    train_y: np.ndarray,
    eval_text: pd.Series,
) -> np.ndarray:
    stage1 = _train_svm(
        train_text,
        (train_y != 0).astype(int),
        class_weight={0: 1.0, 1: 3.0},
        c_value=1.0,
    )
    stage2_mask = train_y != 0
    stage2 = _train_svm(
        train_text[stage2_mask],
        train_y[stage2_mask],
        class_weight={1: 4.0, 2: 2.0, 3: 8.0},
        c_value=1.0,
    )
    nonzero = stage1.predict(eval_text).astype(int)
    pred = np.zeros(len(eval_text), dtype=int)
    mask = nonzero == 1
    if mask.any():
        pred[mask] = stage2.predict(eval_text[mask])
    return pred


def _lookup_hybrid(
    train_text: pd.Series,
    train_y: np.ndarray,
    eval_text: pd.Series,
    fallback_pred: np.ndarray,
) -> np.ndarray:
    lookup = (
        pd.DataFrame({"text": train_text.astype(str), "y": train_y})
        .groupby("text")["y"]
        .agg(lambda s: int(s.value_counts().idxmax()))
        .to_dict()
    )
    lookup_pred = eval_text.astype(str).map(lookup).fillna(-1).astype(int).to_numpy()
    return np.where(lookup_pred != -1, lookup_pred, fallback_pred)


def _rule_classifier(
    train_text: pd.Series,
    train_y: np.ndarray,
    eval_text: pd.Series,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000)
    matrix = vectorizer.fit_transform(train_text)
    clf = LinearSVC(class_weight={0: 1.0, 1: 2.0, 2: 2.0, 3: 8.0}, C=1.0)
    clf.fit(matrix, train_y)
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_terms: Dict[int, List[str]] = {}
    for idx, cls in enumerate(CLASS_NAMES):
        coefs = clf.coef_[idx]
        top = coefs.argsort()[-25:][::-1]
        top_terms[idx] = feature_names[top].tolist()

    def score_row(text: str) -> np.ndarray:
        lower = f" {text.lower()} "
        scores = np.zeros(4, dtype=float)
        for cls, terms in top_terms.items():
            for term in terms:
                if f" {term} " in lower:
                    scores[cls] += 1.0
        if scores.max() == 0:
            scores[0] = 1.0
        return scores

    scores = np.vstack([score_row(t) for t in eval_text.astype(str)])
    return scores.argmax(axis=1)


def _tree_classifier(
    train_text: pd.Series,
    train_y: np.ndarray,
    eval_text: pd.Series,
) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=250_000)
    svd = TruncatedSVD(n_components=300, random_state=42)
    train_x = svd.fit_transform(vec.fit_transform(train_text))
    eval_x = svd.transform(vec.transform(eval_text))
    clf = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.08, max_iter=250, random_state=42)
    sample_weight = np.array([1.0 / np.bincount(train_y)[y] for y in train_y], dtype=float)
    clf.fit(train_x, train_y, sample_weight=sample_weight)
    return clf.predict(eval_x)


def _knn_classifier(
    train_text: pd.Series,
    train_y: np.ndarray,
    eval_text: pd.Series,
) -> np.ndarray:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=250_000),
        Normalizer(copy=False),
        KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance", algorithm="brute", n_jobs=-1),
    )
    model.fit(train_text, train_y)
    return model.predict(eval_text)


def _association_rule_classifier(
    train_text: pd.Series,
    train_y: np.ndarray,
    eval_text: pd.Series,
) -> np.ndarray:
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=6, binary=True, max_features=120_000)
    x = vectorizer.fit_transform(train_text)
    feature_names = np.array(vectorizer.get_feature_names_out())
    overall_support = np.asarray(x.mean(axis=0)).ravel()
    class_priors = np.bincount(train_y, minlength=len(CLASS_NAMES)).astype(float)
    class_priors /= class_priors.sum()

    class_rules: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for cls in range(len(CLASS_NAMES)):
        mask = train_y == cls
        if not mask.any():
            continue
        class_support = np.asarray(x[mask].mean(axis=0)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.divide(class_support, overall_support, out=np.zeros_like(class_support), where=overall_support > 0)
        confidence = class_support
        interesting = (class_support >= 0.01) & (lift >= 1.15)
        if not np.any(interesting):
            continue
        scores = np.log1p(np.maximum(lift - 1.0, 0.0)) * confidence
        top_idx = np.argsort(scores[interesting])[-40:]
        selected = np.where(interesting)[0][top_idx]
        class_rules[cls] = [(feature_names[i], float(scores[i])) for i in selected if scores[i] > 0]

    default_class = int(np.argmax(class_priors))

    def score_row(text: str) -> np.ndarray:
        lower = f" {text.lower()} "
        scores = np.zeros(len(CLASS_NAMES), dtype=float)
        for cls, rules in class_rules.items():
            for phrase, weight in rules:
                if f" {phrase} " in lower:
                    scores[cls] += weight
        if scores.max() == 0:
            scores[default_class] = 1.0
        return scores

    scored = np.vstack([score_row(t) for t in eval_text.astype(str)])
    return scored.argmax(axis=1)


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args.input)
    split = stratified_group_train_val_test_split(
        bundle.frame,
        group_column=GROUP_COLUMN,
        label_column="label_rank",
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    leak_report = check_split_leakage(
        bundle.frame,
        split_assignments={
            "train": split.train_idx,
            "val": split.val_idx,
            "test": split.test_idx,
        },
        group_column=GROUP_COLUMN,
        text_column="target_text",
        soft_id_columns=["task_id", "jobrole_task_id", "ssoc_code"],
    )
    if not leak_report.passed:
        raise RuntimeError("Leakage check failed:\n" + "\n".join(leak_report.messages))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = bundle.frame.copy()
    frame["role_task"] = _pair_text(frame, "jobrole_title", "keytask_content")
    frame["role_sector_task"] = _text_join(frame, ["jobrole_title", "sector_title", "keytask_content"])
    frame["combo2"] = _text_join(frame, ["jobrole_title", "sector_title", "keytask_content"])
    frame["combo3"] = _text_join(frame, ["jobrole_title", "ssoc_title", "sector_title", "keytask_content"])

    train = frame.loc[split.train_idx]
    val = frame.loc[split.val_idx]
    test = frame.loc[split.test_idx]
    y_train = train["label_rank"].to_numpy()
    y_val = val["label_rank"].to_numpy()
    y_test = test["label_rank"].to_numpy()

    runs: List[Dict[str, object]] = []

    word_svm = _train_svm(train["target_text"], y_train, class_weight={0: 1, 1: 2, 2: 2, 3: 8}, c_value=1.0)
    combo2_svm = _train_svm(train["combo2"], y_train, class_weight={0: 1, 1: 2, 2: 2, 3: 8}, c_value=0.5)
    combo3_svm = _train_svm(train["combo3"], y_train, class_weight={0: 1, 1: 2, 2: 2, 3: 8}, c_value=0.5)
    multiview_svm = _train_multiview_svm(train, y_train, class_weight={0: 1, 1: 2, 2: 2, 3: 8}, c_value=0.5)

    candidates = {
        "word_svm": {
            "val_pred": word_svm.predict(val["target_text"]),
            "test_pred": word_svm.predict(test["target_text"]),
        },
        "char_word_svm": None,
        "multinomial_nb": None,
        "knn_retrieval": None,
        "extra_trees": None,
        "decision_tree": None,
        "combo2_svm": {
            "val_pred": combo2_svm.predict(val["combo2"]),
            "test_pred": combo2_svm.predict(test["combo2"]),
        },
        "combo3_svm": {
            "val_pred": combo3_svm.predict(val["combo3"]),
            "test_pred": combo3_svm.predict(test["combo3"]),
        },
        "multiview_svm": {
            "val_pred": multiview_svm.predict(val),
            "test_pred": multiview_svm.predict(test),
        },
        "word_lr_bias": None,
        "hierarchical": None,
        "lookup_hybrid": None,
        "rule_classifier": None,
        "tree_hgb": None,
    }

    calibrated_combo2 = CalibratedClassifierCV(combo2_svm, cv="prefit", method="sigmoid")
    calibrated_combo2.fit(val["combo2"], y_val)
    calibrated_val_scores = calibrated_combo2.predict_proba(val["combo2"])
    calibrated_bias = _best_bias_from_scores(y_val, calibrated_val_scores)
    candidates["combo2_calibrated_bias"] = {
        "val_pred": np.argmax(calibrated_val_scores + calibrated_bias, axis=1),
        "test_pred": np.argmax(calibrated_combo2.predict_proba(test["combo2"]) + calibrated_bias, axis=1),
    }

    ordinal_val_scores = _ordinal_threshold_probs(train["combo2"], y_train, val["combo2"], class_weight={0: 1, 1: 2, 2: 2, 3: 8})
    ordinal_bias = _best_bias_from_scores(y_val, ordinal_val_scores)
    ordinal_test_scores = _ordinal_threshold_probs(train["combo2"], y_train, test["combo2"], class_weight={0: 1, 1: 2, 2: 2, 3: 8})
    candidates["ordinal_threshold_lr"] = {
        "val_pred": np.argmax(ordinal_val_scores + ordinal_bias, axis=1),
        "test_pred": np.argmax(ordinal_test_scores + ordinal_bias, axis=1),
    }

    ordered_logit = _train_ordered_logit(
        train["combo2"],
        y_train,
        class_weight={0: 1, 1: 2, 2: 2, 3: 8},
        n_components=180,
        l2=1.0,
    )
    candidates["ordered_logit_svd"] = {
        "val_pred": ordered_logit.predict(val["combo2"]),
        "test_pred": ordered_logit.predict(test["combo2"]),
    }

    char_word_svm = _train_char_word_svm(train["combo2"], y_train, class_weight={0: 1, 1: 2, 2: 2, 3: 8}, c_value=0.75)
    candidates["char_word_svm"] = {
        "val_pred": char_word_svm.predict(val["combo2"]),
        "test_pred": char_word_svm.predict(test["combo2"]),
    }

    nb = _train_nb(train["role_task"], y_train)
    candidates["multinomial_nb"] = {
        "val_pred": nb.predict(val["role_task"]),
        "test_pred": nb.predict(test["role_task"]),
    }

    candidates["knn_retrieval"] = {
        "val_pred": _knn_classifier(train["combo2"], y_train, val["combo2"]),
        "test_pred": _knn_classifier(train["combo2"], y_train, test["combo2"]),
    }

    # SVD-fed tree models, using the same role+task view as a compact semantic summary.
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=250_000)
    svd = TruncatedSVD(n_components=250, random_state=42)
    train_x = svd.fit_transform(tfidf.fit_transform(train["role_sector_task"]))
    val_x = svd.transform(tfidf.transform(val["role_sector_task"]))
    test_x = svd.transform(tfidf.transform(test["role_sector_task"]))
    tree_weight = np.array([1.0 / np.bincount(y_train)[y] for y in y_train], dtype=float)

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    extra_trees.fit(train_x, y_train, sample_weight=tree_weight)
    candidates["extra_trees"] = {
        "val_pred": extra_trees.predict(val_x),
        "test_pred": extra_trees.predict(test_x),
    }

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    random_forest.fit(train_x, y_train, sample_weight=tree_weight)
    candidates["random_forest"] = {
        "val_pred": random_forest.predict(val_x),
        "test_pred": random_forest.predict(test_x),
    }

    decision_tree = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
    )
    decision_tree.fit(train_x, y_train, sample_weight=tree_weight)
    candidates["decision_tree"] = {
        "val_pred": decision_tree.predict(val_x),
        "test_pred": decision_tree.predict(test_x),
    }

    # Word LR with bias tuning
    word_lr = _train_lr(train["combo2"], y_train, class_weight={0: 1, 1: 2, 2: 2, 3: 8})
    val_scores = word_lr.predict_proba(val["combo2"])
    bias = _best_bias_from_scores(y_val, val_scores)
    candidates["word_lr_bias"] = {
        "val_pred": np.argmax(val_scores + bias, axis=1),
        "test_pred": np.argmax(word_lr.predict_proba(test["combo2"]) + bias, axis=1),
    }
    if args.include_teacher_score:
        score_aided_lr = _train_score_aided_lr(
            train[["combo2", "ai_impact_score"]], y_train, class_weight={0: 1, 1: 2, 2: 2, 3: 8}
        )
        candidates["teacher_score_aided_lr"] = {
            "val_pred": score_aided_lr.predict(val[["combo2", "ai_impact_score"]]),
            "test_pred": score_aided_lr.predict(test[["combo2", "ai_impact_score"]]),
        }

    blend_val_mats = [
        calibrated_val_scores,
        word_lr.predict_proba(val["combo2"]),
        nb.predict_proba(val["role_task"]),
        ordinal_val_scores,
        _softmax_rows(char_word_svm.decision_function(val["combo2"])),
    ]
    blend_test_mats = [
        calibrated_combo2.predict_proba(test["combo2"]),
        word_lr.predict_proba(test["combo2"]),
        nb.predict_proba(test["role_task"]),
        ordinal_test_scores,
        _softmax_rows(char_word_svm.decision_function(test["combo2"])),
    ]
    blended_val_scores, blended_test_scores, blend_weights = _search_weighted_blend(
        y_val,
        blend_val_mats,
        blend_test_mats,
        n_trials=2500,
        random_state=args.seed,
    )
    blend_bias = _best_bias_from_scores(y_val, blended_val_scores)
    candidates["weighted_prob_blend"] = {
        "val_pred": np.argmax(blended_val_scores + blend_bias, axis=1),
        "test_pred": np.argmax(blended_test_scores + blend_bias, axis=1),
    }

    # Hierarchical, lookup hybrid, rule classifier, tree model
    candidates["hierarchical"] = {
        "val_pred": _hierarchical_predict(train["combo2"], y_train, val["combo2"]),
        "test_pred": _hierarchical_predict(train["combo2"], y_train, test["combo2"]),
    }
    candidates["lookup_hybrid"] = {
        "val_pred": _lookup_hybrid(train["target_text"], y_train, val["target_text"], combo2_svm.predict(val["combo2"])),
        "test_pred": _lookup_hybrid(train["target_text"], y_train, test["target_text"], combo2_svm.predict(test["combo2"])),
    }
    candidates["rule_classifier"] = {
        "val_pred": _rule_classifier(train["combo2"], y_train, val["combo2"]),
        "test_pred": _rule_classifier(train["combo2"], y_train, test["combo2"]),
    }
    candidates["association_rules"] = {
        "val_pred": _association_rule_classifier(train["role_sector_task"], y_train, val["role_sector_task"]),
        "test_pred": _association_rule_classifier(train["role_sector_task"], y_train, test["role_sector_task"]),
    }
    candidates["tree_hgb"] = {
        "val_pred": _tree_classifier(train["combo2"], y_train, val["combo2"]),
        "test_pred": _tree_classifier(train["combo2"], y_train, test["combo2"]),
    }

    best_name = None
    best_val = -1.0
    best_test_pred = None
    for name, payload in candidates.items():
        val_pred = payload["val_pred"]
        test_pred = payload["test_pred"]
        val_metrics = _metrics(y_val, val_pred)
        test_metrics = _metrics(y_test, test_pred) if test_pred is not None else {}
        runs.append({"model": name, "split": "val", **val_metrics})
        runs.append({"model": name, "split": "test", **test_metrics})
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_name = name
            best_test_pred = test_pred

    leaderboard = pd.DataFrame(runs)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    if best_test_pred is not None:
        cm = confusion_matrix(y_test, best_test_pred, labels=list(range(len(CLASS_NAMES))))
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            output_dir / "best_model_test_confusion_matrix.csv"
        )
    _save_json(
        output_dir / "run_metadata.json",
        {
            "input": str(Path(args.input).resolve()),
            "label_order": bundle.label_order,
            "label_to_rank": bundle.label_to_rank,
            "score_by_label": bundle.score_by_label,
            "train_rows": int(len(split.train_idx)),
            "val_rows": int(len(split.val_idx)),
            "test_rows": int(len(split.test_idx)),
            "split_leakage_passed": leak_report.passed,
            "split_leakage_messages": leak_report.messages,
            "split_leakage_warnings": leak_report.warnings,
            "best_model_by_val_macro_f1": best_name,
            "best_val_macro_f1": best_val,
            "best_test_confusion_matrix_path": str((output_dir / "best_model_test_confusion_matrix.csv").resolve())
            if best_test_pred is not None
            else None,
            "blend_weights": blend_weights.tolist() if "blend_weights" in locals() else None,
            "teacher_score_included": args.include_teacher_score,
            "candidates": list(candidates.keys()),
        },
    )
    print(leaderboard.sort_values(["split", "macro_f1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
