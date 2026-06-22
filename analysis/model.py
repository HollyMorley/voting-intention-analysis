"""Model definitions and cross-validated evaluation.

The pipelines bundle imputation (+ scaling for the linear model) with the
estimator, so cross-validation refits them *per fold* — no leakage from the
held-out data into the medians/means used to fill and scale.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score


def make_lr_pipeline() -> Pipeline:
    """Logistic regression: median-impute -> standardise -> fit."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(max_iter=1000)),
    ])


def make_rf_pipeline(impute: bool = True) -> Pipeline:
    """Random forest: median-impute -> fit (trees need no scaling)."""
    if impute:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=0, n_jobs=-1)),
        ])
    else:
        return Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=0,
                n_jobs=-1)),
        ])



def dummy_baselines(X, y, cv=5, seed=0,
                    strategies=("most_frequent", "stratified", "uniform")) -> pd.Series:
    """CV accuracy of trivial classifiers — the 'no real signal' reference points.

    most_frequent : always predict the largest class (= its prevalence)
    stratified    : guess in proportion to class frequencies
    uniform       : guess uniformly at random (= 1 / n_classes)
    A real model must clear these to be worth anything.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    out = {s: cross_val_score(DummyClassifier(strategy=s, random_state=seed),
                              X, y, cv=splitter, scoring="accuracy").mean()
           for s in strategies}
    return pd.Series(out, name="cv_accuracy").round(3)


def cross_validate_model(estimator, X, y, cv=10, scoring=("accuracy", "f1_macro"), seed=0) -> pd.Series:
    """Stratified k-fold CV for one estimator.

    Returns a Series with mean and std of each metric across folds, e.g.
        accuracy_mean, accuracy_std, f1_macro_mean, f1_macro_std
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    results = cross_validate(estimator, X, y, cv=splitter, scoring=list(scoring))

    summary = {}
    for metric in scoring:
        scores = results[f"test_{metric}"]
        summary[f"{metric}_mean"] = scores.mean()
        summary[f"{metric}_std"] = scores.std()
    return pd.Series(summary)


def compare_models(models: dict, X, y, cv=5, scoring=("accuracy", "f1_macro"), seed=0) -> pd.DataFrame:
    """Cross-validate each named estimator; return one row per model."""
    return pd.DataFrame({
        name: cross_validate_model(est, X, y, cv=cv, scoring=scoring, seed=seed)
        for name, est in models.items()
    }).T.round(3)


def grouped_permutation_importance(make_pipeline, X, y, groups, cv=10, n_repeats=10,
                                   score_fn=balanced_accuracy_score, seed=0) -> pd.DataFrame:
    """Grouped permutation importance, computed within each CV fold.

    Per fold: fit a fresh pipeline on the training part, then for each group
    shuffle all its columns together on the held-out part and record the drop in
    `score_fn`. Returns one row per (fold, repeat) with a '_fold' column; averaging
    per fold gives a spread that reflects generalisation uncertainty, not just
    shuffle noise on one split. `groups` maps a label -> columns (e.g.
    encode.curated_groups), so a multi-column block (region, a life-background
    sub-domain) is judged as one question. High value = the model relies on this
    block *given everything else*; a redundant block scores ~0 here.

    Defaults to balanced accuracy: the drop is meaningful even when the target is
    very imbalanced (raw-accuracy drops would be tiny and dominated by big classes).
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    rows = []

    for fold_idx, (tr, te) in enumerate(splitter.split(X, y)):
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        m = make_pipeline()
        m.fit(X_tr, y_tr)
        baseline = score_fn(y_te, m.predict(X_te))

        for _ in range(n_repeats):
            row = {"_fold": fold_idx}
            for name, cols in groups.items():
                cols = [c for c in cols if c in X_te.columns]
                if not cols:
                    continue
                Xs = X_te.copy()
                Xs[cols] = Xs[cols].to_numpy()[rng.permutation(len(X_te))]
                row[name] = baseline - score_fn(y_te, m.predict(Xs))
            rows.append(row)

    return pd.DataFrame(rows)


def univariate_score(make_pipeline, X, y, groups, cv=10,
                     score_fn=balanced_accuracy_score, seed=0) -> pd.DataFrame:
    """Standalone predictive power of each group: fit a model on ONLY that group's
    columns. Returns a (cv, n_groups) DataFrame of held-out scores; spread across
    rows = fold-to-fold variability.

    Read alongside grouped_permutation_importance: high here but low there = signal
    that is real but redundant given the rest (e.g. the cosmopolitan background block
    vs the nationalism–internationalism value axis).
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    rows = {name: [] for name in groups}

    for tr, te in splitter.split(X, y):
        for name, cols in groups.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                rows[name].append(np.nan)
                continue
            m = make_pipeline()
            m.fit(X.iloc[tr][cols], y.iloc[tr])
            rows[name].append(score_fn(y.iloc[te], m.predict(X.iloc[te][cols])))

    return pd.DataFrame(rows)
