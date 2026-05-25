#!/usr/bin/env python3
"""Train sklearn classifiers from TAB-separated feature streams (stdin)."""

import json
import random
import sys

import numpy as np
from joblib import dump
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from utils.eval_features import analyze_model
from utils.sklearn_model_registry import get_estimator_and_param_grid, is_grid_search_model


RANDOM_STATE = 42


def load_data(lines):
    features = []
    labels = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        fields = line.split("\t")
        label = fields[0]

        feats = {}
        for feat in fields[1:]:
            if "=" in feat:
                k, v = feat.split("=", 1)
                feats[k] = v

        features.append(feats)
        labels.append(label)

    return features, np.asarray(labels)


def train_baseline(X_train, y_train, model_file, report_file):
    classes = np.unique(y_train)
    clf = MultinomialNB(alpha=0.01)
    clf.partial_fit(X_train, y_train, classes)

    dump(clf, model_file)

    if report_file:
        report = {
            "model": "baseline",
            "note": "MultinomialNB(alpha=0.01) partial_fit without CV",
        }
        with open(report_file, "w", encoding="utf8") as f:
            json.dump(report, f, indent=2)


def train_grid_search(model_name, X_train, y_train, model_file, report_file, plot_dir, vectorizer):
    clf, grid = get_estimator_and_param_grid(model_name, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        clf,
        grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    dump(search.best_estimator_, model_file)

    if report_file:
        report = {
            "model": model_name,
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "model_analysis": analyze_model(
                search.best_estimator_,
                vectorizer,
                outdir=(plot_dir or "plots"),
                n=20,
            ),
        }

        with open(report_file, "w", encoding="utf8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]
    report_file = sys.argv[3] if len(sys.argv) > 3 else None
    plot_dir = sys.argv[4] if len(sys.argv) > 4 else None
    model_name = sys.argv[5] if len(sys.argv) > 5 else "baseline"

    train_features, y_train = load_data(sys.stdin)

    v = DictVectorizer()
    X_train = v.fit_transform(train_features)
    dump(v, vectorizer_file)

    if model_name == "baseline":
        train_baseline(X_train, y_train, model_file, report_file)
    elif is_grid_search_model(model_name):
        train_grid_search(model_name, X_train, y_train, model_file, report_file, plot_dir, v)
    else:
        print(f"Unknown model name: {model_name}", file=sys.stderr)
        sys.exit(1)
