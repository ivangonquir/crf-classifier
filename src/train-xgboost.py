#!/usr/bin/env python3
"""Train XGBoost classifier on clf.feat format (tag\\tfeat1\\tfeat2\\t...)."""

import sys
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn_utils import load_train_data

model_file = sys.argv[1]
vectorizer_file = sys.argv[2]
encoder_file = sys.argv[3]

features, labels = load_train_data(sys.stdin)

v = DictVectorizer()
X = v.fit_transform(features)

le = LabelEncoder()
y = le.fit_transform(labels)

clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    n_jobs=-1, eval_metric='mlogloss', verbosity=0)
clf.fit(X, y)

dump(clf, model_file)
dump(v, vectorizer_file)
dump(le, encoder_file)
print(f"Saved model to {model_file}, vectorizer to {vectorizer_file}, encoder to {encoder_file}", file=sys.stderr)
