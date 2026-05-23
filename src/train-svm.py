#!/usr/bin/env python3
"""Train Linear SVM classifier on clf.feat format (tag\\tfeat1\\tfeat2\\t...)."""

import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from joblib import dump
from sklearn_utils import load_train_data

model_file = sys.argv[1]
vectorizer_file = sys.argv[2]

features, labels = load_train_data(sys.stdin)

v = DictVectorizer()
X = v.fit_transform(features)

clf = LinearSVC(C=0.1, max_iter=2000)
clf.fit(X, labels)

dump(clf, model_file)
dump(v, vectorizer_file)
print(f"Saved model to {model_file}, vectorizer to {vectorizer_file}", file=sys.stderr)
