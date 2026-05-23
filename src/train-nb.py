#!/usr/bin/env python3
"""Train Naive Bayes classifier on clf.feat format (tag\\tfeat1\\tfeat2\\t...)."""

import sys
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from joblib import dump
from sklearn_utils import load_train_data

model_file = sys.argv[1]
vectorizer_file = sys.argv[2]

features, labels = load_train_data(sys.stdin)
labels = np.asarray(labels)

v = DictVectorizer()
X = v.fit_transform(features)

clf = MultinomialNB(alpha=0.01)
clf.fit(X, labels)

dump(clf, model_file)
dump(v, vectorizer_file)
print(f"Saved model to {model_file}, vectorizer to {vectorizer_file}", file=sys.stderr)
