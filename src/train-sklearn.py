#!/usr/bin/env python3

import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import argparse
from joblib import dump



def load_data(data):
    features = []
    labels = []
    for line in data:
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')
        labels.append(fields[0])
        d = {}
        for feat in fields[1:]:
            if '=' in feat:
                k, v = feat.split('=', 1)
                d[k] = v
            else:
                d[feat] = 1
        features.append(d)
    return features, labels


if __name__ == '__main__':

	model_file = sys.argv[1]
	vectorizer_file = sys.argv[2] 	

	train_features, y_train = load_data(sys.stdin)
	y_train = np.asarray(y_train)
	classes = np.unique(y_train)

	v = DictVectorizer()
	X_train = v.fit_transform(train_features)

	clf = MultinomialNB(alpha=0.01)
	clf.partial_fit(X_train, y_train, classes)

	#Save classifier and DictVectorizer
	dump(clf, model_file) 
	dump(v, vectorizer_file)

	 






		


	 
