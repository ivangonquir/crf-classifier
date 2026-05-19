#!/usr/bin/env python3
"""Predict with any sklearn model (NB, SVM) that outputs BIO label strings directly."""

import sys
from joblib import load
from sklearn_utils import instances

model = load(sys.argv[1])
v = load(sys.argv[2])

for xseq, toks in instances(sys.stdin):
    if not xseq:
        continue
    vectors = v.transform(xseq)
    predictions = model.predict(vectors)

    inside = False
    for k, y in enumerate(predictions):
        sid, form, offS, offE = toks[k]
        if y[0] == 'B':
            if inside:
                print(sid, entity_start + '-' + entity_end, entity_form, entity_type, sep='|')
            entity_form = form
            entity_start = offS
            entity_end = offE
            entity_type = y[2:]
            inside = True
        elif y[0] == 'I' and inside:
            entity_form += ' ' + form
            entity_end = offE
        elif inside:
            print(sid, entity_start + '-' + entity_end, entity_form, entity_type, sep='|')
            inside = False

    if inside:
        print(sid, entity_start + '-' + entity_end, entity_form, entity_type, sep='|')
