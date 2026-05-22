#!/usr/bin/env python3

import sys
from joblib import load


def instances(fi):
    xseq = []
    toks = []
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield xseq, toks
            xseq = []
            toks = []
            continue
        fields = line.split('\t')
        # fields: 0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N=features
        xseq.append(fields[5:])
        toks.append([fields[0], fields[1], fields[2], fields[3]])


def prepare_instances(xseq):
    features = []
    for token in xseq:
        d = {}
        for feat in token:
            if '=' in feat:
                k, v = feat.split('=', 1)
                d[k] = v
            else:
                d[feat] = 1
        features.append(d)
    return features


if __name__ == '__main__':

    model = load(sys.argv[1])
    v = load(sys.argv[2])

    for xseq, toks in instances(sys.stdin):
        if not xseq:
            continue
        vectors = v.transform(prepare_instances(xseq))
        predictions = model.predict(vectors)

        inside = False
        for k in range(len(predictions)):
            y = predictions[k]
            (sid, form, offS, offE) = toks[k]

            if y[0] == "B":
                entity_form = form
                entity_start = offS
                entity_end = offE
                entity_type = y[2:]
                inside = True
            elif y[0] == "I" and inside:
                entity_form += " " + form
                entity_end = offE
            elif y[0] == "O" and inside:
                print(sid, entity_start + "-" + entity_end, entity_form, entity_type, sep="|")
                inside = False

        if inside:
            print(sid, entity_start + "-" + entity_end, entity_form, entity_type, sep="|")
