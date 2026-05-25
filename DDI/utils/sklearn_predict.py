#!/usr/bin/env python3

import sys

from joblib import load


def prepare_instances(xseq):
    features = []
    for interaction in xseq:
        token_dict = {feat.split("=")[0]: feat.split("=")[1] for feat in interaction[1:]}
        features.append(token_dict)
    return features


def main(argv):
    model_path = argv[1]
    vectorizer_path = argv[2]

    model = load(model_path)
    v = load(vectorizer_path)

    for line in sys.stdin:
        fields = line.strip("\n").split("\t")
        (sid, e1, e2) = fields[0:3]
        vectors = v.transform(prepare_instances([fields[4:]]))
        prediction = model.predict(vectors)

        if prediction != "null":
            print(sid, e1, e2, prediction[0], sep="|")


if __name__ == "__main__":
    main(sys.argv)
