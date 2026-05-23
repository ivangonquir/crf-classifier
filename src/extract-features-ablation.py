#! /usr/bin/python3
"""Feature extraction with ablation support.
Usage: extract-features-ablation.py <datadir> [--ablate=shape|affixes|dict|context]
"""

import sys
import os
import argparse
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

_DRUGBANK = None
_HSDB = None

def _load_drugbank(path):
    entries = {}
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if '|' not in line:
                    continue
                name, etype = line.rsplit('|', 1)
                for word in name.lower().split():
                    entries.setdefault(word, set()).add(etype.strip())
    except FileNotFoundError:
        pass
    return entries

def _load_hsdb(path):
    tokens = set()
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                for word in line.strip().lower().split():
                    tokens.add(word)
    except FileNotFoundError:
        pass
    return tokens

def _init_dicts():
    global _DRUGBANK, _HSDB
    if _DRUGBANK is not None:
        return
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    _DRUGBANK = _load_drugbank(os.path.join(base, "lab_resources/DDI/resources/DrugBank.txt"))
    _HSDB = _load_hsdb(os.path.join(base, "lab_resources/DDI/resources/HSDB.txt"))


def _shape_features(t, prefix="", no_shape=False, no_affixes=False):
    feats = []
    tl = t.lower()
    feats.append(f"formlower{prefix}={tl}")
    if not no_affixes:
        feats.append(f"suf3{prefix}={t[-3:]}")
        feats.append(f"suf4{prefix}={t[-4:]}")
        feats.append(f"pref3{prefix}={t[:3]}")
        feats.append(f"pref4{prefix}={t[:4]}")
    if not no_shape:
        if t[0].isupper() and len(t) > 1 and t[1:].islower():
            feats.append(f"isTitle{prefix}")
        if t.isupper() and t.isalpha():
            feats.append(f"isUpper{prefix}")
        if any(c.isdigit() for c in t):
            feats.append(f"hasDigit{prefix}")
        if '-' in t:
            feats.append(f"hasDash{prefix}")
        if any(c.isupper() for c in t[1:]):
            feats.append(f"isCamel{prefix}")
    return feats

def _dict_features(t, prefix=""):
    feats = []
    tl = t.lower()
    if tl in _DRUGBANK:
        for etype in _DRUGBANK[tl]:
            feats.append(f"inDB{prefix}={etype}")
    if tl in _HSDB:
        feats.append(f"inHSDB{prefix}")
    return feats


def extract_features(tokens, ablate=None):
    _init_dicts()
    no_shape   = ablate == 'shape'
    no_affixes = ablate == 'affixes'
    no_dict    = ablate == 'dict'
    no_context = ablate == 'context'

    result = []
    for k in range(len(tokens)):
        feats = []
        t = tokens[k][0]

        feats.append("form=" + t)
        feats.extend(_shape_features(t, no_shape=no_shape, no_affixes=no_affixes))
        if not no_dict:
            feats.extend(_dict_features(t))

        if not no_context:
            if k > 0:
                tPrev = tokens[k-1][0]
                feats.append("formPrev=" + tPrev)
                feats.extend(_shape_features(tPrev, "Prev", no_shape=no_shape, no_affixes=no_affixes))
                if not no_dict:
                    feats.extend(_dict_features(tPrev, "Prev"))
            else:
                feats.append("BoS")

            if k < len(tokens) - 1:
                tNext = tokens[k+1][0]
                feats.append("formNext=" + tNext)
                feats.extend(_shape_features(tNext, "Next", no_shape=no_shape, no_affixes=no_affixes))
                if not no_dict:
                    feats.extend(_dict_features(tNext, "Next"))
            else:
                feats.append("EoS")
        else:
            feats.append("BoS" if k == 0 else "EoS" if k == len(tokens) - 1 else "")

        result.append([f for f in feats if f])
    return result


def tokenize(txt):
    offset = 0
    tks = []
    for t in word_tokenize(txt):
        offset = txt.find(t, offset)
        tks.append((t, offset, offset + len(t) - 1))
        offset += len(t)
    return tks

def get_tag(token, spans):
    (form, start, end) = token
    for (spanS, spanE, spanT) in spans:
        if start == spanS and end <= spanE:
            return "B-" + spanT
        elif start >= spanS and end <= spanE:
            return "I-" + spanT
    return "O"


parser = argparse.ArgumentParser()
parser.add_argument("datadir")
parser.add_argument("--ablate", choices=["shape", "affixes", "dict", "context"], default=None)
args = parser.parse_args()

for f in listdir(args.datadir):
    tree = parse(args.datadir + "/" + f)
    for s in tree.getElementsByTagName("sentence"):
        sid = s.attributes["id"].value
        stext = s.attributes["text"].value
        spans = []
        for e in s.getElementsByTagName("entity"):
            (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
            spans.append((int(start), int(end), e.attributes["type"].value))

        tokens = tokenize(stext)
        features = extract_features(tokens, ablate=args.ablate)

        for i in range(len(tokens)):
            tag = get_tag(tokens[i], spans)
            print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag,
                  "\t".join(features[i]), sep='\t')
        print()
