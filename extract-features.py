#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize


## --------- Load external drug dictionaries -----------
## DrugBank: "name|type" per line; HSDB: one drug name per line (all drug_n)

def _load_drugbank(path):
    """Return dict mapping lowercase token -> set of entity types it appears in."""
    entries = {}
    try:
        with open(path) as f:
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
    """Return set of lowercase tokens appearing in HSDB drug names."""
    tokens = set()
    try:
        with open(path) as f:
            for line in f:
                for word in line.strip().lower().split():
                    tokens.add(word)
    except FileNotFoundError:
        pass
    return tokens

# Paths relative to this script's location
_script_dir = listdir.__module__ and ""  # placeholder; resolved at runtime below
_DRUGBANK = None
_HSDB = None

def _init_dicts():
    global _DRUGBANK, _HSDB
    if _DRUGBANK is not None:
        return
    base = sys.argv[0].rsplit('/', 1)[0] if '/' in sys.argv[0] else '.'
    _DRUGBANK = _load_drugbank(base + "/lab_resources/DDI/resources/DrugBank.txt")
    _HSDB = _load_hsdb(base + "/lab_resources/DDI/resources/HSDB.txt")


## --------- Per-token shape features -----------

def _shape_features(t, prefix=""):
    """Return list of active binary shape features for token t."""
    feats = []
    tl = t.lower()
    feats.append(f"formlower{prefix}={tl}")
    feats.append(f"suf3{prefix}={t[-3:]}")
    feats.append(f"suf4{prefix}={t[-4:]}")
    feats.append(f"pref3{prefix}={t[:3]}")
    feats.append(f"pref4{prefix}={t[:4]}")
    if t[0].isupper() and len(t) > 1 and t[1:].islower():
        feats.append(f"isTitle{prefix}")
    if t.isupper() and t.isalpha():
        feats.append(f"isUpper{prefix}")
    if any(c.isdigit() for c in t):
        feats.append(f"hasDigit{prefix}")
    if '-' in t:
        feats.append(f"hasDash{prefix}")
    # camelCase: has uppercase after the first char
    if any(c.isupper() for c in t[1:]):
        feats.append(f"isCamel{prefix}")
    return feats

def _dict_features(t, prefix=""):
    """Return active dictionary features for token t."""
    feats = []
    tl = t.lower()
    if tl in _DRUGBANK:
        for etype in _DRUGBANK[tl]:
            feats.append(f"inDB{prefix}={etype}")
    if tl in _HSDB:
        feats.append(f"inHSDB{prefix}")
    return feats


## --------- tokenize sentence -----------
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag -----------
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"

## --------- Feature extractor -----------
## -- Extract features for each token in given sentence

def extract_features(tokens) :
   _init_dicts()

   result = []
   for k in range(0, len(tokens)):
      tokenFeatures = []
      t = tokens[k][0]

      # Current token: form, shape, and dictionary features
      tokenFeatures.append("form=" + t)
      tokenFeatures.extend(_shape_features(t))
      tokenFeatures.extend(_dict_features(t))

      if k > 0:
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev=" + tPrev)
         tokenFeatures.extend(_shape_features(tPrev, "Prev"))
         tokenFeatures.extend(_dict_features(tPrev, "Prev"))
      else:
         tokenFeatures.append("BoS")

      if k < len(tokens) - 1:
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext=" + tNext)
         tokenFeatures.extend(_shape_features(tNext, "Next"))
         tokenFeatures.extend(_dict_features(tNext, "Next"))
      else:
         tokenFeatures.append("EoS")

      result.append(tokenFeatures)

   return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
