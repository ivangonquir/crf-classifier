#! /usr/bin/python3
# Receta de features canónica v0 (comparación entre clasificadores).
# Ver README en esta carpeta.

import sys
from os import listdir

from xml.dom.minidom import parse

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "utils"))

from deptree import *
#import patterns

def context_pos(pos):
   try:
      return tree.get_tag(pos)
   except:
      return "NONE"

## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2) :
   feats = set()

   # get head token for each gold entity
   tkE1 = tree.get_fragment_head(entities[e1]['start'],entities[e1]['end'])
   tkE2 = tree.get_fragment_head(entities[e2]['start'],entities[e2]['end'])

   if tkE1 is not None and tkE2 is not None:
      # features for tokens in between E1 and E2
      #for tk in range(tkE1+1, tkE2) :
      # local lexical feature
      tk = tkE1 + 1

      try:
         while (tree.is_stopword(tk)):
            tk += 1
      except:
         return set()

      local_lemma = tree.get_lemma(tk).lower()
      local_tag = tree.get_tag(tk)

      feats.add("lib_local=" + local_lemma)
      feats.add("posib_local=" + local_tag)

      # features about paths in the tree
      lcs = tree.get_LCS(tkE1, tkE2)

      if lcs is None:
         return set()

      # LCS lexical feature
      lcs_lemma = tree.get_lemma(lcs).lower()
      lcs_tag = tree.get_tag(lcs)

      feats.add("lib_lcs=" + lcs_lemma)
      feats.add("posib_lcs=" + lcs_tag)
      
      eib = False
      for tk in range(tkE1+1, tkE2) :
         if tree.is_entity(tk, entities):
            eib = True 
      
	  # feature indicating the presence of an entity in between E1 and E2
      feats.add('eib='+ str(eib))
      
      feats.add("e1_left2_pos=" + context_pos(tkE1 - 2))
      feats.add("e1_left1_pos=" + context_pos(tkE1 - 1))
      feats.add("e1_right1_pos=" + context_pos(tkE1 + 1))
      feats.add("e1_right2_pos=" + context_pos(tkE1 + 2))

      feats.add("e2_left2_pos=" + context_pos(tkE2 - 2))
      feats.add("e2_left1_pos=" + context_pos(tkE2 - 1))
      feats.add("e2_right1_pos=" + context_pos(tkE2 + 1))
      feats.add("e2_right2_pos=" + context_pos(tkE2 + 2))

      def normalized_node(x):
         lemma = tree.get_lemma(x).lower()
         rel = tree.get_rel(x)

         if x == tkE1:
            lemma = "DRUG1"
         elif x == tkE2:
            lemma = "DRUG2"
         elif tree.is_entity(x, entities):
            lemma = "DRUG_OTHER"

         return lemma + "_" + rel

      path1 = tree.get_up_path(tkE1,lcs)
      path1 = "<".join([normalized_node(x) for x in path1])
      feats.add("path1="+path1)

      path2 = tree.get_down_path(lcs,tkE2)
      path2 = ">".join([normalized_node(x) for x in path2])
      feats.add("path2="+path2)

      path = path1+"<"+normalized_node(lcs)+">"+path2      
      feats.add("path="+path)
      
   return feats


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in sorted(listdir(datadir)):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")           
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction

            feats = extract_features(analysis,entities,id_e1,id_e2) 
            # resulting vector
            if len(feats) != 0:
              print(sid, id_e1, id_e2, dditype, "\t".join(sorted(feats)), sep="\t")

