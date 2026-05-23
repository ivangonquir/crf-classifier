#! /bin/bash

BASEDIR="lab_resources/DDI"

# convert datasets to feature vectors
echo "Extracting features..."
python3 src/extract-features.py $BASEDIR/data/train/ > features/train.feat
python3 src/extract-features.py $BASEDIR/data/devel/ > features/devel.feat

# train CRF model
echo "Training CRF model..."
python3 src/train-crf.py models/model.crf < features/train.feat
# run CRF model
echo "Running CRF model..."
python3 src/predict.py models/model.crf < features/devel.feat > results/devel-CRF.out
# evaluate CRF results
echo "Evaluating CRF results..."
python3 src/evaluator.py NER $BASEDIR/data/devel results/devel-CRF.out > results/devel-CRF.stats


#Extract Classification Features
cat features/train.feat | cut -f5- | grep -v ^$ > features/train.clf.feat


# train Naive Bayes model
echo "Training Naive Bayes model..."
python3 src/train-sklearn.py models/model.joblib models/vectorizer.joblib < features/train.clf.feat
# run Naive Bayes model
echo "Running Naive Bayes model..."
python3 src/predict-sklearn.py models/model.joblib models/vectorizer.joblib < features/devel.feat > results/devel-NB.out
# evaluate Naive Bayes results
echo "Evaluating Naive Bayes results..."
python3 src/evaluator.py NER $BASEDIR/data/devel results/devel-NB.out > results/devel-NB.stats

# remove auxiliary files.
rm features/train.clf.feat
