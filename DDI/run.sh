#! /bin/bash

BASEDIR=../lab_resources/DDI

OUTDIR=outputs
FEATURESDIR=$OUTDIR/features
MODELDIR=$OUTDIR/model
PREDIR=$OUTDIR/predictions
STATSDIR=$OUTDIR/stats

mkdir -p $FEATURESDIR $MODELDIR $PREDIR $STATSDIR

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
elif command -v py >/dev/null 2>&1; then
    PYTHON=py
else
    echo "Python not found"
    exit 1
fi

./corenlp-server.sh -quiet true -port 9000 -timeout 15000 &
sleep 1

echo "Extracting features"

$PYTHON extract-features.py $BASEDIR/data/devel/ > $FEATURESDIR/devel.cod &
$PYTHON extract-features.py $BASEDIR/data/train/ | tee $FEATURESDIR/train.cod | cut -f4- > $FEATURESDIR/train.cod.cl

kill `cat /tmp/corenlp-server.running` 2>/dev/null

echo "Training model"

$PYTHON train-sklearn.py \
    $MODELDIR/model.joblib \
    $MODELDIR/vectorizer.joblib \
    < $FEATURESDIR/train.cod.cl

echo "Running model..."

$PYTHON predict-sklearn.py \
    $MODELDIR/model.joblib \
    $MODELDIR/vectorizer.joblib \
    < $FEATURESDIR/devel.cod \
    > $PREDIR/devel.out

echo "Evaluating results..."

$PYTHON evaluator.py DDI \
    $BASEDIR/data/devel/ \
    $PREDIR/devel.out \
    > $STATSDIR/devel.stats