# CRF Classifier — Drug Named Entity Recognition

ML-based system for recognizing and classifying drug names in biomedical text using the DDI corpus.

## Setup

```bash
pip install nltk scikit-learn python-crfsuite joblib xgboost
python3 -c "import nltk; nltk.download('punkt_tab')"
```

## Pipeline Overview

The system has three stages: **feature extraction → training → prediction/evaluation**.
Feature extraction is done once; the same feature file is reused across all models.

---

## 1. Extract Features

```bash
python3 extract-features.py lab_resources/DDI/data/train > train.feat
python3 extract-features.py lab_resources/DDI/data/devel > devel.feat
python3 extract-features.py lab_resources/DDI/data/test  > test.feat
```

---

## 2. Train & Run Models

### CRF (best — 75.0% macro F1)

```bash
# Train
python3 train-crf.py model.crf < train.feat

# Predict
python3 predict.py model.crf < devel.feat > devel-CRF.out

# Evaluate
python3 evaluator.py NER lab_resources/DDI/data/devel devel-CRF.out
```

---

### Linear SVM (72.0% macro F1)

```bash
# Prepare classification feature file
cat train.feat | cut -f5- | grep -v ^$ > train.clf.feat

# Train
python3 train-svm.py model.svm vectorizer.svm < train.clf.feat

# Predict
python3 predict-ml.py model.svm vectorizer.svm < devel.feat > devel-SVM.out

# Evaluate
python3 evaluator.py NER lab_resources/DDI/data/devel devel-SVM.out
```

---

### XGBoost (63.0% macro F1)

```bash
# Prepare classification feature file (if not done already)
cat train.feat | cut -f5- | grep -v ^$ > train.clf.feat

# Train
python3 train-xgboost.py model.xgb vectorizer.xgb encoder.xgb < train.clf.feat

# Predict
python3 predict-xgboost.py model.xgb vectorizer.xgb encoder.xgb < devel.feat > devel-XGB.out

# Evaluate
python3 evaluator.py NER lab_resources/DDI/data/devel devel-XGB.out
```

---

### Naive Bayes (58.5% macro F1)

```bash
# Prepare classification feature file (if not done already)
cat train.feat | cut -f5- | grep -v ^$ > train.clf.feat

# Train
python3 train-nb.py model.nb vectorizer.nb < train.clf.feat

# Predict
python3 predict-ml.py model.nb vectorizer.nb < devel.feat > devel-NB.out

# Evaluate
python3 evaluator.py NER lab_resources/DDI/data/devel devel-NB.out
```

---

## 3. Run Full Pipeline (CRF + NB)

```bash
bash run.sh
```

---

## Results Summary (Devel Set)

| Model       | brand  | drug   | drug_n | group  | Macro F1 | Micro F1 |
|-------------|--------|--------|--------|--------|----------|----------|
| CRF         | 87.7%  | 92.0%  | 34.9%  | 85.2%  | **75.0%**| 89.3%    |
| SVM         | 87.1%  | 92.2%  | 24.1%  | 84.5%  | 72.0%    | 89.1%    |
| XGBoost     | 84.3%  | 90.1%  | 7.5%   | 70.1%  | 63.0%    | 84.4%    |
| Naive Bayes | 67.0%  | 85.5%  | 8.5%   | 72.7%  | 58.5%    | 76.4%    |

CRF outperforms the others because it models the full token sequence. The remaining classifiers predict each token independently.
