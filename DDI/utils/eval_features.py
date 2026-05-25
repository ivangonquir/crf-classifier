import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def clean_feature_name(name, max_len=60):
    name = str(name)
    if len(name) > max_len:
        return name[:max_len - 3] + "..."
    return name


def save_global_barplot(features, values, title, outfile):
    plt.figure(figsize=(12, max(6, len(features) * 0.35)))

    y = np.arange(len(features))

    plt.barh(y, values)
    plt.yticks(y, [clean_feature_name(f) for f in features])

    plt.xlabel("Global importance")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def save_heatmap(matrix, features, classes, title, outfile):
    plt.figure(figsize=(max(12, len(features) * 0.35), max(5, len(classes) * 0.7)))

    plt.imshow(matrix, aspect="auto")

    plt.xticks(
        np.arange(len(features)),
        [clean_feature_name(f, max_len=35) for f in features],
        rotation=90,
        fontsize=8
    )

    plt.yticks(np.arange(len(classes)), classes)

    plt.colorbar(label="weight / importance")

    plt.title(title)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def get_model_matrix(estimator):
    """
    Devuelve una matriz de pesos de forma:
        n_clases x n_features

    Para modelos lineales:
        coef_

    Para MultinomialNB:
        feature_log_prob_

    Para ComplementNB:
        -feature_log_prob_
    """

    if isinstance(estimator, ComplementNB):
        return -estimator.feature_log_prob_, "ComplementNB"

    if isinstance(estimator, MultinomialNB):
        return estimator.feature_log_prob_, "MultinomialNB"

    if isinstance(estimator, LogisticRegression):
        return estimator.coef_, "LogisticRegression"

    if isinstance(estimator, LinearSVC):
        return estimator.coef_, "LinearSVC"

    return None, None


def analyze_model(estimator, vectorizer, outdir="plots", n=30):
    os.makedirs(outdir, exist_ok=True)

    names = np.asarray(vectorizer.get_feature_names_out())
    classes = np.asarray(estimator.classes_)

    matrix, model_name = get_model_matrix(estimator)

    if matrix is None:
        return None

    # Caso binario raro: sklearn puede devolver coef_ con shape (1, n_features)
    if matrix.shape[0] == 1 and len(classes) == 2:
        matrix = np.vstack([-matrix[0], matrix[0]])

    #
    # Importancia global:
    # para cada feature, miramos el peso máximo absoluto entre clases
    #
    global_importance = np.max(np.abs(matrix), axis=0)

    top_idx = np.argsort(global_importance)[::-1][:n]

    top_features = names[top_idx]
    top_global_values = global_importance[top_idx]
    top_matrix = matrix[:, top_idx]

    #
    # 1) Heatmap feature x clase
    #
    save_heatmap(
        top_matrix,
        top_features,
        classes,
        f"{model_name}: top {n} features by class",
        os.path.join(outdir, "summary_heatmap.png")
    )

    #
    # 2) Barplot global
    #
    order = np.argsort(top_global_values)

    save_global_barplot(
        top_features[order],
        top_global_values[order],
        f"{model_name}: global feature importance",
        os.path.join(outdir, "global_importance.png")
    )

    #
    # Report JSON
    #
    report = {
        "model_type": model_name,
        "plots": {
            "summary_heatmap": os.path.join(outdir, "summary_heatmap.png"),
            "global_importance": os.path.join(outdir, "global_importance.png")
        },
        "top_global_features": [
            {
                "feature": str(names[i]),
                "importance": float(global_importance[i])
            }
            for i in top_idx
        ],
        "top_features_by_class": {}
    }

    for class_i, cls in enumerate(classes):
        class_scores = matrix[class_i]

        pos_idx = np.argsort(class_scores)[::-1][:n]
        neg_idx = np.argsort(class_scores)[:n]

        if isinstance(estimator, (LogisticRegression, LinearSVC)):
            report["top_features_by_class"][str(cls)] = {
                "positive": [
                    {
                        "feature": str(names[i]),
                        "value": float(class_scores[i])
                    }
                    for i in pos_idx
                ],
                "negative": [
                    {
                        "feature": str(names[i]),
                        "value": float(class_scores[i])
                    }
                    for i in neg_idx
                ]
            }
        else:
            report["top_features_by_class"][str(cls)] = [
                {
                    "feature": str(names[i]),
                    "value": float(class_scores[i])
                }
                for i in pos_idx
            ]

    return report