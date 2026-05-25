"""Map logical model names -> (estimator, param_grid) for GridSearchCV."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def get_estimator_and_param_grid(name, random_state=42):
    if name == "multinomial_nb":
        return MultinomialNB(), {
            "alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        }

    if name == "complement_nb":
        return ComplementNB(), {
            "alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            "norm": [False, True],
        }

    if name == "logistic_regression":
        return LogisticRegression(max_iter=5000, random_state=random_state), {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        }

    if name == "linear_svc":
        return LinearSVC(max_iter=20000, dual="auto", random_state=random_state), {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        }

    raise ValueError(f"Unknown model for grid search: {name}")


def is_grid_search_model(name):
    return name in (
        "multinomial_nb",
        "complement_nb",
        "logistic_regression",
        "linear_svc"
    )
