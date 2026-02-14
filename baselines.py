"""
Baseline classifiers for comparison with the KG + BNN approach.

Models: XGBoost, Decision Tree, SVM, k-NN, AdaBoost, LightGBM.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from evaluate import evaluate_sklearn_model


def get_baseline_models(seed: int) -> dict:
    """Return a dict of {name: unfitted model}."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "SVM": SVC(kernel="rbf", random_state=seed),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=seed,
            algorithm="SAMME",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            verbose=-1,
        ),
    }


def train_and_evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict:
    """
    Train all baselines and return
    {model_name: {precision, recall, f1}}.
    """
    models = get_baseline_models(seed)
    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        metrics = evaluate_sklearn_model(clf, X_test, y_test)
        results[name] = metrics
    return results
