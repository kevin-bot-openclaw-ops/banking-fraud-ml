"""
Model Suite for Fraud Detection

Three-tier approach reflecting production ML engineering:

1. BASELINE: Logistic Regression
   - Interpretable, fast, good starting point
   - Establishes minimum bar
   - Interview angle: "always start simple"

2. ENSEMBLE: Random Forest
   - Handles non-linear patterns
   - Natural feature importance
   - Robust to outliers

3. PRODUCTION: XGBoost
   - State-of-the-art for tabular fraud data
   - Handles class imbalance natively
   - Industry standard at PayPal, Stripe, banks

Key fraud detection insight: we optimize for RECALL at acceptable PRECISION,
not accuracy. Missing a fraud (false negative) costs €100-5000. Blocking a
legitimate transaction (false positive) costs ~€2 (customer friction).
Cost matrix drives threshold selection.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from typing import Dict, Any


def build_logistic_regression(class_weight: str = "balanced") -> Pipeline:
    """
    Logistic Regression baseline with L2 regularization.

    class_weight='balanced': adjusts weights inversely proportional to
    class frequencies — key technique for imbalanced fraud data.
    Equivalent to oversampling minority class in closed form.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            C=1.0,  # inverse regularization strength
            solver="lbfgs",
            random_state=42,
        )),
    ])


def build_random_forest(n_estimators: int = 200) -> RandomForestClassifier:
    """
    Random Forest with class weighting.

    n_estimators=200: diminishing returns after ~100-200 trees.
    max_features='sqrt': standard for classification, reduces correlation
    between trees (a key property of good ensembles).

    Interview talking point: Random Forest is an ensemble of decorrelated
    decision trees via bootstrap sampling (bagging) + random feature selection.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        max_features="sqrt",
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )


def build_xgboost(
    scale_pos_weight: float = None,
    n_fraud: int = None,
    n_legit: int = None,
) -> xgb.XGBClassifier:
    """
    XGBoost with fraud-optimized hyperparameters.

    scale_pos_weight: ratio of negative to positive samples.
    Critical for imbalanced data — tells XGBoost how much to penalize
    missing a fraud transaction vs blocking a legitimate one.

    eval_metric='aucpr': Area under precision-recall curve.
    Better than AUC-ROC for imbalanced datasets because it focuses on
    the minority class (fraud). AUC-ROC can look good even when the
    model fails on the rare class.
    """
    if scale_pos_weight is None and n_fraud is not None and n_legit is not None:
        scale_pos_weight = n_legit / n_fraud

    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight or 200,  # ~0.5% fraud rate → 199x weight
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def get_all_models(n_fraud: int, n_legit: int) -> Dict[str, Any]:
    """
    Return dict of all models for comparison.

    Args:
        n_fraud: Count of fraud samples in training set
        n_legit: Count of legitimate samples in training set

    Returns:
        Dict mapping model name to sklearn-compatible estimator
    """
    return {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(n_fraud=n_fraud, n_legit=n_legit),
    }
