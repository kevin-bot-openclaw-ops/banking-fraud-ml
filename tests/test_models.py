"""Tests for model training and evaluation."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import generate_transaction_dataset
from src.feature_engineering import TransactionFeatureEngineer, prepare_train_test_split
from src.models import build_logistic_regression, build_random_forest, build_xgboost, get_all_models
from src.evaluator import evaluate_model, find_optimal_threshold


@pytest.fixture(scope="module")
def train_test_data():
    """Prepare features for model tests."""
    df = generate_transaction_dataset(
        n_customers=50,
        n_transactions=3000,
        fraud_rate=0.02,  # Higher for testing (more fraud samples)
        random_seed=42,
    )
    fe = TransactionFeatureEngineer()
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, fe)
    return X_train, X_test, y_train, y_test


def test_logistic_regression_trains(train_test_data):
    """LR should train and predict."""
    X_train, X_test, y_train, y_test = train_test_data
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})


def test_logistic_regression_has_proba(train_test_data):
    """LR should support predict_proba."""
    X_train, X_test, y_train, y_test = train_test_data
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_random_forest_trains(train_test_data):
    """RF should train and produce valid predictions."""
    X_train, X_test, y_train, y_test = train_test_data
    model = build_random_forest(n_estimators=50)  # fast for tests
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)


def test_random_forest_feature_importance(train_test_data):
    """RF should expose feature importance."""
    X_train, X_test, y_train, y_test = train_test_data
    model = build_random_forest(n_estimators=50)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    assert len(importances) == X_train.shape[1]
    assert (importances >= 0).all()
    assert abs(importances.sum() - 1.0) < 1e-5


def test_xgboost_trains(train_test_data):
    """XGBoost should train without errors."""
    X_train, X_test, y_train, y_test = train_test_data
    n_fraud = y_train.sum()
    n_legit = len(y_train) - n_fraud
    model = build_xgboost(n_fraud=int(n_fraud), n_legit=int(n_legit))
    model.set_params(n_estimators=50)  # fast for tests
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)


def test_evaluate_model_returns_required_keys(train_test_data):
    """evaluate_model should return all required metric keys."""
    X_train, X_test, y_train, y_test = train_test_data
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = evaluate_model("Test LR", y_test, y_pred, y_proba)

    required_keys = [
        "model", "precision", "recall", "f1", "auprc", "auroc",
        "tp", "fp", "tn", "fn", "net_value_eur",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_evaluate_model_metrics_in_range(train_test_data):
    """All metric values should be in valid ranges."""
    X_train, X_test, y_train, y_test = train_test_data
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = evaluate_model("Test", y_test, y_pred, y_proba)

    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["f1"] <= 1
    assert 0 <= result["auprc"] <= 1
    assert 0 <= result["auroc"] <= 1


def test_confusion_matrix_consistency(train_test_data):
    """TP + FP + TN + FN should equal total test samples."""
    X_train, X_test, y_train, y_test = train_test_data
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = evaluate_model("Test", y_test, y_pred, y_proba)
    total = result["tp"] + result["fp"] + result["tn"] + result["fn"]
    assert total == len(y_test)


def test_find_optimal_threshold(train_test_data):
    """find_optimal_threshold should return threshold in [0, 1]."""
    X_train, X_test, y_train, y_test = train_test_data
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold, precision, recall = find_optimal_threshold(y_test, y_proba, target_recall=0.5)
    assert 0 < threshold < 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1


def test_models_beat_random_baseline(train_test_data):
    """All models should have AUPRC better than random (= fraud rate)."""
    X_train, X_test, y_train, y_test = train_test_data
    n_fraud = int(y_train.sum())
    n_legit = int(len(y_train) - n_fraud)
    random_baseline = y_test.mean()  # random classifier AUPRC ≈ fraud rate

    models = get_all_models(n_fraud=n_fraud, n_legit=n_legit)
    # Use smaller trees for test speed
    for name in ["Random Forest"]:
        models[name].set_params(n_estimators=50)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        result = evaluate_model(name, y_test, y_pred, y_proba)
        assert result["auprc"] > random_baseline, (
            f"{name} AUPRC ({result['auprc']:.4f}) should beat random baseline ({random_baseline:.4f})"
        )
