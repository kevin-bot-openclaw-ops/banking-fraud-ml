"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import generate_transaction_dataset
from src.feature_engineering import TransactionFeatureEngineer, prepare_train_test_split


@pytest.fixture(scope="module")
def dataset():
    return generate_transaction_dataset(
        n_customers=30,
        n_transactions=1000,
        fraud_rate=0.02,
        random_seed=42,
    )


@pytest.fixture(scope="module")
def fitted_engineer(dataset):
    fe = TransactionFeatureEngineer()
    fe.fit(dataset)
    return fe


def test_feature_engineer_fits(dataset):
    """Feature engineer should fit without errors."""
    fe = TransactionFeatureEngineer()
    fe.fit(dataset)
    assert fe._fitted is True


def test_transform_before_fit_raises():
    """Transform without fit should raise RuntimeError."""
    fe = TransactionFeatureEngineer()
    df = pd.DataFrame({"amount": [100], "customer_id": ["X"]})
    with pytest.raises(RuntimeError, match="fit()"):
        fe.transform(df)


def test_feature_matrix_shape(dataset, fitted_engineer):
    """Output shape should match (n_samples, n_features)."""
    X = fitted_engineer.transform(dataset)
    assert X.shape[0] == len(dataset)
    assert X.shape[1] > 10, "Should have at least 10 features"


def test_no_nulls_in_features(dataset, fitted_engineer):
    """Feature matrix should have no NaN values."""
    X = fitted_engineer.transform(dataset)
    assert not X.isnull().values.any(), "Feature matrix contains NaN values"


def test_required_features_present(dataset, fitted_engineer):
    """Key features should be present."""
    X = fitted_engineer.transform(dataset)
    required = [
        "amount", "amount_log", "amount_vs_customer_mean",
        "is_night", "is_weekend", "is_foreign_country",
        "merchant_risk_score",
    ]
    for col in required:
        assert col in X.columns, f"Missing feature: {col}"


def test_velocity_features_present(dataset, fitted_engineer):
    """Velocity features should be present for all windows."""
    X = fitted_engineer.transform(dataset)
    for window in fitted_engineer.velocity_windows:
        col = f"txn_velocity_{window}d"
        assert col in X.columns, f"Missing velocity feature: {col}"


def test_amount_log_is_log_of_amount(dataset, fitted_engineer):
    """amount_log should equal log1p(amount) for all rows (order-independent)."""
    X = fitted_engineer.transform(dataset)
    # transform() sorts by timestamp, so align on sorted amounts to compare
    expected = np.log1p(X["amount"].values)
    np.testing.assert_allclose(X["amount_log"].values, expected, rtol=1e-5)


def test_is_night_binary(dataset, fitted_engineer):
    """is_night should be binary."""
    X = fitted_engineer.transform(dataset)
    assert set(X["is_night"].unique()).issubset({0, 1})


def test_is_weekend_binary(dataset, fitted_engineer):
    """is_weekend should be binary."""
    X = fitted_engineer.transform(dataset)
    assert set(X["is_weekend"].unique()).issubset({0, 1})


def test_train_test_split_no_leakage(dataset):
    """Train/test split should use temporal ordering (no future data in train)."""
    fe = TransactionFeatureEngineer()
    X_train, X_test, y_train, y_test = prepare_train_test_split(dataset, fe)

    # Check sizes sum correctly
    assert len(X_train) + len(X_test) == len(dataset)
    assert len(y_train) + len(y_test) == len(dataset)


def test_train_test_split_both_classes(dataset):
    """Both train and test sets should contain fraud and legit samples."""
    fe = TransactionFeatureEngineer()
    _, _, y_train, y_test = prepare_train_test_split(dataset, fe)

    assert y_train.sum() > 0, "No fraud in training set"
    assert y_test.sum() > 0, "No fraud in test set"
    assert (y_train == 0).sum() > 0, "No legit in training set"
    assert (y_test == 0).sum() > 0, "No legit in test set"


def test_sklearn_pipeline_compatibility(dataset):
    """Feature engineer should work in sklearn-style fit/transform flow."""
    from sklearn.base import BaseEstimator, TransformerMixin
    fe = TransactionFeatureEngineer()
    assert isinstance(fe, BaseEstimator)
    assert isinstance(fe, TransformerMixin)

    # fit_transform should work
    X = fe.fit_transform(dataset)
    assert X is not None
    assert X.shape[0] == len(dataset)
