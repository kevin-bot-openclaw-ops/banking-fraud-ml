"""Tests for synthetic data generation."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import generate_transaction_dataset


@pytest.fixture(scope="module")
def small_dataset():
    """Small dataset for fast tests."""
    return generate_transaction_dataset(
        n_customers=50,
        n_transactions=2000,
        fraud_rate=0.01,
        random_seed=42,
    )


def test_dataset_size(small_dataset):
    """Dataset should have exactly the requested number of transactions."""
    assert len(small_dataset) == 2000


def test_fraud_rate_approximate(small_dataset):
    """Fraud rate should be close to requested rate."""
    actual_rate = small_dataset["is_fraud"].mean()
    assert 0.005 <= actual_rate <= 0.02, f"Fraud rate {actual_rate:.3f} out of expected range"


def test_required_columns(small_dataset):
    """All required columns must be present."""
    required = [
        "transaction_id", "customer_id", "amount", "merchant_category",
        "merchant_risk_score", "country", "is_foreign_country", "card_type",
        "hour_of_day", "day_of_week", "timestamp", "is_fraud",
        "customer_credit_score", "customer_avg_amount",
    ]
    for col in required:
        assert col in small_dataset.columns, f"Missing column: {col}"


def test_no_null_values(small_dataset):
    """No null values in generated dataset."""
    assert small_dataset.isnull().sum().sum() == 0, "Dataset contains null values"


def test_amount_positive(small_dataset):
    """All transaction amounts should be positive."""
    assert (small_dataset["amount"] > 0).all(), "Found non-positive transaction amounts"


def test_hour_of_day_valid(small_dataset):
    """Hour of day should be 0-23."""
    assert small_dataset["hour_of_day"].between(0, 23).all()


def test_day_of_week_valid(small_dataset):
    """Day of week should be 0-6."""
    assert small_dataset["day_of_week"].between(0, 6).all()


def test_fraud_label_binary(small_dataset):
    """Fraud label should be 0 or 1 only."""
    assert set(small_dataset["is_fraud"].unique()).issubset({0, 1})


def test_is_foreign_country_binary(small_dataset):
    """Foreign country flag should be binary."""
    assert set(small_dataset["is_foreign_country"].unique()).issubset({0, 1})


def test_merchant_risk_score_range(small_dataset):
    """Merchant risk scores should be between 0 and 1."""
    assert small_dataset["merchant_risk_score"].between(0, 1).all()


def test_credit_score_range(small_dataset):
    """Credit scores should be in valid range (300-850)."""
    assert small_dataset["customer_credit_score"].between(300, 850).all()


def test_reproducibility():
    """Same seed should produce identical datasets."""
    df1 = generate_transaction_dataset(n_transactions=500, random_seed=99)
    df2 = generate_transaction_dataset(n_transactions=500, random_seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    """Different seeds should produce different datasets."""
    df1 = generate_transaction_dataset(n_transactions=500, random_seed=1)
    df2 = generate_transaction_dataset(n_transactions=500, random_seed=2)
    assert not df1["amount"].equals(df2["amount"])


def test_fraud_higher_night_rate(small_dataset):
    """Fraud transactions should have higher night-time rate than legit."""
    night_mask = small_dataset["hour_of_day"].between(1, 5)
    fraud_night = small_dataset.loc[small_dataset["is_fraud"] == 1, "hour_of_day"].between(1, 5).mean()
    legit_night = small_dataset.loc[small_dataset["is_fraud"] == 0, "hour_of_day"].between(1, 5).mean()
    # Fraud should have at least 1.5x more night transactions
    assert fraud_night > legit_night * 1.5, (
        f"Expected fraud to skew night. Fraud night: {fraud_night:.3f}, Legit night: {legit_night:.3f}"
    )
