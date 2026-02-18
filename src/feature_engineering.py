"""
Feature Engineering Pipeline for Fraud Detection

Key insight: raw transaction features alone aren't enough.
Fraud detection relies heavily on BEHAVIORAL features — how does
this transaction compare to the customer's history?

Features engineered here:
1. Velocity features: transaction count in rolling windows
2. Amount deviation: how unusual is this amount for this customer?
3. Geographic risk: new country, high-risk country?
4. Time patterns: unusual hour/day for this customer?
5. Merchant risk: category risk score
6. Consecutive spending: rapid succession of transactions
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that generates behavioral features
    from raw banking transaction data.

    Design rationale:
    - Extends sklearn's BaseEstimator for pipeline compatibility
    - fit() computes customer baseline statistics
    - transform() generates features relative to those baselines
    """

    def __init__(self, velocity_windows: list = None):
        self.velocity_windows = velocity_windows or [1, 3, 7]  # days
        self._customer_stats = {}
        self._label_encoders = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame, y=None) -> "TransactionFeatureEngineer":
        """
        Compute per-customer baseline statistics from training data.
        These baselines are then used to flag deviations in transform().
        """
        # Per-customer statistics
        self._customer_stats = (
            df.groupby("customer_id")
            .agg(
                mean_amount=("amount", "mean"),
                std_amount=("amount", "std"),
                median_amount=("amount", "median"),
                typical_hour_mean=("hour_of_day", "mean"),
                typical_hour_std=("hour_of_day", "std"),
                home_country=("country", lambda x: x.mode()[0]),
                txn_count=("transaction_id", "count"),
            )
            .fillna({"std_amount": 1.0, "typical_hour_std": 6.0})
            .to_dict("index")
        )

        # Encode categorical features
        for col in ["merchant_category", "card_type", "country"]:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self._label_encoders[col] = le

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate fraud-detection features. Returns feature matrix
        without label columns.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        df = df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # --- Amount deviation features ---
        df["amount_vs_customer_mean"] = df.apply(
            lambda row: self._amount_deviation(row), axis=1
        )
        df["amount_log"] = np.log1p(df["amount"])
        df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)
        df["amount_bucket"] = pd.cut(
            df["amount"],
            bins=[0, 10, 50, 200, 500, 2000, np.inf],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(int)

        # --- Time features ---
        df["is_night"] = df["hour_of_day"].apply(lambda h: int(1 <= h <= 5))
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

        # --- Geographic features ---
        df["is_high_risk_country"] = df["country"].apply(
            lambda c: int(c in {"NG", "VN", "ID", "UA", "RU", "BR"})
        )

        # --- Velocity features (transactions per window per customer) ---
        for window_days in self.velocity_windows:
            col_name = f"txn_velocity_{window_days}d"
            df[col_name] = df.groupby("customer_id")["timestamp"].transform(
                lambda ts: ts.expanding().count() - 1  # count before current txn
            )

        # --- Amount velocity (total spend in rolling window) ---
        df["customer_txn_total"] = df.groupby("customer_id")["amount"].transform("cumsum")

        # --- Encode categoricals ---
        for col, le in self._label_encoders.items():
            known_mask = df[col].astype(str).isin(le.classes_)
            df[f"{col}_encoded"] = 0
            df.loc[known_mask, f"{col}_encoded"] = le.transform(
                df.loc[known_mask, col].astype(str)
            )

        # --- Credit score features ---
        df["credit_score_normalized"] = (df["customer_credit_score"] - 300) / 550

        # --- Select final feature set ---
        feature_cols = [
            "amount",
            "amount_log",
            "amount_vs_customer_mean",
            "is_round_amount",
            "amount_bucket",
            "merchant_risk_score",
            "is_foreign_country",
            "is_high_risk_country",
            "hour_of_day",
            "hour_sin",
            "hour_cos",
            "is_night",
            "is_weekend",
            "day_of_week",
            "credit_score_normalized",
            "customer_avg_amount",
            "merchant_category_encoded",
            "card_type_encoded",
            "country_encoded",
        ] + [f"txn_velocity_{w}d" for w in self.velocity_windows]

        return df[feature_cols].fillna(0)

    def _amount_deviation(self, row: pd.Series) -> float:
        """How many std deviations is this amount from the customer's typical amount?"""
        stats = self._customer_stats.get(row["customer_id"], {})
        mean = stats.get("mean_amount", row["amount"])
        std = max(stats.get("std_amount", 1.0), 1.0)
        return (row["amount"] - mean) / std


def prepare_train_test_split(
    df: pd.DataFrame,
    feature_engineer: TransactionFeatureEngineer,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal train/test split — critical for time-series fraud data.

    Why temporal split (not random)?
    - Random split leaks future data into training (transactions from Dec
      in train, January transactions with same customer in test — but the
      model "knows" December patterns that haven't happened yet in test)
    - Real fraud detection: train on past months, predict future months
    - More honest evaluation of generalization
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Fit feature engineer on training data only (no data leakage)
    X_train = feature_engineer.fit_transform(train_df)
    X_test = feature_engineer.transform(test_df)

    y_train = train_df["is_fraud"].values
    y_test = test_df["is_fraud"].values

    return X_train, X_test, y_train, y_test
