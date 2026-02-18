"""
Synthetic Banking Transaction Data Generator

Generates realistic banking transaction data with fraud patterns.
No real customer data is used — all values are synthetically generated.

Key design decisions:
- Fraud rate: ~0.5% (realistic for card-not-present fraud)
- Features reflect real banking patterns: velocity, amount distribution,
  merchant risk, geographic patterns
- Fraud transactions have distinct statistical signatures
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


# Merchant categories with associated fraud risk scores (0-1)
MERCHANT_CATEGORIES = {
    "grocery": 0.02,
    "gas_station": 0.03,
    "restaurant": 0.04,
    "retail_clothing": 0.08,
    "electronics": 0.18,
    "online_marketplace": 0.22,
    "travel_booking": 0.15,
    "wire_transfer": 0.35,
    "crypto_exchange": 0.42,
    "gift_cards": 0.48,
    "atm_withdrawal": 0.12,
    "pharmacy": 0.05,
    "utilities": 0.01,
    "entertainment": 0.07,
    "luxury_goods": 0.25,
}

COUNTRIES = ["ES", "PL", "DE", "FR", "NL", "US", "GB", "CZ", "HU", "RO"]
HIGH_RISK_COUNTRIES = {"NG", "VN", "ID", "UA", "RU", "BR"}  # for fraud transactions

CARD_TYPES = ["visa_debit", "visa_credit", "mastercard_debit", "mastercard_credit", "amex"]


def generate_transaction_dataset(
    n_customers: int = 500,
    n_transactions: int = 50000,
    fraud_rate: float = 0.005,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic banking transaction dataset.

    Args:
        n_customers: Number of unique customers
        n_transactions: Total number of transactions
        fraud_rate: Fraction of transactions that are fraudulent (default 0.5%)
        random_seed: For reproducibility

    Returns:
        DataFrame with transaction features and fraud label
    """
    rng = np.random.default_rng(random_seed)
    n_fraud = int(n_transactions * fraud_rate)
    n_legitimate = n_transactions - n_fraud

    # --- Customer profiles ---
    customer_ids = [f"CUST_{i:06d}" for i in range(n_customers)]
    customer_avg_amount = rng.lognormal(mean=4.0, sigma=0.8, size=n_customers)  # €55 median
    customer_home_country = rng.choice(COUNTRIES, size=n_customers)
    customer_credit_score = rng.integers(300, 850, size=n_customers)

    # --- Generate legitimate transactions ---
    legit_records = []
    base_time = datetime(2025, 1, 1)

    for _ in range(n_legitimate):
        cust_idx = rng.integers(0, n_customers)
        cust_id = customer_ids[cust_idx]
        avg_amt = customer_avg_amount[cust_idx]
        home_country = customer_home_country[cust_idx]

        # Transaction amount: log-normal around customer's typical amount
        amount = float(np.clip(rng.lognormal(np.log(avg_amt), 0.6), 1.0, 10000.0))

        merchant_cat = rng.choice(list(MERCHANT_CATEGORIES.keys()))
        # Mostly home country, occasionally foreign
        country = home_country if rng.random() > 0.05 else rng.choice(COUNTRIES)

        # Time: weighted toward business hours, realistic week pattern
        days_offset = float(rng.integers(0, 365))
        hour = float(rng.choice(
            range(24),
            p=_hour_weights(is_fraud=False),
        ))
        txn_time = base_time + timedelta(days=days_offset, hours=hour)

        legit_records.append({
            "transaction_id": f"TXN_{rng.integers(100000000, 999999999):09d}",
            "customer_id": cust_id,
            "amount": round(amount, 2),
            "merchant_category": merchant_cat,
            "merchant_risk_score": MERCHANT_CATEGORIES[merchant_cat],
            "country": country,
            "is_foreign_country": int(country != home_country),
            "card_type": rng.choice(CARD_TYPES),
            "hour_of_day": txn_time.hour,
            "day_of_week": txn_time.weekday(),
            "timestamp": txn_time,
            "customer_credit_score": customer_credit_score[cust_idx],
            "customer_avg_amount": round(avg_amt, 2),
            "is_fraud": 0,
        })

    # --- Generate fraudulent transactions ---
    fraud_records = []
    for _ in range(n_fraud):
        # Fraudsters target real customer accounts
        cust_idx = rng.integers(0, n_customers)
        cust_id = customer_ids[cust_idx]
        avg_amt = customer_avg_amount[cust_idx]
        home_country = customer_home_country[cust_idx]

        # Fraud patterns: unusual amounts (often large or just below limit)
        fraud_pattern = rng.choice(["large_amount", "round_amount", "split_transactions"])
        if fraud_pattern == "large_amount":
            amount = float(np.clip(rng.lognormal(np.log(avg_amt * 8), 0.3), 100.0, 15000.0))
        elif fraud_pattern == "round_amount":
            amount = float(rng.choice([500, 1000, 2000, 5000, 10000]))
        else:
            amount = float(np.clip(rng.lognormal(np.log(avg_amt * 0.5), 0.2), 1.0, 500.0))

        # High-risk merchant categories more common
        merchant_cat = rng.choice(
            list(MERCHANT_CATEGORIES.keys()),
            p=_merchant_fraud_weights(),
        )

        # Often foreign countries, sometimes high-risk
        if rng.random() > 0.3:
            country = rng.choice(list(HIGH_RISK_COUNTRIES))
        else:
            country = rng.choice(COUNTRIES)

        # Fraud often happens at unusual hours (late night)
        days_offset = float(rng.integers(0, 365))
        hour = float(rng.choice(range(24), p=_hour_weights(is_fraud=True)))
        txn_time = base_time + timedelta(days=days_offset, hours=hour)

        fraud_records.append({
            "transaction_id": f"TXN_{rng.integers(100000000, 999999999):09d}",
            "customer_id": cust_id,
            "amount": round(amount, 2),
            "merchant_category": merchant_cat,
            "merchant_risk_score": MERCHANT_CATEGORIES[merchant_cat],
            "country": country,
            "is_foreign_country": int(country != home_country),
            "card_type": rng.choice(CARD_TYPES),
            "hour_of_day": txn_time.hour,
            "day_of_week": txn_time.weekday(),
            "timestamp": txn_time,
            "customer_credit_score": customer_credit_score[cust_idx],
            "customer_avg_amount": round(avg_amt, 2),
            "is_fraud": 1,
        })

    # Combine and shuffle
    df = pd.DataFrame(legit_records + fraud_records)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def _hour_weights(is_fraud: bool) -> list:
    """
    Return probability weights for hour of day.
    Legitimate transactions peak at business hours.
    Fraudulent transactions more common at night (2-5am).
    """
    if is_fraud:
        # Peak at 2-5am
        weights = [0.08, 0.07, 0.09, 0.10, 0.09, 0.05,
                   0.03, 0.02, 0.02, 0.02, 0.02, 0.02,
                   0.02, 0.02, 0.02, 0.02, 0.03, 0.04,
                   0.06, 0.06, 0.07, 0.06, 0.05, 0.06]
    else:
        # Peak at 10am-8pm
        weights = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
                   0.03, 0.04, 0.05, 0.07, 0.08, 0.08,
                   0.07, 0.07, 0.07, 0.07, 0.06, 0.06,
                   0.06, 0.05, 0.05, 0.04, 0.03, 0.02]
    total = sum(weights)
    return [w / total for w in weights]


def _merchant_fraud_weights() -> list:
    """Higher probability for high-risk merchant categories in fraud transactions."""
    categories = list(MERCHANT_CATEGORIES.keys())
    # Boost crypto, gift cards, wire transfers, online marketplaces
    boost = {
        "crypto_exchange": 4.0,
        "gift_cards": 4.0,
        "wire_transfer": 3.5,
        "online_marketplace": 3.0,
        "luxury_goods": 2.5,
        "electronics": 2.0,
    }
    weights = [boost.get(cat, 0.5) for cat in categories]
    total = sum(weights)
    return [w / total for w in weights]


if __name__ == "__main__":
    print("Generating synthetic banking transaction dataset...")
    df = generate_transaction_dataset(n_transactions=50000)
    print(f"Generated {len(df):,} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Fraud count: {df['is_fraud'].sum():,}")
    print(f"\nSample:\n{df.head()}")
    print(f"\nFeatures: {list(df.columns)}")
