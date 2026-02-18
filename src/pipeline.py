"""
End-to-End Fraud Detection Pipeline

Orchestrates: data generation → feature engineering → model training
→ evaluation → threshold optimization → model persistence.

This is the main entry point for the ML workflow.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from .data_generator import generate_transaction_dataset
from .feature_engineering import TransactionFeatureEngineer, prepare_train_test_split
from .models import get_all_models
from .evaluator import evaluate_model, find_optimal_threshold, print_results_table


def run_pipeline(
    n_transactions: int = 50000,
    fraud_rate: float = 0.005,
    target_recall: float = 0.80,
    models_dir: str = "models",
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete fraud detection pipeline.

    Args:
        n_transactions: Number of synthetic transactions to generate
        fraud_rate: Fraction of fraudulent transactions
        target_recall: Minimum recall requirement for threshold optimization
        models_dir: Directory to save trained models
        random_seed: Random seed for reproducibility
        verbose: Print progress and results

    Returns:
        Dict with results for all models and the best model info
    """
    Path(models_dir).mkdir(exist_ok=True)

    # === Step 1: Generate Data ===
    if verbose:
        print("=" * 60)
        print("BANKING FRAUD ML DETECTION PIPELINE")
        print("=" * 60)
        print(f"\n[1/5] Generating {n_transactions:,} synthetic transactions...")

    df = generate_transaction_dataset(
        n_transactions=n_transactions,
        fraud_rate=fraud_rate,
        random_seed=random_seed,
    )
    n_fraud_total = df["is_fraud"].sum()

    if verbose:
        print(f"      Total transactions: {len(df):,}")
        print(f"      Fraud transactions: {n_fraud_total:,} ({df['is_fraud'].mean():.2%})")
        print(f"      Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # === Step 2: Feature Engineering ===
    if verbose:
        print("\n[2/5] Engineering features...")

    feature_engineer = TransactionFeatureEngineer()
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df, feature_engineer, random_state=random_seed
    )
    n_fraud_train = y_train.sum()
    n_legit_train = len(y_train) - n_fraud_train

    if verbose:
        print(f"      Train: {len(X_train):,} transactions ({n_fraud_train:,} fraud)")
        print(f"      Test:  {len(X_test):,} transactions ({y_test.sum():,} fraud)")
        print(f"      Features: {X_train.shape[1]}")

    # === Step 3: Train Models ===
    if verbose:
        print("\n[3/5] Training models...")

    models = get_all_models(n_fraud=int(n_fraud_train), n_legit=int(n_legit_train))
    trained_models = {}

    for name, model in models.items():
        if verbose:
            print(f"      Training {name}...", end="", flush=True)
        model.fit(X_train, y_train)
        trained_models[name] = model
        if verbose:
            print(" ✓")

    # === Step 4: Evaluate ===
    if verbose:
        print("\n[4/5] Evaluating models...")

    results = []
    model_details = {}

    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]

        # Find optimal threshold for target recall
        opt_threshold, opt_precision, opt_recall = find_optimal_threshold(
            y_test, y_proba, target_recall=target_recall
        )
        y_pred_opt = (y_proba >= opt_threshold).astype(int)

        result = evaluate_model(
            model_name=name,
            y_true=y_test,
            y_pred=y_pred_opt,
            y_proba=y_proba,
            threshold=opt_threshold,
        )
        results.append(result)
        model_details[name] = {
            "model": model,
            "y_proba": y_proba,
            "result": result,
            "threshold": opt_threshold,
        }

    if verbose:
        print_results_table(results)

    # === Step 5: Save Best Model ===
    best_result = max(results, key=lambda r: r["auprc"])
    best_name = best_result["model"]
    best_model = trained_models[best_name]

    model_path = Path(models_dir) / "best_model.joblib"
    fe_path = Path(models_dir) / "feature_engineer.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(feature_engineer, fe_path)

    if verbose:
        print(f"\n[5/5] Best model: {best_name}")
        print(f"      AUPRC: {best_result['auprc']:.4f}")
        print(f"      Recall: {best_result['recall']:.3f} ({best_result['tp']} fraud caught, "
              f"{best_result['fn']} missed)")
        print(f"      Precision: {best_result['precision']:.3f}")
        print(f"      Net business value: €{best_result['net_value_eur']:,.0f}")
        print(f"      Saved to: {model_path}")
        print("\n" + "=" * 60)
        print("Pipeline complete.")
        print("=" * 60)

    return {
        "results": results,
        "best_model_name": best_name,
        "best_result": best_result,
        "model_path": str(model_path),
        "feature_engineer_path": str(fe_path),
        "dataset_stats": {
            "n_total": len(df),
            "n_fraud": int(n_fraud_total),
            "fraud_rate": float(df["is_fraud"].mean()),
        },
    }


def predict_transaction(
    transaction: Dict[str, Any],
    model_path: str = "models/best_model.joblib",
    feature_engineer_path: str = "models/feature_engineer.joblib",
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Score a single transaction for fraud risk.

    Args:
        transaction: Dict with transaction fields
        model_path: Path to saved model
        feature_engineer_path: Path to saved feature engineer
        threshold: Decision threshold

    Returns:
        Dict with fraud_probability and is_fraud_predicted
    """
    model = joblib.load(model_path)
    fe = joblib.load(feature_engineer_path)

    df = pd.DataFrame([transaction])
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    X = fe.transform(df)
    proba = model.predict_proba(X)[0, 1]

    return {
        "fraud_probability": float(proba),
        "is_fraud_predicted": bool(proba >= threshold),
        "risk_level": "HIGH" if proba > 0.7 else "MEDIUM" if proba > threshold else "LOW",
        "threshold_used": threshold,
    }
