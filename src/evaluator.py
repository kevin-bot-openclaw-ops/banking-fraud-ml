"""
Fraud Detection Evaluator

The right metrics for fraud detection:

WRONG metrics (commonly misused):
- Accuracy: 99.5% accuracy by predicting "not fraud" on everything. Useless.

RIGHT metrics:
- Precision: Of all predicted frauds, how many were real? (false alert rate)
- Recall: Of all real frauds, how many did we catch? (miss rate — critical!)
- F1: Harmonic mean (balance of precision and recall)
- AUPRC: Area under Precision-Recall curve (best single metric for imbalanced)
- Business cost: What's the actual € impact of our threshold choice?

Business cost model (simplified):
- False negative (missed fraud): avg transaction amount (€200)
- False positive (blocked legit): €2 (customer friction, potential churn)
- Target: maximize recall while keeping precision > 30%
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    avg_fraud_amount: float = 200.0,
    fp_cost: float = 2.0,
) -> Dict[str, Any]:
    """
    Comprehensive fraud detection evaluation.

    Args:
        model_name: Display name for reports
        y_true: True labels
        y_pred: Binary predictions at threshold
        y_proba: Fraud probability scores
        threshold: Decision threshold (default 0.5, often tuned lower for fraud)
        avg_fraud_amount: Average fraud transaction amount in EUR
        fp_cost: Cost of blocking a legitimate transaction in EUR

    Returns:
        Dict with all metrics and business cost analysis
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auprc = average_precision_score(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)

    # Business cost analysis
    fraud_caught_value = tp * avg_fraud_amount  # money saved by catching fraud
    missed_fraud_cost = fn * avg_fraud_amount   # money lost by missing fraud
    false_alert_cost = fp * fp_cost             # cost of blocking legit transactions
    net_value = fraud_caught_value - missed_fraud_cost - false_alert_cost

    return {
        "model": model_name,
        "threshold": threshold,
        # Core metrics
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "auroc": auroc,
        # Confusion matrix
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # Business metrics
        "fraud_caught_pct": recall,
        "fraud_caught_value_eur": fraud_caught_value,
        "missed_fraud_cost_eur": missed_fraud_cost,
        "false_alert_cost_eur": false_alert_cost,
        "net_value_eur": net_value,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.85,
) -> Tuple[float, float, float]:
    """
    Find the probability threshold that maximizes F1 while maintaining
    target recall level.

    In fraud detection, you often set a minimum recall requirement
    (e.g., "catch at least 85% of fraud") then maximize precision within
    that constraint.

    Returns:
        (optimal_threshold, precision_at_threshold, recall_at_threshold)
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_pred_t, zero_division=0)
        prec = precision_score(y_true, y_pred_t, zero_division=0)

        if rec >= target_recall:
            f1_t = f1_score(y_true, y_pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t
                best_precision = prec
                best_recall = rec

    return best_threshold, best_precision, best_recall


def print_results_table(results: list) -> None:
    """Print a formatted comparison table of model results."""
    header = f"\n{'Model':<22} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUPRC':>8} {'Net Value €':>12}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: x["auprc"], reverse=True):
        print(
            f"{r['model']:<22} "
            f"{r['precision']:>10.3f} "
            f"{r['recall']:>10.3f} "
            f"{r['f1']:>8.3f} "
            f"{r['auprc']:>8.3f} "
            f"{r['net_value_eur']:>12,.0f}"
        )

    print("\nKey: AUPRC = Area Under Precision-Recall Curve (higher = better)")
    print("Net Value = (Fraud caught × avg €200) - (Missed fraud × €200) - (False alerts × €2)")
