"""
Banking Fraud ML Detection — Main Entry Point

Usage:
    python main.py                    # Run full pipeline (50k transactions)
    python main.py --quick            # Quick demo (10k transactions)
    python main.py --transactions N   # Custom transaction count
    python main.py --recall 0.90      # Target 90% recall
"""

import argparse
import sys
import os

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Banking Fraud ML Detection — Portfolio Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Full pipeline (50k transactions)
  python main.py --quick          # Fast demo (10k transactions)
  python main.py --recall 0.90    # Target 90% recall threshold
  python main.py --seed 123       # Different random seed
        """,
    )
    parser.add_argument(
        "--transactions",
        type=int,
        default=50000,
        help="Number of synthetic transactions to generate (default: 50000)",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.005,
        help="Fraction of fraudulent transactions (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--recall",
        type=float,
        default=0.80,
        help="Target minimum recall for threshold optimization (default: 0.80)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick demo with 10k transactions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.quick:
        args.transactions = 10000

    results = run_pipeline(
        n_transactions=args.transactions,
        fraud_rate=args.fraud_rate,
        target_recall=args.recall,
        random_seed=args.seed,
        verbose=True,
    )

    # Exit with code 0 on success
    best = results["best_result"]
    print(f"\n✓ Best model ({results['best_model_name']}) AUPRC: {best['auprc']:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
