# Sample Dataset

`transactions_sample.csv` — 200 synthetically generated banking transactions (5% fraud rate for demo visibility).

## Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique transaction identifier |
| `customer_id` | string | Customer account identifier |
| `amount` | float | Transaction amount in EUR |
| `merchant_category` | string | Merchant type (grocery, electronics, crypto_exchange, etc.) |
| `merchant_risk_score` | float | Category-level fraud risk (0 = low, 1 = high) |
| `country` | string | 2-letter ISO country code |
| `is_foreign_country` | int | 1 if transaction country ≠ customer's home country |
| `card_type` | string | Card network and type |
| `hour_of_day` | int | Hour of transaction (0-23) |
| `day_of_week` | int | Day of week (0=Monday, 6=Sunday) |
| `customer_credit_score` | int | Simulated credit score (300-850) |
| `customer_avg_amount` | float | Customer's typical transaction amount |
| `is_fraud` | int | Ground truth label (1 = fraud, 0 = legitimate) |

## Notes

- This is **synthetic data** — no real customer information
- Full dataset: use `python main.py` to generate 50,000 transactions
- Fraud patterns are realistic: night-time timing, high-risk merchants, foreign countries, unusual amounts
