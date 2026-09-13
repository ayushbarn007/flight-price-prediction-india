# Flight Price Prediction — India (Business & Economy)

End-to-end machine learning project predicting Indian domestic flight ticket prices, built separately for Business and Economy class fares. Covers raw data cleaning, exploratory analysis, and a three-model comparison (Linear Regression, Random Forest, XGBoost) — with an emphasis on testing assumptions rather than accepting the first plausible explanation for a pattern.

**Dataset:** [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) (Kaggle, `shubhambathwal`) — Indian domestic flights scraped over 48 days (Feb–Mar 2022), covering 6 metro cities and multiple airlines.

## Dashboard

![Indian Flight Price Analysis Dashboard](assets/dashboard.png)

An interactive dashboard summarizing the dataset at a glance: 300K total flights, an overall average price of ₹20.88K (₹52.54K Business vs ₹6.57K Economy), price breakdowns by airline, stop count, and route, plus the Business-class last-minute price spike as departure approaches.

## Repository Structure

```
├── business_cleaning.ipynb    # Raw → cleaned pipeline, Business class
├── economy_cleaning.ipynb     # Raw → cleaned pipeline, Economy class
├── business_eda.ipynb         # Exploratory analysis, Business class
├── economy_eda.ipynb          # Exploratory analysis, Economy class
├── business_model.ipynb       # LR / RF / XGBoost, Business class
├── economy_model.ipynb        # LR / RF / XGBoost, Economy class
├── business_cleaned.csv       # Cleaned dataset (93,487 rows)
├── economy_cleaned.csv        # Cleaned dataset (206,772 rows)
├── assets/dashboard.png       # Dashboard screenshot
└── README.md
```

## Key Findings

- **`stop_count` is the dominant price driver in both classes** — non-stop flights sit in a fundamentally different price tier than any flight with a layover.
- **A "Kolkata is the most expensive city" finding turned out to be a statistical artifact, not a real effect.** Kolkata doesn't lead any individual stop-count segment — it only *looks* most expensive in the aggregate because it has one of the lowest shares of non-stop flights of any city. Delhi shows the same pattern in reverse. Verified this by decomposing price by stop-count segment before trusting the city-level ranking.
- **`duration_mins` behaves differently across classes:** in Business, it's a threshold effect (a jump from non-stop to any-stop, then flat/declining) — in Economy, it's a genuinely continuous, steadily-rising relationship. Confirmed with bucketed medians, not just a single correlation coefficient.
- **`days_left` has near-zero linear correlation with price (Business) but hides a real, sharp last-minute price spike** in the final few days before departure — invisible to a raw scatterplot or Pearson/Spearman correlation, only visible when plotting median price by exact day.
- **Vistara is consistently priced higher than Air India**, and this is *not* explained by route mix, flight duration, or stop count — duration and stops were tested directly as confounds and ruled out.
- **Economy's achievable model accuracy is structurally capped, not a modeling failure.** Tested directly: even for *identical* flights (same airline, route, stops, duration, time-of-day), price still varies with a standard deviation ~52% as large as the overall spread — meaning over half of Economy's price variance comes from information no available feature captures (most likely a `days_left`-equivalent booking-date signal that couldn't be reliably reconstructed for this class).

## Modeling Results

| Class | Model | R² | RMSE | RMSE (% of median price) |
|---|---|---|---|---|
| Business | Linear Regression | 0.561 | ₹8,613 | 16.19% |
| Business | Random Forest | 0.814 | ₹5,602 | 10.53% |
| Business | **XGBoost** | **0.879** | **₹4,527** | **8.51%** |
| Economy | Linear Regression | 0.219 | ₹3,305 | 57.13% |
| Economy | Random Forest | 0.317 | ₹3,091 | 53.42% |
| Economy | **XGBoost** | **0.349** | **₹3,018** | **52.16%** |

XGBoost outperforms both other models in each class. The gap between Linear Regression and the tree-based models is far larger for Business (+0.32 R²) than Economy (+0.13 R²) — because Business benefits from `days_left`'s non-linear last-minute-spike pattern, a signal tree models can exploit and Economy simply doesn't have access to.

## Technical Notes

- **Data cleaning:** regex-based extraction of stop count and layover city from a single messy `stop` field, duration parsing from free-text (`"2h 15m"` → minutes), and a verified row-order join to attach `days_left` to Business class from a separate reference dataset (not possible for Economy — the join was ~85% ambiguous, so the feature was deliberately excluded rather than guessed).
- **Feature engineering was tuned per model, not applied uniformly** — bucketed time-of-day features improve Linear Regression (avoids a false-linearity assumption on a cyclical variable) but *reduce* Random Forest's accuracy (trees split on numeric thresholds natively and lose granularity when pre-bucketed). Verified with a direct before/after comparison, not assumed.
- **Every major EDA claim was stress-tested for confounds** before being accepted — e.g. the Kolkata/Delhi finding above, and ruling out route-mix/duration as explanations for the Vistara price premium.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn

## Author

Ayush Barnwal — [GitHub](https://github.com/ayushbarn007) · [LinkedIn](https://linkedin.com/in/ayushbarnwal18)
