# HomeScope — House Price Prediction

A clean, production-structured regression project that predicts median housing prices using Linear, Ridge, and Lasso regression on a Boston-style housing dataset.

---

## Project Structure

```
HomeScope/
├── data/
│   └── housing.csv               # Boston-style housing dataset (506 rows, 13 features)
├── models/
│   └── linear_regression_model.pkl  # Saved model + scaler (joblib)
├── notebooks/
│   └── HomeScope.ipynb           # Exploratory analysis notebook
├── src/
│   ├── data_preprocessing.py     # Load, clean, split, scale
│   ├── train_model.py            # Train, cross-validate, save/load model
│   └── evaluate_model.py         # Metrics + 3 evaluation plots
├── reports/
│   ├── metrics.txt               # MAE, MSE, RMSE, R² report
│   ├── predictions_vs_actual.png # Scatter: actual vs predicted
│   ├── residuals_plot.png        # Residual analysis (2 panels)
│   └── feature_importance.png    # Coefficient bar chart
├── main.py                       # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## Features

| Column   | Description |
|----------|-------------|
| CRIM     | Per capita crime rate |
| ZN       | Residential land zoning proportion |
| INDUS    | Non-retail business acres |
| CHAS     | Charles River dummy (1 = bounds river) |
| NOX      | Nitric oxide concentration |
| RM       | Average rooms per dwelling |
| AGE      | Proportion of pre-1940 units |
| DIS      | Distance to employment centres |
| RAD      | Highway accessibility index |
| TAX      | Property-tax rate per $10,000 |
| PTRATIO  | Pupil-teacher ratio |
| B        | Proportion of Black residents (legacy feature) |
| LSTAT    | % lower status population |
| **MEDV** | **Target — median home value ($1000s)** |

---

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

---

## Pipeline Steps

1. **Load & Validate** — checks file existence, missing columns
2. **Clean** — removes duplicates, median-imputes missing values
3. **Split** — 80/20 train/test split on raw text (before scaling)
4. **Scale** — StandardScaler fit on train only (no leakage)
5. **Model Selection** — 5-fold CV across Linear, Ridge, Lasso
6. **Save** — model + scaler persisted together in one `.pkl`
7. **Evaluate** — MAE, RMSE, R² + 3 plots saved to `reports/`
8. **Predict** — custom house input with trained model

---

## Reports Generated

- `reports/metrics.txt` — numeric evaluation summary
- `reports/predictions_vs_actual.png` — how close predictions are to truth
- `reports/residuals_plot.png` — residual distribution & heteroscedasticity check
- `reports/feature_importance.png` — which features most influence price