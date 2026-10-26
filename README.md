# Retail Intelligence - Demand Forecasting & Inventory Optimization
(README a completer : architecture, resultats, limites, lancement.)
# Retail Intelligence — Sales Forecasting

## Overview

Retail Intelligence is an end-to-end data analytics and machine learning project designed to analyze retail sales and forecast product-family sales across multiple stores.

The project combines SQL, DuckDB, Python, scikit-learn, and Streamlit to build a reproducible workflow from data preparation to forecasting and dashboard visualization.

## Objectives

- Explore retail sales trends across stores and product families.
- Build analytical datasets using SQL and DuckDB.
- Engineer time-based features for sales forecasting.
- Train and evaluate a machine learning regression model.
- Compare model performance with a simple historical baseline.
- Visualize historical indicators and forecasts in an interactive dashboard.

## Tech Stack

- **Python** — data processing and machine learning
- **Pandas** — data manipulation
- **DuckDB and SQL** — data storage, transformations, and feature engineering
- **scikit-learn** — model training and evaluation
- **Streamlit** — interactive dashboard
- **Joblib** — model serialization

## Project Structure

```text
retail-intelligence/
├── api/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── sql/
├── src/
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local datasets, the DuckDB database, and generated CSV files are excluded from version control.

## Data Pipeline

1. Load the raw retail datasets.
2. Clean and transform the data using SQL and DuckDB.
3. Create time-based features, including sales lags and rolling statistics.
4. Train a regression model using historical sales data.
5. Evaluate forecasts on a held-out 14-day historical period.
6. Generate an illustrative 14-day forecast.
7. Display results in the Streamlit dashboard.

## Model Evaluation

The model was evaluated on a historical 14-day test period.

| Metric | Machine Learning Model | Baseline |
|---|---:|---:|
| Mean Absolute Error (MAE) | 81.88 | 98.23 |
| Weighted Absolute Percentage Error (WAPE) | 17.85% | 21.41% |

The model reduced MAE by approximately **16.64%** compared with the baseline, which uses sales from 14 days earlier.

These results apply to the historical evaluation period and its available input features; they do not guarantee future performance.

## Future Forecasts

The project generates forecasts for a 14-day horizon. The current illustrative scenario assumes no future promotions because future promotion information is unavailable.

These predictions are estimates, not observed sales. Their accuracy cannot be verified without actual sales for the forecast period.

## Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd retail-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
python -m streamlit run app.py
```

The raw datasets and local database must be available in the expected paths before running the complete pipeline.

## Limitations

- Future forecasts depend on assumptions about unknown future inputs.
- The current no-promotion scenario may differ from actual retail conditions.
- Historical evaluation results may not generalize to other periods.
- Data and model versions should be documented for reproducible results.

## Author

**Khalid Umar Shabazz**

Master's student in Mathematics, Data Analysis and Machine Learning (MADA).

Interested in data analytics, machine learning, and applied statistics.
