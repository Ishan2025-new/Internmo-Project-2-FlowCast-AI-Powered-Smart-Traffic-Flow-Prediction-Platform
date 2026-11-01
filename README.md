# FlowCast — Smart Traffic Flow Prediction

FlowCast is an end-to-end AI-powered traffic forecasting platform for the 25-road-segment Northline Corridor. It collects and validates traffic, weather, and calendar data, cleans and merges the information, and creates chronological features for machine-learning and deep-learning models.

The project processes 30-minute road telemetry to forecast traffic volume, classify congestion as Free-flow, Moderate, Heavy, or Severe, estimate accident risk, and present model results through an interactive Streamlit dashboard. Classical models provide interpretable baselines, while a multi-task PyTorch LSTM learns temporal patterns across the previous eight 30-minute observations to predict the next traffic window.

## Project description

### Data and feature engineering

Raw sensor, weather, and calendar records are validated against their expected schemas before cleaning. FlowCast removes duplicate road/timestamp records, handles invalid values, imputes short and long gaps, aligns hourly weather observations to 30-minute traffic windows, expands vehicle-distribution data, and derives congestion categories.

Feature engineering creates cyclic time values, 30-minute and hourly lag features, 24-hour traffic history, rolling statistics, peak-hour indicators, rainfall interactions, and road capacity ratios. The final dataset preserves chronological order within each road segment to reduce target leakage.

### Machine-learning pipeline

FlowCast uses strict chronological train, validation, and test periods covering January through May 2025. Preprocessing is fitted only on training data. The project includes a NumPy gradient-descent linear regression implementation and classical regression and classification models based on decision trees, random forests, XGBoost, logistic regression, and support vector machines.

### Deep-learning pipeline

The LSTM module scales features with training-fitted parameters and builds separate sequences for each road segment. Each input contains 8 observations, representing a four-hour lookback. A two-layer LSTM with 64 and 32 units produces both a scaled traffic-volume forecast and a four-class congestion prediction.

### Dashboard and operational use

The Streamlit dashboard lets operators filter road segments and dates, review live indicators, explore historical trends and weather effects, compare road performance, inspect congestion patterns, view model benchmarks, and validate uploaded sensor CSV files. It also displays forecast uncertainty, residuals, and model feature importance.

### Project outputs

The pipeline produces cleaned and feature-engineered datasets, trained model artifacts, sequence tensors, benchmark results, a data-quality report, and the trained dashboard application. The project was designed to support reproducible training, comparison across model families, and integration with future real-time traffic monitoring systems.

## Project features

- Validates and cleans traffic, weather, and calendar data
- Creates temporal, lag, rolling, peak-hour, weather, and congestion features
- Splits data chronologically without using random cross-validation
- Trains classical models with NumPy, scikit-learn, and XGBoost
- Trains a PyTorch multi-task LSTM for volume and congestion forecasts
- Benchmarks all models on the held-out May 2025 test period
- Provides an interactive operational dashboard with 10 views

## Project structure

```text
flowcast/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── dashboard/
│   └── app.py
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── features.py
│   ├── evaluate.py
│   ├── ml_models.py
│   ├── dl_model.py
│   ├── benchmark.py
│   ├── quality_report.py
│   └── utils.py
├── config.yaml
├── requirements.txt
└── run_flowcast.py
```

## Requirements

- Windows 10/11
- Python 3.11 recommended
- At least 8 GB RAM
- Disk space for processed datasets and trained models

## Installation

From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If script execution is disabled in PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\activate.bat
```

## Run the complete pipeline

Run every processing, training, and benchmarking stage with one command:

```powershell
python run_flowcast.py
```

The complete run starts with random seed `42` for reproducibility and may take several minutes depending on the available CPU and training configuration.

## Run stages individually

```powershell
python -m src.clean
python -m src.features
python -m src.ml_models
python -m src.dl_model
python -m src.benchmark
python -m src.quality_report
```

To prepare sequence tensors without running the full deep-learning training loop:

```powershell
python -c "from src.dl_model import prepare_sequences; prepare_sequences()"
```

## Launch the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.headless=true
```

Open [http://localhost:8501](http://localhost:8501) in a browser. The dashboard is run separately from the training pipeline.

## Dashboard views

1. Live prediction
2. Historical traffic trends
3. Congestion heatmap
4. Road comparison
5. Model performance scoreboard
6. Feature importance
7. Forecast overlay
8. Prediction confidence and uncertainty
9. Weather and traffic analytics
10. Sensor CSV upload validation

## Important outputs

| Output | Description |
|---|---|
| `data/interim/traffic_cleaned.parquet` | Validated, deduplicated, quarantined, imputed, and merged traffic data |
| `data/processed/northline_corridor_processed.parquet` | Chronological feature-engineered model dataset |
| `models/flowcast_sequences.npz` | Train, validation, and test LSTM sequences |
| `models/flowcast_lstm.pt` | Trained multi-task PyTorch LSTM checkpoint |
| `models/*.joblib` | Classical models, encoders, scalers, and feature lists |
| `reports/model_benchmark.csv` | Test-set model comparison |
| `reports/data_quality.md` | Data cleaning and quality summary |

## Data split

FlowCast uses chronological splits to prevent future observations leaking into training:

- Training: January 1–April 15, 2025
- Validation: April 16–April 30, 2025
- Test: May 1–May 31, 2025

Scalers and imputers are fitted on the training data only and are reused for validation and test data.

## Troubleshooting

### Missing parquet engine

```powershell
python -m pip install pyarrow
```

### Missing XGBoost

```powershell
python -m pip install xgboost
```

### Streamlit cannot be found

Ensure the project virtual environment is activated, then install the requirements again:

```powershell
python -m pip install -r requirements.txt
```

## Project status

The implementation guide is complete. Model benchmarking is included for acceptance checks and should be rerun after model tuning.
