from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import sys
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

try:
    from .dl_model import FlowCastLSTM
    from .evaluate import split_chronologically
except ImportError:
    from dl_model import FlowCastLSTM
    from evaluate import split_chronologically


def protected_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def traffic_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": protected_mape(y_true, y_pred),
    }


def evaluate_lstm(
    model_path: Path = Path("models/flowcast_lstm.pt"),
    sequence_path: Path = Path("models/flowcast_sequences.npz"),
    model_dir: Path = Path("models"),
) -> tuple[dict[str, float], dict[str, float]]:
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {model_path}")

    sequences = np.load(sequence_path)
    X_test = torch.from_numpy(sequences["X_test"].astype(np.float32))
    y_volume_scaled = sequences["y_volume_test"].astype(np.float32)
    y_congestion = sequences["y_congestion_test"].astype(np.int64)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = FlowCastLSTM(
        feature_count=int(checkpoint["feature_count"]),
        congestion_classes=int(checkpoint["congestion_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    logits = []
    with torch.no_grad():
        for start in range(0, len(X_test), 256):
            volume_pred, congestion_logits = model(X_test[start:start + 256])
            predictions.append(volume_pred.numpy())
            logits.append(congestion_logits.numpy())

    volume_scaled_pred = np.concatenate(predictions)
    congestion_probabilities = np.concatenate(logits)
    congestion_predictions = congestion_probabilities.argmax(axis=1)

    volume_scaler = joblib.load(model_dir / "sequence_target_scaler.joblib")
    y_volume = volume_scaler.inverse_transform(y_volume_scaled.reshape(-1, 1)).ravel()
    volume_pred = volume_scaler.inverse_transform(volume_scaled_pred.reshape(-1, 1)).ravel()

    volume_metrics = traffic_regression_metrics(y_volume, volume_pred)
    congestion_metrics = {
        "Macro-F1": float(f1_score(y_congestion, congestion_predictions, average="macro", zero_division=0))
    }
    return volume_metrics, congestion_metrics


def evaluate_classical_models(
    data_path: Path = Path("data/processed/northline_corridor_processed.parquet"),
    model_dir: Path = Path("models"),
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    df = pd.read_parquet(data_path)
    _, _, test = split_chronologically(df)
    volume_results = {}

    traffic_features = joblib.load(model_dir / "traffic_volume_features.joblib")
    from .ml_models import LinearRegressionGD
    sys.modules["__main__"].LinearRegressionGD = LinearRegressionGD
    custom_model = joblib.load(model_dir / "traffic_volume_custom.joblib")
    custom_imputer = joblib.load(model_dir / "traffic_volume_custom_imputer.joblib")
    custom_scaler = joblib.load(model_dir / "traffic_volume_custom_scaler.joblib")
    custom_test_features = test[traffic_features].replace([np.inf, -np.inf], np.nan)
    custom_test_features = custom_scaler.transform(custom_imputer.transform(custom_test_features))
    custom_prediction = custom_model.predict(custom_test_features)
    volume_results["linear_regression"] = traffic_regression_metrics(
        test["traffic_volume"].to_numpy(), custom_prediction
    )

    for name in ("decision_tree", "random_forest", "xgboost"):
        model = joblib.load(model_dir / f"traffic_volume_{name}.joblib")
        prediction = model.predict(test[traffic_features])
        volume_results[name] = traffic_regression_metrics(test["traffic_volume"].to_numpy(), prediction)

    accident_features = joblib.load(model_dir / "accident_count_features.joblib")
    accident_model = joblib.load(model_dir / "accident_count_xgboost.joblib")
    accident_true = test["accident_count"].fillna(0).astype(float).ge(1).astype(int)
    if hasattr(accident_model, "predict_proba"):
        accident_probability = accident_model.predict_proba(test[accident_features])[:, 1]
        accident_auc = float(roc_auc_score(accident_true, accident_probability))
    else:
        accident_auc = float("nan")

    congestion_features = joblib.load(model_dir / "congestion_level_features.joblib")
    congestion_encoder = joblib.load(model_dir / "congestion_level_encoder.joblib")
    congestion_model = joblib.load(model_dir / "congestion_level_xgboost.joblib")
    congestion_true = congestion_encoder.transform(test["congestion_level"])
    congestion_prediction = congestion_model.predict(test[congestion_features])
    congestion_metrics = {
        "Macro-F1": float(f1_score(congestion_true, congestion_prediction, average="macro", zero_division=0))
    }

    return volume_results, congestion_metrics, {"ROC-AUC": accident_auc}


def run_benchmark() -> pd.DataFrame:
    volume_models, congestion_xgb, accident_xgb = evaluate_classical_models()
    lstm_volume, lstm_congestion = evaluate_lstm()

    rows = []
    for name, metrics in volume_models.items():
        rows.append({"Target": "Traffic Volume", "Model": name, **metrics})
    rows.append({"Target": "Traffic Volume", "Model": "LSTM", **lstm_volume})
    rows.append({"Target": "Congestion Level", "Model": "XGBoost", **congestion_xgb})
    rows.append({"Target": "Congestion Level", "Model": "LSTM", **lstm_congestion})
    rows.append({"Target": "Accident Risk", "Model": "XGBoost", **accident_xgb})

    benchmark = pd.DataFrame(rows)
    output_path = Path("reports/model_benchmark.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_path, index=False)
    print(benchmark.to_string(index=False))
    print(f"Saved benchmark to {output_path}")
    return benchmark


if __name__ == "__main__":
    run_benchmark()
