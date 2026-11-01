from pathlib import Path

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "road_id",
        "timestamp",
        "traffic_volume",
        "avg_speed",
        "road_capacity",
        "rainfall",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns for feature engineering: {sorted(missing_columns)}"
        )

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result = result.sort_values(["road_id", "timestamp"], kind="stable").reset_index(drop=True)

    hour = result["timestamp"].dt.hour
    day_of_week = result["timestamp"].dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    grouped = result.groupby("road_id", sort=False)
    for column, feature_prefix in (
        ("traffic_volume", "vol"),
        ("avg_speed", "speed"),
    ):
        for lag in (1, 2):
            result[f"{feature_prefix}_lag_{lag}"] = grouped[column].shift(lag)
        if column == "traffic_volume":
            result["vol_lag_48"] = grouped[column].shift(48)

    for window in (4, 8):
        result[f"vol_roll_mean_{window}"] = grouped["traffic_volume"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
        result[f"vol_roll_std_{window}"] = grouped["traffic_volume"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=2).std()
        )

    capacity_per_window = result["road_capacity"] / 2
    result["vc_ratio"] = result["traffic_volume"].div(
        capacity_per_window.where(capacity_per_window.ne(0))
    )

    minute_of_day = hour * 60 + result["timestamp"].dt.minute
    morning_peak = minute_of_day.between(7 * 60, 9 * 60 + 30, inclusive="both")
    evening_peak = minute_of_day.between(16 * 60 + 30, 19 * 60, inclusive="both")
    result["is_peak_hour"] = (morning_peak | evening_peak).astype("int8")
    result["rain_impact"] = result["rainfall"] * result["is_peak_hour"]

    return result


def run_feature_engineering(
    input_path: Path = Path("data/interim/traffic_cleaned.parquet"),
    output_path: Path = Path("data/processed/northline_corridor_processed.parquet"),
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    featured = engineer_features(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(output_path, index=False)
    return featured


if __name__ == "__main__":
    run_feature_engineering()
    print("Feature engineering completed.")
