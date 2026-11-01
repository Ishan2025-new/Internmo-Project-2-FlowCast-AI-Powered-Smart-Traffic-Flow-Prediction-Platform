import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(raw_dir: Path):
    traffic = pd.read_csv(raw_dir / "traffic_sensor_log.csv")
    weather = pd.read_csv(raw_dir / "weather_observations.csv")
    calendar = pd.read_csv(raw_dir / "calendar_events.csv")

    traffic["date"] = pd.to_datetime(traffic["date"])
    traffic["time"] = pd.to_datetime(traffic["time"], format="%H:%M").dt.strftime("%H:%M")
    traffic["timestamp"] = pd.to_datetime(
        traffic["date"].dt.strftime("%Y-%m-%d") + " " + traffic["time"],
        format="%Y-%m-%d %H:%M",
    )

    weather["date"] = pd.to_datetime(weather["date"], format="%d/%m/%Y")
    calendar["date"] = pd.to_datetime(calendar["date"], format="%Y-%m-%d")

    return traffic, weather, calendar


def deduplicate_traffic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["null_count"] = df.isna().sum(axis=1)

    # keep row with minimum nulls within duplicated (road_id, timestamp)
    df = (
        df.sort_values(["road_id", "timestamp", "null_count"])
        .drop_duplicates(subset=["road_id", "timestamp"], keep="first")
        .drop(columns=["null_count"])
    )
    return df


def quarantine_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Traffic volume negative
    df.loc[df["traffic_volume"] < 0, "traffic_volume"] = np.nan

    # Avg speed constraints
    df.loc[(df["avg_speed"] < 0) | (df["avg_speed"] > 200), "avg_speed"] = np.nan

    # Occupancy constraints
    df.loc[(df["occupancy"] < 0) | (df["occupancy"] > 100), "occupancy"] = np.nan

    # Z-score anomalies by road_id and hour
    df["hour"] = df["timestamp"].dt.hour
    traffic_group = df.groupby(["road_id", "hour"])["traffic_volume"]
    group_mean = traffic_group.transform("mean")
    group_std = traffic_group.transform("std", ddof=0)
    df["traffic_volume_z"] = (df["traffic_volume"] - group_mean).div(
        group_std.where(group_std.gt(0))
    )
    anomalies = df["traffic_volume_z"].abs().gt(4.0).fillna(False)
    df.loc[anomalies, "traffic_volume"] = np.nan

    df = df.drop(columns=["hour", "traffic_volume_z"], errors="ignore")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # sort chronologically per road
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    # short gaps: interpolate per road_id using each row's timestamp
    numeric_cols = ["traffic_volume", "avg_speed", "occupancy", "travel_time", "signal_timing", "road_capacity"]
    for col in numeric_cols:
        if col in df.columns:
            for road_positions in df.groupby("road_id", sort=False).indices.values():
                timestamps = pd.DatetimeIndex(df.loc[road_positions, "timestamp"])
                values = df.loc[road_positions, col].copy()
                values.index = timestamps
                df.loc[road_positions, col] = values.interpolate(
                    method="time", limit_direction="both"
                ).to_numpy()

    # longer gaps: median by (road_id, day_of_week, time)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["time_of_day"] = df["timestamp"].dt.strftime("%H:%M")

    for col in numeric_cols:
        if col in df.columns:
            medians = df.groupby(["road_id", "day_of_week", "time_of_day"])[col].transform("median")
            df[col] = df[col].fillna(medians)

    # weather fill per station
    for col in ["temperature", "visibility"]:
        if col in df.columns:
            df[col] = df.groupby("weather_station_id")[col].transform(lambda s: s.ffill().bfill())

    return df


def derive_congestion_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vc_ratio"] = df["traffic_volume"] / (df["road_capacity"] / 2)

    def congestion_rule(v):
        if pd.isna(v):
            return np.nan
        if v < 0.50:
            return "Free-flow"
        elif v < 0.80:
            return "Moderate"
        elif v < 1.00:
            return "Heavy"
        else:
            return "Severe"

    derived_congestion = df["vc_ratio"].apply(congestion_rule)
    df["congestion_level"] = df["congestion_level"].fillna(derived_congestion)
    return df


def align_weather_to_traffic(traffic: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather["align_hour"] = weather["date"].dt.strftime("%Y-%m-%d") + " " + weather["time"]
    weather["align_hour"] = pd.to_datetime(weather["align_hour"])

    # map hourly weather to 30-min traffic windows
    traffic = traffic.copy()
    traffic["align_hour"] = traffic["timestamp"].dt.floor("h")

    merged = traffic.merge(
        weather[["station_id", "align_hour", "temperature", "rainfall", "visibility", "weather_condition"]],
        how="left",
        left_on=["weather_station_id", "align_hour"],
        right_on=["station_id", "align_hour"]
    )

    return merged.drop(columns=["station_id"], errors="ignore")


def attach_calendar(traffic: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    merged = traffic.merge(calendar, on="date", how="left")
    return merged


def expand_vehicle_distribution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed_rows = []

    for _, row in df.iterrows():
        raw = row.get("vehicle_type_dist", "")
        try:
            vals = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(vals, dict):
                parsed = {
                    "veh_dist_2w": vals.get("2w", 0),
                    "veh_dist_car": vals.get("car", 0),
                    "veh_dist_lcv": vals.get("lcv", 0),
                    "veh_dist_hcv": vals.get("hcv", 0),
                }
            else:
                parsed = {
                    "veh_dist_2w": 0,
                    "veh_dist_car": 0,
                    "veh_dist_lcv": 0,
                    "veh_dist_hcv": 0,
                }
        except Exception:
            parsed = {
                "veh_dist_2w": 0,
                "veh_dist_car": 0,
                "veh_dist_lcv": 0,
                "veh_dist_hcv": 0,
            }

        parsed_rows.append(parsed)

    expanded = pd.DataFrame(parsed_rows, index=df.index)
    df = pd.concat([df, expanded], axis=1)
    return df


def run_cleaning_pipeline(raw_dir: Path, output_dir: Path):
    traffic, weather, calendar = load_data(raw_dir)

    traffic = deduplicate_traffic(traffic)
    traffic = quarantine_outliers(traffic)
    traffic = impute_missing_values(traffic)
    traffic = derive_congestion_level(traffic)
    traffic = align_weather_to_traffic(traffic, weather)
    traffic = attach_calendar(traffic, calendar)
    traffic = expand_vehicle_distribution(traffic)

    output_dir.mkdir(parents=True, exist_ok=True)
    traffic.to_parquet(output_dir / "traffic_cleaned.parquet", index=False)
    return traffic


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    output_dir = Path("data/interim")
    run_cleaning_pipeline(raw_dir, output_dir)
    print("Cleaning pipeline completed.")