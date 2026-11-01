import pandas as pd
from pathlib import Path

def validate_columns(df, expected_columns, file_name):
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}")
    return True

def parse_calendar_events(path):
    df = pd.read_csv(path)
    validate_columns(df, [
        "date", "public_holiday", "holiday_name", "event_flag", "event_name", "roadwork_flag"
    ], "calendar_events.csv")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df

def parse_weather_observations(path):
    df = pd.read_csv(path)
    validate_columns(df, [
        "station_id", "date", "time", "weather_condition", "temperature", "rainfall", "visibility"
    ], "weather_observations.csv")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df['weather_condition'] = (
        df["weather_condition"].astype(str).str.strip().str.lower().replace(
            {
                "rain": "rain",
                "rainy": "rain",
                "clear": "clear",
                "cloudy": "cloudy",
                "fog": "fog"
            }
        )
    )
    return df

def parse_traffic_sensor_log(path):
    df = pd.read_csv(path)
    validate_columns(df, [
        "road_id", "road_name", "latitude", "longitude", "weather_station_id", "date", "time", "traffic_volume", 
        "vehicle_count", "vehicle_type_dist", "avg_speed", "occupancy", "congestion_level", "travel_time", "accident_count", 
        "signal_timing", "road_capacity"
    ], "traffic_sensor_log.csv")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"])
    return df

def main():
    raw_dir = Path("data/raw")
    calendar = parse_calendar_events(raw_dir / "calendar_events.csv")
    weather = parse_weather_observations(raw_dir / "weather_observations.csv")
    traffic = parse_traffic_sensor_log(raw_dir / "traffic_sensor_log.csv")
    print("Validated files successfully.")
    return calendar, weather, traffic

if __name__ == "__main__":
    main()