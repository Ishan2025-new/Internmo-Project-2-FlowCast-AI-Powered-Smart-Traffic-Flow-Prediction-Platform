from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


TRAIN_START = pd.Timestamp("2025-01-01")
VALIDATION_START = pd.Timestamp("2025-04-16")
TEST_START = pd.Timestamp("2025-05-01")
DATA_END = pd.Timestamp("2025-06-01")


def split_chronologically(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the 2025 dataset into train, validation, and test date ranges."""
    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    timestamps = pd.to_datetime(df[timestamp_column], errors="raise")
    if timestamps.isna().any():
        raise ValueError(f"{timestamp_column} contains missing timestamps")

    in_range = timestamps.ge(TRAIN_START) & timestamps.lt(DATA_END)
    if not in_range.all():
        raise ValueError(
            "All rows must fall between 2025-01-01 and 2025-05-31 "
            "to use the prescribed chronological split"
        )

    train_mask = timestamps.lt(VALIDATION_START)
    validation_mask = timestamps.ge(VALIDATION_START) & timestamps.lt(TEST_START)
    test_mask = timestamps.ge(TEST_START)

    train = df.loc[train_mask].copy()
    validation = df.loc[validation_mask].copy()
    test = df.loc[test_mask].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "The dataset must contain rows in the train, validation, and test periods"
        )

    return tuple(
        split.sort_values(timestamp_column, kind="stable")
        for split in (train, validation, test)
    )


def scale_feature_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler | MinMaxScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler | MinMaxScaler]:
    """Fit a scaler on training features only, then transform all three splits."""
    if not feature_columns:
        raise ValueError("feature_columns must contain at least one column")

    for split_name, split in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        missing = set(feature_columns).difference(split.columns)
        if missing:
            raise ValueError(
                f"{split_name} split is missing features: {sorted(missing)}"
            )

    fitted_scaler = scaler if scaler is not None else StandardScaler()
    train_scaled = fitted_scaler.fit_transform(train[feature_columns])
    validation_scaled = fitted_scaler.transform(validation[feature_columns])
    test_scaled = fitted_scaler.transform(test[feature_columns])

    def to_frame(values, source: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(values, columns=feature_columns, index=source.index)

    return (
        to_frame(train_scaled, train),
        to_frame(validation_scaled, validation),
        to_frame(test_scaled, test),
        fitted_scaler,
    )


def main() -> None:
    data_path = Path("data/processed/northline_corridor_processed.parquet")
    df = pd.read_parquet(data_path)
    train, validation, test = split_chronologically(df)

    print(f"Train:      {len(train):,} rows")
    print(f"Validation: {len(validation):,} rows")
    print(f"Test:       {len(test):,} rows")


if __name__ == "__main__":
    main()
