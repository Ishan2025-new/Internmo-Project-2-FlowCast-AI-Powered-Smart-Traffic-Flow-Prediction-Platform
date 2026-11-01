from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .evaluate import split_chronologically
except ImportError:
    from evaluate import split_chronologically


LOOKBACK = 8


class FlowCastLSTM(nn.Module):
    """Multi-task LSTM with traffic-volume and congestion heads."""

    def __init__(self, feature_count: int, congestion_classes: int = 4, dropout: float = 0.2) -> None:
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if congestion_classes <= 0:
            raise ValueError("congestion_classes must be positive")

        super().__init__()
        self.lstm_1 = nn.LSTM(feature_count, 64, batch_first=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.lstm_2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout_2 = nn.Dropout(dropout)
        self.volume_head = nn.Linear(32, 1)
        self.congestion_head = nn.Linear(32, congestion_classes)

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.lstm_1(sequences)
        hidden = self.dropout_1(hidden)
        _, (hidden, _) = self.lstm_2(hidden)
        hidden = self.dropout_2(hidden[-1])
        volume = self.volume_head(hidden).squeeze(-1)
        congestion_logits = self.congestion_head(hidden)
        return volume, congestion_logits


def select_sequence_features(df: pd.DataFrame) -> list[str]:
    excluded = {
        "timestamp", "date", "time", "align_hour", "traffic_volume",
        "congestion_level", "accident_count", "vc_ratio",
        "road_name", "vehicle_type_dist", "weather_condition",
        "holiday_name", "event_name",
    }
    numeric = df.select_dtypes(include=[np.number]).columns
    return [column for column in numeric if column not in excluded]


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    imputer: SimpleImputer,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    congestion_encoder: LabelEncoder,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    for _, road_df in df.groupby("road_id", sort=False):
        road_df = road_df.sort_values("timestamp")
        expected = pd.date_range(
            road_df["timestamp"].min(), road_df["timestamp"].max(), freq="30min"
        )
        if len(road_df) != len(expected):
            road_df = road_df.drop_duplicates("timestamp", keep="last")

        features = road_df[feature_columns].replace([np.inf, -np.inf], np.nan)
        features = imputer.transform(features)
        features = feature_scaler.transform(features).astype(np.float32)

        volumes = road_df["traffic_volume"].to_numpy(dtype=np.float32)
        scaled_volumes = target_scaler.transform(volumes.reshape(-1, 1)).ravel().astype(np.float32)
        congestion = congestion_encoder.transform(road_df["congestion_level"])

        for index in range(lookback, len(road_df)):
            previous_timestamp = road_df["timestamp"].iloc[index - 1]
            current_timestamp = road_df["timestamp"].iloc[index]
            if current_timestamp - previous_timestamp != pd.Timedelta(minutes=30):
                continue

            frames.append((features[index - lookback:index], scaled_volumes[index], congestion[index]))

    if not frames:
        raise ValueError("No sequences could be created; check timestamps and lookback")

    return (
        np.stack([frame[0] for frame in frames]),
        np.asarray([frame[1] for frame in frames], dtype=np.float32),
        np.asarray([frame[2] for frame in frames], dtype=np.int64),
    )


def prepare_sequences(
    data_path: Path = Path("data/processed/northline_corridor_processed.parquet"),
    output_path: Path = Path("models/flowcast_sequences.npz"),
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    df = pd.read_parquet(data_path)
    train, validation, test = split_chronologically(df)
    feature_columns = select_sequence_features(df)
    if not feature_columns:
        raise ValueError("No numeric input features were found")

    imputer = SimpleImputer(strategy="median")
    train_features = imputer.fit_transform(train[feature_columns].replace([np.inf, -np.inf], np.nan))
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_features)

    volumes = train["traffic_volume"].to_numpy(dtype=np.float32).reshape(-1, 1)
    target_scaler = MinMaxScaler()
    target_scaler.fit(volumes)

    congestion_encoder = LabelEncoder()
    congestion_encoder.fit(train["congestion_level"].dropna())

    datasets = {}
    for split_name, split in (("train", train), ("validation", validation), ("test", test)):
        sequences = build_sequences(
            split,
            feature_columns,
            imputer,
            feature_scaler,
            target_scaler,
            congestion_encoder,
        )
        datasets[split_name] = sequences
        print(f"{split_name}: X={sequences[0].shape}, y_volume={sequences[1].shape}, y_congestion={sequences[2].shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=datasets["train"][0], y_volume_train=datasets["train"][1], y_congestion_train=datasets["train"][2],
        X_validation=datasets["validation"][0], y_volume_validation=datasets["validation"][1], y_congestion_validation=datasets["validation"][2],
        X_test=datasets["test"][0], y_volume_test=datasets["test"][1], y_congestion_test=datasets["test"][2],
    )
    joblib.dump(feature_columns, output_path.parent / "sequence_feature_columns.joblib")
    joblib.dump(feature_scaler, output_path.parent / "sequence_feature_scaler.joblib")
    joblib.dump(target_scaler, output_path.parent / "sequence_target_scaler.joblib")
    joblib.dump(congestion_encoder, output_path.parent / "sequence_congestion_encoder.joblib")
    joblib.dump(imputer, output_path.parent / "sequence_feature_imputer.joblib")
    return datasets


def multitask_loss(
    volume_predictions: torch.Tensor,
    congestion_logits: torch.Tensor,
    volume_targets: torch.Tensor,
    congestion_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    volume_loss = nn.functional.mse_loss(volume_predictions, volume_targets)
    congestion_loss = nn.functional.cross_entropy(congestion_logits, congestion_targets)
    return volume_loss + 0.5 * congestion_loss, volume_loss, congestion_loss


def train_model(
    sequence_path: Path = Path("models/flowcast_sequences.npz"),
    output_path: Path = Path("models/flowcast_lstm.pt"),
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    patience: int = 10,
    max_train_rows: int | None = None,
) -> FlowCastLSTM:
    data = np.load(sequence_path)
    X_train = data["X_train"].astype(np.float32)
    y_volume_train = data["y_volume_train"].astype(np.float32)
    y_congestion_train = data["y_congestion_train"].astype(np.int64)
    X_validation = data["X_validation"].astype(np.float32)
    y_volume_validation = data["y_volume_validation"].astype(np.float32)
    y_congestion_validation = data["y_congestion_validation"].astype(np.int64)

    if max_train_rows is not None:
        X_train = X_train[:max_train_rows]
        y_volume_train = y_volume_train[:max_train_rows]
        y_congestion_train = y_congestion_train[:max_train_rows]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_volume_train),
            torch.from_numpy(y_congestion_train),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_validation),
            torch.from_numpy(y_volume_validation),
            torch.from_numpy(y_congestion_validation),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    model = FlowCastLSTM(feature_count=X_train.shape[2], congestion_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, volume_batch, congestion_batch in train_loader:
            X_batch = X_batch.to(device)
            volume_batch = volume_batch.to(device)
            congestion_batch = congestion_batch.to(device)

            optimizer.zero_grad()
            volume_predictions, congestion_logits = model(X_batch)
            loss, _, _ = multitask_loss(
                volume_predictions, congestion_logits, volume_batch, congestion_batch
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for X_batch, volume_batch, congestion_batch in validation_loader:
                volume_predictions, congestion_logits = model(X_batch.to(device))
                loss, _, _ = multitask_loss(
                    volume_predictions,
                    congestion_logits,
                    volume_batch.to(device),
                    congestion_batch.to(device),
                )
                validation_losses.append(loss.item())

        validation_loss = float(np.mean(validation_losses))
        scheduler.step(validation_loss)
        print(f"Epoch {epoch}/{epochs} - validation loss: {validation_loss:.6f}")

        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    model.load_state_dict(best_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "feature_count": X_train.shape[2],
        "congestion_classes": 4,
        "best_validation_loss": best_loss,
    }, output_path)
    print(f"Saved best model to {output_path}")
    return model


if __name__ == "__main__":
    train_model(max_train_rows=50000)
