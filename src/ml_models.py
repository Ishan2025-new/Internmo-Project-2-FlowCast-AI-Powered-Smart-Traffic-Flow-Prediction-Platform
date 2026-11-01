from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost import XGBClassifier, XGBRegressor

try:
    from .evaluate import split_chronologically
except ImportError:
    from evaluate import split_chronologically


class LinearRegressionGD:
    """Linear regression trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be greater than zero")
        if tolerance < 0:
            raise ValueError("tolerance cannot be negative")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.loss_history_: list[float] = []
        self.n_features_in_: int | None = None

    @staticmethod
    def _validate_features(X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be a 2D array of shape (samples, features)")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values")
        return features

    @staticmethod
    def _validate_target(y: np.ndarray, n_samples: int) -> np.ndarray:
        target = np.asarray(y, dtype=float)
        if target.ndim == 2 and target.shape[1] == 1:
            target = target.ravel()
        if target.ndim != 1:
            raise ValueError("y must be a 1D array or a single-column 2D array")
        if target.shape[0] != n_samples:
            raise ValueError("X and y must contain the same number of samples")
        if not np.isfinite(target).all():
            raise ValueError("y must contain only finite values")
        return target

    @staticmethod
    def _objective(residual: np.ndarray) -> float:
        return float(np.dot(residual, residual) / (2 * residual.size))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        features = self._validate_features(X)
        target = self._validate_target(y, features.shape[0])
        n_samples, n_features = features.shape

        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.n_features_in_ = n_features
        self.loss_history_ = []

        previous_loss = np.inf
        for _ in range(self.n_iterations):
            predictions = features @ self.weights_ + self.bias_
            residual = predictions - target
            loss = self._objective(residual)
            self.loss_history_.append(loss)

            gradient_weights = (features.T @ residual) / n_samples
            gradient_bias = float(np.mean(residual))
            self.weights_ -= self.learning_rate * gradient_weights
            self.bias_ -= self.learning_rate * gradient_bias

            if abs(previous_loss - loss) <= self.tolerance:
                break
            previous_loss = loss

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None or self.n_features_in_ is None:
            raise ValueError("Call fit before predict")

        features = self._validate_features(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {features.shape[1]} features; expected {self.n_features_in_}"
            )
        return features @ self.weights_ + self.bias_

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the guide's half-MSE objective: sum((prediction - y)^2) / (2N)."""
        predictions = self.predict(X)
        target = self._validate_target(y, predictions.shape[0])
        return self._objective(predictions - target)


NON_FEATURE_COLUMNS = {
    "timestamp",
    "date",
    "time",
    "align_hour",
    "road_name",
    "vehicle_type_dist",
    "weather_condition",
    "holiday_name",
    "event_name",
    "congestion_level",
    "traffic_volume",
    "accident_count",
}


def select_features(df: pd.DataFrame, target: str, include_target: bool = False) -> list[str]:
    excluded = NON_FEATURE_COLUMNS.copy()
    excluded.discard(target)
    if include_target:
        excluded.discard(target)
    else:
        excluded.add(target)

    excluded.update({"vc_ratio"})  # Derived from traffic_volume; avoids target leakage.
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if column not in excluded]


def build_regression_models() -> dict[str, object]:
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return {
        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(max_depth=20, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", preprocessor.named_steps["scaler"]),
            ("model", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)),
        ]),
    }


def build_classification_models() -> dict[str, object]:
    scaled = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    return {
        "logistic_regression": scaled(LogisticRegression(max_iter=1000, random_state=42)),
        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=20, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)),
        ]),
        "xgboost": scaled(XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42, eval_metric="mlogloss")),
        "svm": scaled(SVC(kernel="rbf", probability=True, random_state=42)),
    }


def binary_target(accident_count: pd.Series) -> pd.Series:
    return accident_count.fillna(0).astype(float).ge(1).astype("int8")


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }


def classification_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro")),
    }


def train_model_suite(
    data_path: Path = Path("data/processed/northline_corridor_processed.parquet"),
    model_dir: Path = Path("models"),
    max_train_rows: int | None = 50000,
) -> dict[str, dict[str, object]]:
    """Train regression and classification baselines on chronological splits."""
    df = pd.read_parquet(data_path)
    train, validation, test = split_chronologically(df)
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.tail(max_train_rows).copy()

    model_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}

    traffic_features = select_features(df, "traffic_volume")
    train_volume = train[traffic_features].replace([np.inf, -np.inf], np.nan)
    validation_volume = validation[traffic_features].replace([np.inf, -np.inf], np.nan)
    y_train_volume = train["traffic_volume"]
    y_validation_volume = validation["traffic_volume"]

    custom_imputer = SimpleImputer(strategy="median")
    custom_X_train = custom_imputer.fit_transform(train_volume)
    custom_X_validation = custom_imputer.transform(validation_volume)
    custom_scaler = StandardScaler()
    custom_X_train = custom_scaler.fit_transform(custom_X_train)
    custom_X_validation = custom_scaler.transform(custom_X_validation)
    custom_model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000).fit(
        custom_X_train, y_train_volume.to_numpy()
    )
    custom_predictions = custom_model.predict(custom_X_validation)
    joblib.dump(custom_scaler, model_dir / "traffic_volume_custom_scaler.joblib")
    joblib.dump(custom_imputer, model_dir / "traffic_volume_custom_imputer.joblib")
    joblib.dump(custom_model, model_dir / "traffic_volume_custom.joblib")

    volume_results = {
        "custom_linear_regression": regression_metrics(y_validation_volume.to_numpy(), custom_predictions)
    }
    for name, model in build_regression_models().items():
        model.fit(train_volume, y_train_volume)
        predictions = model.predict(validation_volume)
        volume_results[name] = regression_metrics(y_validation_volume.to_numpy(), predictions)
        joblib.dump(model, model_dir / f"traffic_volume_{name}.joblib")
    results["traffic_volume"] = volume_results

    congestion_features = select_features(df, "congestion_level")
    congestion_encoder = LabelEncoder()
    y_train_congestion = congestion_encoder.fit_transform(train["congestion_level"])
    y_validation_congestion = congestion_encoder.transform(validation["congestion_level"])
    congestion_results = {}
    for name, model in build_classification_models().items():
        model.fit(train[congestion_features], y_train_congestion)
        predictions = model.predict(validation[congestion_features])
        congestion_results[name] = classification_metrics(y_validation_congestion, predictions)
        joblib.dump(model, model_dir / f"congestion_level_{name}.joblib")
    joblib.dump(congestion_encoder, model_dir / "congestion_level_encoder.joblib")
    results["congestion_level"] = congestion_results

    accident_features = select_features(df, "accident_count")
    y_train_accident = binary_target(train["accident_count"])
    y_validation_accident = binary_target(validation["accident_count"])
    accident_models = build_classification_models()
    accident_models.pop("decision_tree")
    accident_results = {}
    for name, model in accident_models.items():
        model.fit(train[accident_features], y_train_accident)
        predictions = model.predict(validation[accident_features])
        accident_results[name] = classification_metrics(y_validation_accident, predictions)
        joblib.dump(model, model_dir / f"accident_count_{name}.joblib")
    results["accident_count"] = accident_results

    joblib.dump(traffic_features, model_dir / "traffic_volume_features.joblib")
    joblib.dump(congestion_features, model_dir / "congestion_level_features.joblib")
    joblib.dump(accident_features, model_dir / "accident_count_features.joblib")
    return results


if __name__ == "__main__":
    training_results = train_model_suite()
    for target, models in training_results.items():
        print(f"\n{target}")
        for model_name, metrics in models.items():
            print(f"  {model_name}: {metrics}")

