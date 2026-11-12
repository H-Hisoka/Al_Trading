
# File: models/predict.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


EXPECTED_CLASSES = [
    "TOP_SOON",
    "BOTTOM_SOON",
    "NONE",
]


@dataclass
class PredictConfig:
    """Cấu hình dự đoán."""

    model_path: str = "models_store/swing_model.joblib"

    # None nghĩa là chỉ trả xác suất, chưa tạo trạng thái WATCH.
    watch_threshold: Optional[float] = None

    # Chỉ trả kết quả của nến cuối cùng.
    latest_only: bool = True

    # Số chữ số thập phân khi hiển thị xác suất.
    probability_digits: int = 4


def load_model(
    model_path: str = "models_store/swing_model.joblib",
) -> dict:
    """
    Nạp model và metadata.

    Artifact phải được tạo bởi train_model.py.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {path}"
        )

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError("Model artifact không hợp lệ")

    if "model" not in artifact or "metadata" not in artifact:
        raise ValueError(
            "Artifact thiếu model hoặc metadata"
        )

    metadata = artifact["metadata"]

    required_metadata = {
        "feature_columns",
        "classes",
        "target_column",
    }

    missing = required_metadata - set(metadata)

    if missing:
        raise ValueError(
            f"Metadata thiếu: {sorted(missing)}"
        )

    if metadata["classes"] != EXPECTED_CLASSES:
        raise ValueError(
            "Class trong model không đúng thứ tự dự án"
        )

    return artifact


def _prepare_features(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Đồng bộ feature đầu vào theo danh sách feature lúc train.

    - Bỏ các cột dư.
    - Báo lỗi nếu thiếu cột.
    - Không tự điền NaN thành 0.
    - Loại bỏ infinity.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "features phải là pandas DataFrame"
        )

    if features.empty:
        raise ValueError("Không có dữ liệu để dự đoán")

    if not features.index.is_monotonic_increasing:
        raise ValueError(
            "Features phải được sắp xếp theo thời gian tăng dần"
        )

    if features.index.has_duplicates:
        raise ValueError(
            "Features có timestamp trùng lặp"
        )

    missing = (
        set(feature_columns)
        - set(features.columns)
    )

    if missing:
        raise ValueError(
            f"Thiếu features đã dùng khi train: {sorted(missing)}"
        )

    X = features[feature_columns].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # LightGBM xử lý NaN trực tiếp.
    # Không tự động biến NaN thành 0.
    return X


def _map_probabilities(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Chuẩn hóa thứ tự xác suất theo:
        TOP_SOON
        BOTTOM_SOON
        NONE
    """
    raw_probabilities = model.predict_proba(X)

    model_classes = [
        str(value)
        for value in model.classes_
    ]

    probabilities = np.zeros(
        (len(X), len(EXPECTED_CLASSES)),
        dtype=float,
    )

    for source_index, class_name in enumerate(model_classes):
        if class_name not in EXPECTED_CLASSES:
            raise ValueError(
                f"Model chứa class không hợp lệ: {class_name}"
            )

        target_index = EXPECTED_CLASSES.index(
            class_name
        )

        probabilities[:, target_index] = (
            raw_probabilities[:, source_index]
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Model trả về xác suất không hợp lệ"
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Tổng xác suất không bằng 1"
        )

    return pd.DataFrame(
        probabilities,
        columns=[
            "p_top_soon",
            "p_bottom_soon",
            "p_none",
        ],
        index=X.index,
    )


def _get_signal_status(
    p_top: float,
    p_bottom: float,
    threshold: Optional[float],
) -> str:
    """
    Phân loại trạng thái theo ngưỡng được cấu hình.

    Đây là trạng thái theo dõi dự báo,
    không phải xác nhận swing đã hình thành.
    """
    if threshold is None:
        return "NO_SIGNAL"

    if not 0 < threshold < 1:
        raise ValueError(
            "watch_threshold phải nằm trong khoảng (0, 1)"
        )

    top_detected = p_top >= threshold
    bottom_detected = p_bottom >= threshold

    if top_detected and bottom_detected:
        return "WATCH_BOTH"

    if top_detected:
        return "WATCH_TOP"

    if bottom_detected:
        return "WATCH_BOTTOM"

    return "NO_SIGNAL"


def predict(
    features: pd.DataFrame,
    artifact: Optional[dict] = None,
    config: Optional[PredictConfig] = None,
) -> pd.DataFrame:
    """
    Dự đoán xác suất TOP_SOON / BOTTOM_SOON / NONE.

    Input:
        features:
            DataFrame được tạo bởi feature_builder.py.

        artifact:
            Kết quả từ load_model().
            Nếu không truyền, tự nạp theo model_path.

    Output:
        DataFrame chứa:
            p_top_soon
            p_bottom_soon
            p_none
            predicted_class
            signal_status
            prediction_time

    Chú ý:
        features phải chỉ chứa dữ liệu đã có sẵn
        tại thời điểm dự đoán.
    """
    if config is None:
        config = PredictConfig()

    if artifact is None:
        artifact = load_model(
            config.model_path
        )

    model = artifact["model"]
    metadata = artifact["metadata"]

    feature_columns = metadata["feature_columns"]

    if config.watch_threshold is not None:
        if not 0 < config.watch_threshold < 1:
            raise ValueError(
                "watch_threshold phải nằm trong khoảng (0, 1)"
            )

    X = _prepare_features(
        features,
        feature_columns,
    )

    if config.latest_only:
        X = X.tail(1)

    probabilities = _map_probabilities(
        model,
        X,
    )

    result = probabilities.copy()

    predicted_indices = np.argmax(
        probabilities.to_numpy(),
        axis=1,
    )

    result["predicted_class"] = [
        EXPECTED_CLASSES[index]
        for index in predicted_indices
    ]

    result["signal_status"] = [
        _get_signal_status(
            float(row["p_top_soon"]),
            float(row["p_bottom_soon"]),
            config.watch_threshold,
        )
        for _, row in probabilities.iterrows()
    ]

    result["prediction_time"] = result.index

    # Làm tròn xác suất sau khi tính toán.
    probability_columns = [
        "p_top_soon",
        "p_bottom_soon",
        "p_none",
    ]

    result[probability_columns] = result[
        probability_columns
    ].round(config.probability_digits)

    return result


def predict_latest(
    features: pd.DataFrame,
    model_path: str = "models_store/swing_model.joblib",
    watch_threshold: Optional[float] = None,
) -> dict:
    """
    Hàm tiện ích để lấy dự báo nến mới nhất.
    """
    artifact = load_model(model_path)

    config = PredictConfig(
        model_path=model_path,
        watch_threshold=watch_threshold,
        latest_only=True,
    )

    result = predict(
        features=features,
        artifact=artifact,
        config=config,
    )

    return result.iloc[-1].to_dict()


if __name__ == "__main__":
    # Ví dụ:
    # features.csv phải được tạo bởi feature_builder.py.
    feature_path = Path("data/features.csv")

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy feature file: {feature_path}"
        )

    features = pd.read_csv(
        feature_path,
        index_col=0,
        parse_dates=True,
    )

    artifact = load_model()

    result = predict(
        features=features,
        artifact=artifact,
        config=PredictConfig(
            latest_only=True,
            watch_threshold=None,
        ),
    )

    print(result.to_string())
