
"""
File: backtest/simulator.py

Mô phỏng dự báo Swing High / Swing Low theo thời gian.

- Mỗi lần dự đoán chỉ sử dụng một nến đã đóng.
- Không đưa nhãn tương lai vào feature.
- Không retrain model trong quá trình backtest.
- Không tự động đặt lệnh giao dịch.
- Hỗ trợ TOP_SOON / BOTTOM_SOON / NONE.

Model interface:
    predict.predict(
        features,
        artifact=artifact,
        config=config
    )

Expected prediction fields:
    p_top_soon
    p_bottom_soon
    p_none
    predicted_class
    signal_status
    prediction_time
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

TARGET_COLUMN = "future_label_6"

PROBABILITY_COLUMNS = [
    "p_top_soon",
    "p_bottom_soon",
    "p_none",
]

CLASS_ORDER = [
    "TOP_SOON",
    "BOTTOM_SOON",
    "NONE",
]


def _load_predictor():
    """Import predictor tương thích cả package và chạy trực tiếp."""

    try:
        from ..models.predict import predict
    except ImportError:
        try:
            from models.predict import predict
        except ImportError as exc:
            raise ImportError(
                "Không thể import models.predict. "
                "Hãy chạy từ thư mục gốc dự án."
            ) from exc

    return predict


def _get_feature_columns(
    artifact: dict,
    config: Optional[dict] = None,
) -> Optional[list[str]]:
    """Lấy danh sách feature từ metadata của model."""

    metadata = artifact.get("metadata", {})

    columns = metadata.get("feature_columns")

    if columns is None and config is not None:
        columns = config.get("feature_columns")

    if columns is None:
        return None

    return list(columns)


def _extract_prediction(result: Any) -> dict:
    """Chuẩn hóa kết quả từ models.predict."""

    if isinstance(result, pd.DataFrame):
        if result.empty:
            raise ValueError("Predictor trả về DataFrame rỗng.")

        result = result.iloc[-1].to_dict()

    elif isinstance(result, pd.Series):
        result = result.to_dict()

    elif not isinstance(result, dict):
        raise TypeError(
            "Predictor phải trả về dict, Series hoặc DataFrame."
        )

    output = {}

    for column in PROBABILITY_COLUMNS:
        value = result.get(column, np.nan)

        try:
            output[column] = float(value)
        except (TypeError, ValueError):
            output[column] = np.nan

    output["predicted_class"] = result.get(
        "predicted_class", None
    )

    output["signal_status"] = result.get(
        "signal_status", "NO_SIGNAL"
    )

    output["prediction_time"] = result.get(
        "prediction_time", pd.NaT
    )

    return output


def _validate_probabilities(prediction: dict) -> None:
    """Kiểm tra xác suất đầu ra có hợp lệ."""

    probabilities = np.array(
        [
            prediction[column]
            for column in PROBABILITY_COLUMNS
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Xác suất dự đoán chứa NaN hoặc Inf.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Xác suất phải nằm trong khoảng [0, 1].")

    if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise ValueError(
            f"Tổng xác suất không bằng 1: {probabilities.sum()}"
        )


def _validate_features(features: pd.DataFrame) -> None:
    """Kiểm tra dữ liệu feature trước khi backtest."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features phải là pandas DataFrame.")

    if features.empty:
        raise ValueError("Không có dữ liệu để backtest.")

    if not features.index.is_monotonic_increasing:
        raise ValueError(
            "Dữ liệu phải được sắp xếp theo thời gian tăng dần."
        )

    if features.index.has_duplicates:
        raise ValueError("Timestamp bị trùng lặp.")

    forbidden = [
        column
        for column in features.columns
        if column.startswith("future_")
    ]

    if forbidden:
        raise ValueError(
            f"Phát hiện feature có khả năng lookahead: {forbidden}"
        )


def _prepare_single_row(
    features: pd.DataFrame,
    position: int,
    feature_columns: Optional[list[str]],
) -> pd.DataFrame:
    """
    Lấy đúng một mẫu tại thời điểm dự đoán.

    Không đưa target hoặc metadata tương lai vào model.
    """

    row = features.iloc[[position]].copy()

    forbidden = [
        column
        for column in row.columns
        if column.startswith("future_")
        or column == TARGET_COLUMN
    ]

    row = row.drop(columns=forbidden, errors="ignore")

    if feature_columns is not None:
        missing = [
            column
            for column in feature_columns
            if column not in row.columns
        ]

        if missing:
            raise ValueError(
                f"Thiếu feature cần thiết: {missing}"
            )

        row = row[feature_columns]

    return row


def run_backtest(
    features: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    artifact: Optional[dict] = None,
    config: Optional[dict] = None,
    predictor: Optional[Callable] = None,
    start_position: int = 0,
    end_position: Optional[int] = None,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Chạy backtest dự báo swing tuần tự.

    Parameters
    ----------
    features:
        DataFrame feature theo DatetimeIndex.

    labels:
        Nhãn thực tế TOP_SOON / BOTTOM_SOON / NONE.
        Có thể lấy từ features[TARGET_COLUMN].

    artifact:
        Artifact từ train_model/model_registry.
        Bao gồm model, metadata và calibrator nếu có.

    config:
        Cấu hình dự án.

    predictor:
        Hàm predict tùy chọn. Nếu None, sử dụng models.predict.predict.

    start_position:
        Vị trí bắt đầu mô phỏng.

    end_position:
        Vị trí kết thúc, không bao gồm vị trí này.

    strict:
        True: dừng khi một mẫu dự đoán lỗi.
        False: ghi nhận lỗi và tiếp tục.

    Returns
    -------
    pd.DataFrame
        Kết quả dự đoán theo thời gian.
    """

    _validate_features(features)

    if artifact is None:
        raise ValueError(
            "Chưa cung cấp model artifact. "
            "Không được tự tạo xác suất giả."
        )

    if labels is None:
        if TARGET_COLUMN not in features.columns:
            raise ValueError(
                f"Không tìm thấy nhãn {TARGET_COLUMN}."
            )

        labels = features[TARGET_COLUMN].copy()

    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels, index=features.index)

    if not labels.index.equals(features.index):
        labels = labels.reindex(features.index)

    if predictor is None:
        predictor = _load_predictor()

    feature_columns = _get_feature_columns(
        artifact,
        config,
    )

    end_position = (
        len(features)
        if end_position is None
        else min(end_position, len(features))
    )

    if start_position < 0 or start_position >= len(features):
        raise ValueError("start_position không hợp lệ.")

    if end_position <= start_position:
        raise ValueError("Khoảng backtest không hợp lệ.")

    results = []

    logger.info(
        "Bắt đầu backtest: %d mẫu.",
        end_position - start_position,
    )

    for position in range(start_position, end_position):

        timestamp = features.index[position]

        try:
            # Chỉ lấy feature của nến hiện tại đã đóng.
            X = _prepare_single_row(
                features,
                position,
                feature_columns,
            )

            # Dự đoán độc lập tại từng thời điểm.
            prediction_result = predictor(
                X,
                artifact=artifact,
                config=config,
            )

            prediction = _extract_prediction(
                prediction_result
            )

            _validate_probabilities(prediction)

            actual_label = labels.iloc[position]

            results.append({
                "timestamp": timestamp,
                "actual_label": actual_label,
                **prediction,
                "error": None,
            })

        except Exception as exc:

            logger.exception(
                "Lỗi backtest tại %s: %s",
                timestamp,
                exc,
            )

            if strict:
                raise RuntimeError(
                    f"Backtest thất bại tại {timestamp}"
                ) from exc

            results.append({
                "timestamp": timestamp,
                "actual_label": labels.iloc[position],
                "p_top_soon": np.nan,
                "p_bottom_soon": np.nan,
                "p_none": np.nan,
                "predicted_class": None,
                "signal_status": "ERROR",
                "prediction_time": pd.NaT,
                "error": str(exc),
            })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.set_index("timestamp")
        result_df.index.name = "timestamp"

    logger.info(
        "Hoàn thành backtest: %d mẫu.",
        len(result_df),
    )

    return result_df


def simulate(
    features: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    artifact: Optional[dict] = None,
    config: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Alias ngắn cho run_backtest."""

    return run_backtest(
        features=features,
        labels=labels,
        artifact=artifact,
        config=config,
        **kwargs,
    )


def save_backtest_results(
    results: pd.DataFrame,
    output_path: str,
) -> None:
    """Lưu kết quả backtest ra CSV hoặc Parquet."""

    if results.empty:
        raise ValueError("Không có kết quả để lưu.")

    path = str(output_path)

    if path.endswith(".parquet"):
        results.to_parquet(path)
    elif path.endswith(".csv"):
        results.to_csv(path)
    else:
        raise ValueError(
            "Chỉ hỗ trợ định dạng .csv và .parquet."
        )

    logger.info("Đã lưu kết quả: %s", path)
