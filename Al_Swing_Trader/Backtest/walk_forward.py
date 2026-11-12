
"""
File: backtest/walk_forward.py

Walk-forward validation cho AI Swing Trading.

Hỗ trợ:
- Expanding window.
- Rolling window.
- Purge trước tập test.
- Embargo giữa các fold.
- Callback train/predict/metrics.
- Lưu kết quả từng fold.

Không sử dụng dữ liệu tương lai để huấn luyện.
Không tự động điều chỉnh hyperparameter dựa trên test.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Cấu hình walk-forward."""

    n_splits: int = 5

    test_size: int = 500

    min_train_size: int = 2000

    # None: expanding window.
    # Số nguyên: rolling window.
    train_window: Optional[int] = None

    # Số mẫu cần loại khỏi cuối train
    # để tránh nhãn tương lai chồng lấn.
    purge_size: int = 12

    # Khoảng cách tối thiểu giữa các test fold.
    embargo_size: int = 0

    # Khoảng cách giữa các lần bắt đầu test.
    step_size: Optional[int] = None

    # Expanding hoặc rolling.
    window_type: str = "expanding"

    # Chỉ dùng khi cần bỏ phần dữ liệu đầu.
    start_position: int = 0

    def __post_init__(self):
        if self.n_splits < 1:
            raise ValueError("n_splits phải >= 1.")

        if self.test_size < 1:
            raise ValueError("test_size phải >= 1.")

        if self.min_train_size < 1:
            raise ValueError("min_train_size phải >= 1.")

        if self.purge_size < 0:
            raise ValueError("purge_size không được âm.")

        if self.embargo_size < 0:
            raise ValueError("embargo_size không được âm.")

        if self.start_position < 0:
            raise ValueError("start_position không được âm.")

        if self.window_type not in {"expanding", "rolling"}:
            raise ValueError(
                "window_type chỉ nhận expanding hoặc rolling."
            )

        if self.window_type == "rolling":
            if self.train_window is None:
                raise ValueError(
                    "Rolling window cần train_window."
                )

            if self.train_window < self.min_train_size:
                raise ValueError(
                    "train_window phải >= min_train_size."
                )

        if self.step_size is not None and self.step_size < 1:
            raise ValueError("step_size phải >= 1.")


@dataclass
class Fold:
    """Thông tin một fold."""

    fold_id: int

    train_start: int
    train_end: int

    test_start: int
    test_end: int

    train_index: pd.Index
    test_index: pd.Index


def generate_splits(
    data: pd.DataFrame,
    config: Optional[WalkForwardConfig] = None,
) -> list[Fold]:
    """
    Tạo các fold theo thứ tự thời gian.

    Quy ước:
        train_end: exclusive
        test_end: exclusive

    Khoảng purge được tính từ test_start lùi về trước.

    Ví dụ:
        test_start = 2000
        purge_size = 12

        train_end = 1988
    """

    if config is None:
        config = WalkForwardConfig()

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data phải là DataFrame.")

    if data.empty:
        raise ValueError("Không có dữ liệu.")

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Dữ liệu phải được sắp xếp tăng dần theo thời gian."
        )

    if data.index.has_duplicates:
        raise ValueError("Dữ liệu có timestamp trùng.")

    total = len(data)

    start = config.start_position

    if start >= total:
        raise ValueError("start_position vượt quá dữ liệu.")

    step = (
        config.step_size
        if config.step_size is not None
        else config.test_size + config.embargo_size
    )

    first_test_start = (
        total - config.test_size
        - (config.n_splits - 1) * step
    )

    first_test_start = max(
        first_test_start,
        start + config.min_train_size + config.purge_size,
    )

    folds = []

    for fold_id in range(config.n_splits):

        test_start = first_test_start + fold_id * step

        test_end = test_start + config.test_size

        if test_end > total:
            break

        # Purge mẫu có nhãn tương lai chồng lấn.
        train_end = test_start - config.purge_size

        if config.window_type == "expanding":
            train_start = start

        else:
            train_start = max(
                start,
                train_end - config.train_window,
            )

        train_size = train_end - train_start

        if train_size < config.min_train_size:
            continue

        train_index = data.index[
            train_start:train_end
        ]

        test_index = data.index[
            test_start:test_end
        ]

        folds.append(
            Fold(
                fold_id=fold_id + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_index=train_index,
                test_index=test_index,
            )
        )

    if not folds:
        raise ValueError(
            "Không tạo được fold hợp lệ. "
            "Hãy kiểm tra số lượng dữ liệu, "
            "min_train_size, test_size và purge_size."
        )

    return folds


def _validate_xy(
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Kiểm tra feature và target."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X phải là DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y phải là Series.")

    if len(X) != len(y):
        raise ValueError("X và y không cùng số mẫu.")

    if not X.index.equals(y.index):
        raise ValueError(
            "Index của X và y không khớp."
        )

    if not X.index.is_monotonic_increasing:
        raise ValueError(
            "X phải được sắp xếp theo thời gian."
        )

    forbidden = [
        column
        for column in X.columns
        if column.startswith("future_")
    ]

    if forbidden:
        raise ValueError(
            f"Feature chứa cột tương lai: {forbidden}"
        )


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    train_fn: Callable,
    predict_fn: Callable,
    config: Optional[WalkForwardConfig] = None,
    metrics_fn: Optional[Callable] = None,
) -> dict:
    """
    Thực hiện walk-forward validation.

    train_fn:
        train_fn(X_train, y_train, fold)

        Trả về model artifact của fold.

    predict_fn:
        predict_fn(model, X_test, fold)

        Trả về prediction array, Series hoặc DataFrame.

    metrics_fn:
        metrics_fn(y_true, predictions)

        Trả về dictionary các metrics.

    Lưu ý:
        Việc preprocessing, feature selection,
        calibrator và hyperparameter fitting
        phải nằm bên trong train_fn.

        Không được fit chúng trên toàn bộ dữ liệu.
    """

    _validate_xy(X, y)

    if config is None:
        config = WalkForwardConfig()

    folds = generate_splits(X, config)

    fold_results = []
    prediction_frames = []

    for fold in folds:

        logger.info(
            "Fold %d | train=%d | test=%d",
            fold.fold_id,
            len(fold.train_index),
            len(fold.test_index),
        )

        X_train = X.iloc[
            fold.train_start:fold.train_end
        ].copy()

        y_train = y.iloc[
            fold.train_start:fold.train_end
        ].copy()

        X_test = X.iloc[
            fold.test_start:fold.test_end
        ].copy()

        y_test = y.iloc[
            fold.test_start:fold.test_end
        ].copy()

        # Kiểm tra không có dữ liệu train vượt test.
        if X_train.index.max() >= X_test.index.min():
            raise RuntimeError(
                f"Fold {fold.fold_id}: phát hiện temporal leakage."
            )

        # Huấn luyện riêng từng fold.
        model = train_fn(
            X_train,
            y_train,
            fold,
        )

        if model is None:
            raise RuntimeError(
                f"Fold {fold.fold_id}: train_fn trả về None."
            )

        # Chỉ dự đoán trên test.
        predictions = predict_fn(
            model,
            X_test,
            fold,
        )

        if isinstance(predictions, pd.DataFrame):
            prediction_df = predictions.copy()

            if len(prediction_df) != len(X_test):
                raise ValueError(
                    "Số dự đoán không khớp số mẫu test."
                )

            prediction_df.index = X_test.index

        elif isinstance(predictions, pd.Series):
            if len(predictions) != len(X_test):
                raise ValueError(
                    "Số dự đoán không khớp số mẫu test."
                )

            prediction_df = predictions.to_frame(
                name="prediction"
            )

            prediction_df.index = X_test.index

        else:
            predictions = np.asarray(predictions)

            if len(predictions) != len(X_test):
                raise ValueError(
                    "Số dự đoán không khớp số mẫu test."
                )

            prediction_df = pd.DataFrame(
                {"prediction": predictions},
                index=X_test.index,
            )

        prediction_df["actual"] = y_test

        prediction_df["fold_id"] = fold.fold_id

        prediction_frames.append(prediction_df)

        fold_metrics = {}

        if metrics_fn is not None:
            fold_metrics = metrics_fn(
                y_test,
                predictions,
            )

            if not isinstance(fold_metrics, dict):
                raise TypeError(
                    "metrics_fn phải trả về dictionary."
                )

        fold_results.append({
            "fold_id": fold.fold_id,
            "train_start": X_train.index.min(),
            "train_end": X_train.index.max(),
            "test_start": X_test.index.min(),
            "test_end": X_test.index.max(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "metrics": fold_metrics,
        })

    all_predictions = pd.concat(
        prediction_frames,
        axis=0,
    )

    return {
        "folds": fold_results,
        "predictions": all_predictions,
        "config": config,
        "n_folds_completed": len(fold_results),
    }


def save_walk_forward_results(
    results: dict,
    output_path: str,
) -> None:
    """Lưu kết quả dự báo và báo cáo fold."""

    from pathlib import Path
    import json

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = results["predictions"]

    prediction_path = output.with_name(
        output.stem + "_predictions.csv"
    )

    predictions.to_csv(prediction_path)

    folds = results["folds"]

    fold_path = output.with_name(
        output.stem + "_folds.json"
    )

    with fold_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            folds,
            file,
            ensure_ascii=False,
            indent=4,
            default=str,
            allow_nan=False,
        )

    logger.info(
        "Đã lưu walk-forward tại %s",
        output.parent,
    )
