
# File: models/train_model.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except ImportError as exc:
    raise ImportError(
        "Chưa cài LightGBM. Cài bằng: pip install lightgbm"
    ) from exc


EXPECTED_CLASSES = [
    "TOP_SOON",
    "BOTTOM_SOON",
    "NONE",
]


@dataclass
class TrainConfig:
    """Cấu hình huấn luyện mô hình."""

    target_column: str = "future_label_6"

    horizon_bars: int = 6

    # Tỷ lệ chia dữ liệu theo thứ tự thời gian.
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # Số mẫu bỏ ở hai bên mỗi ranh giới.
    # Mặc định bằng horizon để hạn chế nhãn chồng lấn.
    purge_bars: Optional[int] = None

    random_state: int = 42

    # Số vòng tối đa và early stopping.
    n_estimators: int = 1200
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 30

    # Cân bằng lớp.
    class_weight: Optional[str] = "balanced"

    # Lưu artifact và báo cáo.
    model_dir: str = "models_store"

    # Bắt buộc dữ liệu có đủ cả ba lớp.
    require_all_classes: bool = True


def _validate_dataset(
    dataset: pd.DataFrame,
    config: TrainConfig,
) -> pd.DataFrame:
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset phải là pandas DataFrame")

    if config.target_column not in dataset.columns:
        raise ValueError(
            f"Không tìm thấy target column: {config.target_column}"
        )

    if not dataset.index.is_monotonic_increasing:
        raise ValueError(
            "Dataset phải được sắp xếp theo thời gian tăng dần"
        )

    if dataset.index.has_duplicates:
        raise ValueError(
            "Dataset có index trùng lặp"
        )

    ratios = [
        config.train_ratio,
        config.validation_ratio,
        config.test_ratio,
    ]

    if any(value <= 0 for value in ratios):
        raise ValueError("Các tỷ lệ chia phải lớn hơn 0")

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio phải bằng 1"
        )

    if config.horizon_bars < 1:
        raise ValueError("horizon_bars phải >= 1")

    df = dataset.copy()

    # Chỉ loại các mẫu chưa biết nhãn.
    df = df.dropna(
        subset=[config.target_column]
    )

    if df.empty:
        raise ValueError("Không còn mẫu nào có nhãn hợp lệ")

    df[config.target_column] = (
        df[config.target_column].astype(str)
    )

    unknown = (
        set(df[config.target_column].unique())
        - set(EXPECTED_CLASSES)
    )

    if unknown:
        raise ValueError(
            f"Có target class không hợp lệ: {sorted(unknown)}"
        )

    return df


def _select_features(
    dataset: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
) -> list[str]:
    """
    Chọn feature số và loại các cột có nguy cơ rò rỉ nhãn.

    Không sử dụng:
    - future_*
    - historical_label
    - confirmation_time
    - pivot_time
    - metadata swing
    - metadata zone chỉ biết trong tương lai
    """
    forbidden_exact = {
        target_column,
        "historical_label",
        "historical_size",
        "is_swing",
        "swing_confirmation_time",
        "pivot_time",
        "confirmation_time",
        "confirmation_price",
        "future_event_time",
        "future_confirmation_time",
        "future_swing_id",
        "bars_to_future_event",
        "future_horizon_bars",
        "future_label_version",
        "zone_created_time",
        "invalidated_time",
    }

    def is_forbidden(column: str) -> bool:
        name = column.lower()

        if name in forbidden_exact:
            return True

        if name.startswith("future_"):
            return True

        if name.startswith("target_"):
            return True

        if name.endswith("_label"):
            return True

        if name.endswith("_id"):
            return True

        if name.endswith("_time"):
            return True

        return False

    if feature_columns is None:
        selected = [
            column
            for column in dataset.columns
            if column != target_column
            and not is_forbidden(column)
            and pd.api.types.is_numeric_dtype(
                dataset[column]
            )
        ]
    else:
        missing = set(feature_columns) - set(dataset.columns)

        if missing:
            raise ValueError(
                f"Thiếu feature columns: {sorted(missing)}"
            )

        selected = [
            column
            for column in feature_columns
            if not is_forbidden(column)
            and pd.api.types.is_numeric_dtype(
                dataset[column]
            )
        ]

        if len(selected) != len(feature_columns):
            raise ValueError(
                "Danh sách feature chứa cột bị cấm hoặc không phải số"
            )

    if not selected:
        raise ValueError("Không tìm được feature hợp lệ")

    return selected


def _chronological_split(
    dataset: pd.DataFrame,
    config: TrainConfig,
):
    """
    Chia theo thời gian và tạo khoảng purge giữa các tập.

    Các mẫu nằm sát ranh giới bị loại khỏi quá trình train
    để giảm chồng lấn cửa sổ nhãn tương lai.
    """
    n = len(dataset)

    first_boundary = int(
        n * config.train_ratio
    )

    second_boundary = int(
        n * (
            config.train_ratio
            + config.validation_ratio
        )
    )

    purge = (
        config.horizon_bars
        if config.purge_bars is None
        else config.purge_bars
    )

    if purge < 0:
        raise ValueError("purge_bars không được âm")

    train_end = first_boundary - purge
    validation_start = first_boundary + purge

    validation_end = second_boundary - purge
    test_start = second_boundary + purge

    train = dataset.iloc[:train_end].copy()

    validation = dataset.iloc[
        validation_start:validation_end
    ].copy()

    test = dataset.iloc[test_start:].copy()

    if min(len(train), len(validation), len(test)) < 1:
        raise ValueError(
            "Dữ liệu không đủ sau khi chia và purge. "
            "Hãy tăng dữ liệu hoặc giảm purge_bars."
        )

    return train, validation, test


def _calculate_multiclass_brier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
) -> float:
    """
    Multiclass Brier score dạng trung bình tổng bình phương sai số.
    Giá trị thấp hơn tốt hơn.
    """
    one_hot = np.zeros_like(probabilities)

    class_to_index = {
        name: index
        for index, name in enumerate(classes)
    }

    for row, label in enumerate(y_true):
        one_hot[row, class_to_index[label]] = 1.0

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )


def _evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    probabilities_raw = model.predict_proba(X)

    model_classes = [
        str(value)
        for value in model.classes_
    ]

    # Chuẩn hóa thứ tự xác suất theo EXPECTED_CLASSES.
    probabilities = np.zeros(
        (len(X), len(EXPECTED_CLASSES)),
        dtype=float,
    )

    for source_index, class_name in enumerate(model_classes):
        target_index = EXPECTED_CLASSES.index(
            class_name
        )

        probabilities[:, target_index] = (
            probabilities_raw[:, source_index]
        )

    predictions = np.array(EXPECTED_CLASSES)[
        np.argmax(probabilities, axis=1)
    ]

    labels_present = [
        label
        for label in EXPECTED_CLASSES
        if label in set(y.astype(str))
    ]

    report = classification_report(
        y.astype(str),
        predictions,
        labels=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),

        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=EXPECTED_CLASSES,
                average="macro",
                zero_division=0,
            )
        ),

        "log_loss": float(
            log_loss(
                y,
                probabilities,
                labels=EXPECTED_CLASSES,
            )
        ),

        "multiclass_brier": _calculate_multiclass_brier(
            y.astype(str).to_numpy(),
            probabilities,
            EXPECTED_CLASSES,
        ),

        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=EXPECTED_CLASSES,
        ).tolist(),

        "classification_report": report,

        "classes": EXPECTED_CLASSES,
        "class_counts": {
            label: int((y == label).sum())
            for label in EXPECTED_CLASSES
        },
    }

    return metrics


def train_model(
    dataset: pd.DataFrame,
    config: Optional[TrainConfig] = None,
    feature_columns: Optional[list[str]] = None,
) -> dict:
    """
    Huấn luyện LightGBM và lưu model.

    dataset cần chứa:
        - các feature từ feature_builder.py
        - target_column từ future_event_labels.py

    Hàm không tự tạo nhãn và không tự tạo feature.
    """
    if config is None:
        config = TrainConfig()

    df = _validate_dataset(
        dataset,
        config,
    )

    selected_features = _select_features(
        df,
        config.target_column,
        feature_columns,
    )

    train_df, validation_df, test_df = (
        _chronological_split(df, config)
    )

    y_train = train_df[
        config.target_column
    ]

    y_validation = validation_df[
        config.target_column
    ]

    y_test = test_df[
        config.target_column
    ]

    if config.require_all_classes:
        for name, labels in [
            ("train", y_train),
            ("validation", y_validation),
            ("test", y_test),
        ]:
            missing = (
                set(EXPECTED_CLASSES)
                - set(labels.unique())
            )

            if missing:
                raise ValueError(
                    f"Tập {name} thiếu class: {sorted(missing)}. "
                    "Không nên đánh giá mô hình 3 lớp trên tập "
                    "không có đủ ba lớp."
                )

    X_train = train_df[selected_features]
    X_validation = validation_df[selected_features]
    X_test = test_df[selected_features]

    # LightGBM hỗ trợ NaN nhưng không hỗ trợ infinity.
    for name, X in [
        ("train", X_train),
        ("validation", X_validation),
        ("test", X_test),
    ]:
        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError(
                f"Tập {name} chứa infinity"
            )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(EXPECTED_CLASSES),

        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,

        class_weight=config.class_weight,
        random_state=config.random_state,

        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_validation, y_validation)
        ],

        eval_metric="multi_logloss",

        callbacks=[
            early_stopping(
                stopping_rounds=100,
                verbose=False,
            ),
            log_evaluation(period=100),
        ],
    )

    # Test chỉ được sử dụng sau khi lựa chọn số vòng
    # bằng validation.
    metrics = _evaluate_model(
        model,
        X_test,
        y_test,
    )

    model_dir = Path(config.model_dir)
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = model_dir / "swing_model.joblib"
    metrics_path = model_dir / "test_metrics.json"

    metadata = {
        "model_type": "LightGBM",
        "target_column": config.target_column,
        "horizon_bars": config.horizon_bars,
        "classes": EXPECTED_CLASSES,
        "feature_columns": selected_features,
        "best_iteration": int(
            model.best_iteration_
            if model.best_iteration_
            else config.n_estimators
        ),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "train_start": str(train_df.index[0]),
        "train_end": str(train_df.index[-1]),
        "validation_start": str(validation_df.index[0]),
        "validation_end": str(validation_df.index[-1]),
        "test_start": str(test_df.index[0]),
        "test_end": str(test_df.index[-1]),
        "purge_bars": (
            config.horizon_bars
            if config.purge_bars is None
            else config.purge_bars
        ),
        "config": asdict(config),
    }

    artifact = {
        "model": model,
        "metadata": metadata,
    }

    joblib.dump(
        artifact,
        model_path,
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metadata": metadata,
        "metrics": metrics,
    }


if __name__ == "__main__":
    # Ví dụ chạy độc lập:
    # Dataset cần được tạo từ feature_builder và future labels.
    dataset_path = Path("data/training_dataset.csv")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {dataset_path}. "
            "Hãy tạo training dataset trước."
        )

    dataset = pd.read_csv(
        dataset_path,
        index_col=0,
        parse_dates=True,
    )

    config = TrainConfig(
        target_column="future_label_6",
        horizon_bars=6,
    )

    result = train_model(
        dataset,
        config,
    )

    print("Model:", result["model_path"])
    print("Metrics:", result["metrics_path"])
    print(
        "Macro F1:",
        result["metrics"]["macro_f1"],
    )
