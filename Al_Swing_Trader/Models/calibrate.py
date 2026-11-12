
# File: models/calibrate.py

import json
import os
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import datetime, timezone

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss


# ============================================================
# 1. CONFIGURATION
# ============================================================

EXPECTED_CLASSES = [
    "TOP_SOON",
    "BOTTOM_SOON",
    "NONE",
]


@dataclass
class CalibrationConfig:

    model_path: str = "models_store/swing_model.joblib"

    data_path: str = "data/features.csv"

    output_metrics_path: str = (
        "models_store/calibration_metrics.json"
    )

    method: str = "sigmoid"

    target_column: str = "future_label_6"

    train_ratio: float = 0.70

    validation_ratio: float = 0.15

    purge_bars: int = 6

    min_samples_per_class: int = 20

    random_state: int = 42


# ============================================================
# 2. CALIBRATOR
# ============================================================

class MulticlassCalibrator:

    def __init__(self, method="sigmoid"):

        if method not in ("sigmoid", "isotonic"):
            raise ValueError(
                "method phải là 'sigmoid' hoặc 'isotonic'"
            )

        self.method = method

        self.models = {}

        self.classes = EXPECTED_CLASSES.copy()

        self.is_fitted = False

    def fit(self, probabilities, y_true):

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        y_true = np.asarray(y_true)

        if probabilities.ndim != 2:
            raise ValueError(
                "probabilities phải là ma trận 2 chiều"
            )

        if probabilities.shape[1] != len(self.classes):
            raise ValueError(
                "Số cột xác suất không khớp số class"
            )

        if len(probabilities) != len(y_true):
            raise ValueError(
                "Số mẫu xác suất và nhãn không khớp"
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Xác suất chứa NaN hoặc infinity"
            )

        if np.any(probabilities < 0):
            raise ValueError(
                "Xác suất không được âm"
            )

        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-4
        ):
            raise ValueError(
                "Tổng xác suất mỗi hàng phải bằng 1"
            )

        self.models = {}

        for index, class_name in enumerate(self.classes):

            binary_target = (
                y_true == class_name
            ).astype(int)

            if len(np.unique(binary_target)) < 2:
                raise ValueError(
                    f"Không đủ cả positive/negative "
                    f"cho class {class_name}"
                )

            class_probability = np.clip(
                probabilities[:, index],
                1e-6,
                1 - 1e-6
            )

            if self.method == "sigmoid":

                # Chuyển xác suất thành logit.
                logits = np.log(
                    class_probability /
                    (1 - class_probability)
                ).reshape(-1, 1)

                calibrator = LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )

                calibrator.fit(
                    logits,
                    binary_target
                )

            else:

                calibrator = IsotonicRegression(
                    y_min=0,
                    y_max=1,
                    out_of_bounds="clip"
                )

                calibrator.fit(
                    class_probability,
                    binary_target
                )

            self.models[class_name] = calibrator

        self.is_fitted = True

        return self

    def predict_proba(self, probabilities):

        if not self.is_fitted:
            raise RuntimeError(
                "Calibrator chưa được fit"
            )

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        calibrated_columns = []

        for index, class_name in enumerate(self.classes):

            original_probability = np.clip(
                probabilities[:, index],
                1e-6,
                1 - 1e-6
            )

            calibrator = self.models[class_name]

            if self.method == "sigmoid":

                logits = np.log(
                    original_probability /
                    (1 - original_probability)
                ).reshape(-1, 1)

                calibrated = calibrator.predict_proba(
                    logits
                )[:, 1]

            else:

                calibrated = calibrator.predict(
                    original_probability
                )

            calibrated_columns.append(calibrated)

        calibrated_probabilities = np.column_stack(
            calibrated_columns
        )

        # Chuẩn hóa để tổng xác suất bằng 1.
        row_sums = calibrated_probabilities.sum(
            axis=1,
            keepdims=True
        )

        calibrated_probabilities = np.divide(
            calibrated_probabilities,
            row_sums,
            out=np.full_like(
                calibrated_probabilities,
                1.0 / len(self.classes)
            ),
            where=row_sums > 1e-12
        )

        return calibrated_probabilities


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model_artifact(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path}"
        )

    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict):
        raise ValueError(
            "Artifact không đúng định dạng"
        )

    if "model" not in artifact:
        raise ValueError(
            "Artifact thiếu model"
        )

    if "metadata" not in artifact:
        raise ValueError(
            "Artifact thiếu metadata"
        )

    metadata = artifact["metadata"]

    if metadata.get("classes") != EXPECTED_CLASSES:
        raise ValueError(
            "Thứ tự classes không khớp"
        )

    return artifact


# ============================================================
# 4. PREPARE VALIDATION DATA
# ============================================================

def prepare_validation_data(df, artifact, config):

    if config.target_column not in df.columns:
        raise ValueError(
            f"Không tìm thấy target: {config.target_column}"
        )

    metadata = artifact["metadata"]

    feature_columns = metadata.get(
        "feature_columns"
    )

    if not feature_columns:
        raise ValueError(
            "Metadata thiếu feature_columns"
        )

    # Giữ đúng thứ tự dữ liệu như lúc train.
    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Thiếu feature: {missing_columns}"
        )

    df = df.copy()

    # Loại bỏ các hàng chưa có nhãn tương lai.
    df = df.dropna(
        subset=[config.target_column]
    )

    y = df[config.target_column].astype(str)

    valid_classes = set(EXPECTED_CLASSES)

    invalid_classes = set(y.unique()) - valid_classes

    if invalid_classes:
        raise ValueError(
            f"Target chứa class không hợp lệ: "
            f"{invalid_classes}"
        )

    # Tạo lại split theo thứ tự thời gian.
    # Không shuffle dữ liệu.
    total_rows = len(df)

    train_end = int(
        total_rows * config.train_ratio
    )

    validation_end = int(
        total_rows * (
            config.train_ratio +
            config.validation_ratio
        )
    )

    validation_start = (
        train_end + config.purge_bars
    )

    validation_stop = (
        validation_end - config.purge_bars
    )

    if validation_start >= validation_stop:
        raise ValueError(
            "Validation không đủ dữ liệu sau purge"
        )

    validation_df = df.iloc[
        validation_start:validation_stop
    ].copy()

    X_val = validation_df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_val = X_val.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y_val = validation_df[
        config.target_column
    ].astype(str)

    return X_val, y_val


# ============================================================
# 5. EVALUATION
# ============================================================

def calculate_calibration_metrics(
    y_true,
    probabilities
):

    y_true = np.asarray(y_true)

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(
            EXPECTED_CLASSES
        )
    }

    y_encoded = np.array([
        class_to_index[value]
        for value in y_true
    ])

    result = {}

    result["log_loss"] = float(
        log_loss(
            y_encoded,
            probabilities,
            labels=list(range(len(EXPECTED_CLASSES)))
        )
    )

    brier_scores = {}

    for index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        binary_target = (
            y_true == class_name
        ).astype(int)

        brier_scores[class_name] = float(
            brier_score_loss(
                binary_target,
                probabilities[:, index]
            )
        )

    result["brier_per_class"] = brier_scores

    result["mean_brier"] = float(
        np.mean(list(brier_scores.values()))
    )

    return result


# ============================================================
# 6. SAVE ARTIFACT SAFELY
# ============================================================

def save_artifact_atomic(artifact, output_path):

    directory = os.path.dirname(output_path)

    os.makedirs(
        directory,
        exist_ok=True
    )

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        suffix=".tmp"
    )

    os.close(fd)

    try:

        joblib.dump(
            artifact,
            temp_path
        )

        os.replace(
            temp_path,
            output_path
        )

    except Exception:

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise


# ============================================================
# 7. MAIN CALIBRATION PIPELINE
# ============================================================

def calibrate_model(config=None):

    if config is None:
        config = CalibrationConfig()

    if config.method not in ("sigmoid", "isotonic"):
        raise ValueError(
            "Phương pháp calibration không hợp lệ"
        )

    print("Đang tải model...")

    artifact = load_model_artifact(
        config.model_path
    )

    print("Đang đọc dữ liệu...")

    df = pd.read_csv(
        config.data_path
    )

    print("Đang chuẩn bị validation data...")

    X_val, y_val = prepare_validation_data(
        df,
        artifact,
        config
    )

    print(
        f"Số mẫu validation: {len(X_val)}"
    )

    class_counts = y_val.value_counts()

    for class_name in EXPECTED_CLASSES:

        count = int(
            class_counts.get(class_name, 0)
        )

        print(
            f"{class_name}: {count}"
        )

        if count < config.min_samples_per_class:
            raise ValueError(
                f"Class {class_name} chỉ có "
                f"{count} mẫu, không đủ để calibration"
            )

    print("Đang dự đoán xác suất gốc...")

    model = artifact["model"]

    raw_probabilities = model.predict_proba(
        X_val
    )

    # Đánh giá trước calibration.
    before_metrics = calculate_calibration_metrics(
        y_val,
        raw_probabilities
    )

    print("Đang hiệu chỉnh xác suất...")

    calibrator = MulticlassCalibrator(
        method=config.method
    )

    calibrator.fit(
        raw_probabilities,
        y_val.to_numpy()
    )

    calibrated_probabilities = (
        calibrator.predict_proba(
            raw_probabilities
        )
    )

    # Đánh giá sau calibration.
    after_metrics = calculate_calibration_metrics(
        y_val,
        calibrated_probabilities
    )

    print("\n===== KẾT QUẢ CALIBRATION =====")

    print(
        "Log Loss trước:",
        before_metrics["log_loss"]
    )

    print(
        "Log Loss sau:",
        after_metrics["log_loss"]
    )

    print(
        "Brier trước:",
        before_metrics["mean_brier"]
    )

    print(
        "Brier sau:",
        after_metrics["mean_brier"]
    )

    # Lưu thông tin calibration vào artifact.
    artifact["calibrator"] = calibrator

    artifact["calibration_metadata"] = {

        "method": config.method,

        "classes": EXPECTED_CLASSES,

        "calibrated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "validation_samples": len(X_val),

        "metrics_before": before_metrics,

        "metrics_after": after_metrics,

        "target_column": config.target_column,

    }

    save_artifact_atomic(
        artifact,
        config.model_path
    )

    # Lưu metrics riêng.
    metrics = {

        "method": config.method,

        "metrics_before": before_metrics,

        "metrics_after": after_metrics,

        "validation_samples": len(X_val),

        "calibrated_at": datetime.now(
            timezone.utc
        ).isoformat(),

    }

    os.makedirs(
        os.path.dirname(
            config.output_metrics_path
        ),
        exist_ok=True
    )

    with open(
        config.output_metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nCalibration hoàn tất!")

    return artifact


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    config = CalibrationConfig()

    calibrate_model(config)
