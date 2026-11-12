
"""
File: backtest/metrics.py

Đánh giá mô hình AI Swing Trading.

Các chỉ số:
- Accuracy
- Precision / Recall / F1
- Macro F1 / Weighted F1
- Confusion Matrix
- PR-AUC
- Brier Score
- Log Loss
- Calibration report
- Swing event statistics

Input tương thích với backtest/simulator.py.

Không tự tạo số liệu hoặc giả định hiệu suất.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    log_loss,
    classification_report,
)


logger = logging.getLogger(__name__)


CLASS_ORDER = [
    "TOP_SOON",
    "BOTTOM_SOON",
    "NONE",
]

PROBABILITY_COLUMNS = {
    "TOP_SOON": "p_top_soon",
    "BOTTOM_SOON": "p_bottom_soon",
    "NONE": "p_none",
}


def _validate_results(results: pd.DataFrame) -> None:
    """Kiểm tra dữ liệu đầu vào."""

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results phải là DataFrame.")

    if results.empty:
        raise ValueError("Không có kết quả để đánh giá.")

    required = [
        "actual_label",
        "predicted_class",
        "p_top_soon",
        "p_bottom_soon",
        "p_none",
    ]

    missing = [
        column
        for column in required
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu các cột cần thiết: {missing}"
        )


def _get_probability_matrix(
    results: pd.DataFrame,
) -> np.ndarray:
    """Tạo ma trận xác suất theo thứ tự CLASS_ORDER."""

    probabilities = results[
        [
            PROBABILITY_COLUMNS[label]
            for label in CLASS_ORDER
        ]
    ].to_numpy(dtype=float)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Ma trận xác suất chứa NaN hoặc Inf."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Xác suất phải nằm trong khoảng [0, 1]."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Tổng xác suất mỗi mẫu phải bằng 1."
        )

    return probabilities


def _clean_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Loại bỏ những mẫu không có kết quả dự đoán hợp lệ."""

    _validate_results(results)

    clean = results.copy()

    clean = clean.dropna(
        subset=[
            "actual_label",
            "predicted_class",
            *PROBABILITY_COLUMNS.values(),
        ]
    )

    valid_classes = set(CLASS_ORDER)

    clean = clean[
        clean["actual_label"].isin(valid_classes)
        & clean["predicted_class"].isin(valid_classes)
    ]

    if clean.empty:
        raise ValueError(
            "Không còn mẫu hợp lệ sau khi làm sạch."
        )

    return clean


def calculate_classification_metrics(
    results: pd.DataFrame,
) -> dict:
    """
    Tính các chỉ số phân loại.

    Returns
    -------
    dict:
        Accuracy, Precision, Recall, F1,
        Macro F1, Weighted F1,
        Classification report, Confusion matrix.
    """

    clean = _clean_results(results)

    y_true = clean["actual_label"].to_numpy()
    y_pred = clean["predicted_class"].to_numpy()

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        zero_division=0,
        output_dict=True,
    )

    per_class = {}

    for i, label in enumerate(CLASS_ORDER):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "sample_count": int(len(clean)),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),

        "weighted_f1": float(
            np.average(
                f1,
                weights=support,
            )
        ) if support.sum() > 0 else 0.0,

        "per_class": per_class,

        "confusion_matrix": {
            "labels": CLASS_ORDER,
            "matrix": matrix.tolist(),
        },

        "classification_report": report,
    }


def calculate_probability_metrics(
    results: pd.DataFrame,
) -> dict:
    """
    Đánh giá chất lượng xác suất dự báo.

    Brier Score:
        Trung bình bình phương sai số giữa
        xác suất dự báo và nhãn one-hot.

    Log Loss:
        Đánh giá mức độ phù hợp của xác suất
        với lớp thực tế.
    """

    clean = _clean_results(results)

    y_true = clean["actual_label"].to_numpy()

    probabilities = _get_probability_matrix(clean)

    class_to_index = {
        label: index
        for index, label in enumerate(CLASS_ORDER)
    }

    y_encoded = np.array([
        class_to_index[label]
        for label in y_true
    ])

    y_one_hot = np.eye(len(CLASS_ORDER))[y_encoded]

    # Multiclass Brier Score.
    brier_score = float(
        np.mean(
            np.sum(
                (probabilities - y_one_hot) ** 2,
                axis=1,
            )
        )
    )

    # Log Loss.
    calculated_log_loss = float(
        log_loss(
            y_encoded,
            probabilities,
            labels=np.arange(len(CLASS_ORDER)),
        )
    )

    # PR-AUC cho từng lớp.
    pr_auc = {}

    for index, label in enumerate(CLASS_ORDER):

        binary_true = (
            y_encoded == index
        ).astype(int)

        if len(np.unique(binary_true)) < 2:
            pr_auc[label] = None
            continue

        pr_auc[label] = float(
            average_precision_score(
                binary_true,
                probabilities[:, index],
            )
        )

    return {
        "brier_score": brier_score,
        "log_loss": calculated_log_loss,
        "pr_auc": pr_auc,
        "macro_pr_auc": (
            float(np.mean([
                value
                for value in pr_auc.values()
                if value is not None
            ]))
            if any(
                value is not None
                for value in pr_auc.values()
            )
            else None
        ),
    }


def calculate_calibration_report(
    results: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Kiểm tra độ hiệu chỉnh xác suất.

    Chia dự báo thành các nhóm xác suất.
    So sánh:
        Mean predicted probability
        Actual event frequency

    Ví dụ:
        Nhóm dự báo TOP khoảng 70%
        Thì tần suất TOP thực tế có gần 70% không?
    """

    if n_bins < 2:
        raise ValueError("n_bins phải >= 2.")

    clean = _clean_results(results)

    probability_matrix = _get_probability_matrix(clean)

    calibration_rows = []

    for index, label in enumerate(CLASS_ORDER):

        probabilities = probability_matrix[:, index]

        actual = (
            clean["actual_label"].to_numpy() == label
        ).astype(int)

        bin_edges = np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        )

        bin_ids = np.digitize(
            probabilities,
            bin_edges[1:-1],
            right=False,
        )

        for bin_id in range(n_bins):

            mask = bin_ids == bin_id

            if not mask.any():
                continue

            calibration_rows.append({
                "class": label,
                "bin": bin_id,
                "sample_count": int(mask.sum()),
                "mean_predicted_probability": float(
                    probabilities[mask].mean()
                ),
                "actual_frequency": float(
                    actual[mask].mean()
                ),
                "calibration_error": float(
                    abs(
                        probabilities[mask].mean()
                        - actual[mask].mean()
                    )
                ),
            })

    return pd.DataFrame(calibration_rows)


def calculate_swing_metrics(
    results: pd.DataFrame,
) -> dict:
    """
    Đánh giá riêng khả năng nhận diện swing.

    TOP_SOON và BOTTOM_SOON được đánh giá riêng.
    NONE không được tính là một swing.

    Đây là đánh giá theo từng mẫu thời gian,
    không phải ghép cặp pivot với sự kiện thực tế.
    """

    clean = _clean_results(results)

    actual = clean["actual_label"]
    predicted = clean["predicted_class"]

    output = {}

    for label in ["TOP_SOON", "BOTTOM_SOON"]:

        true_positive = int(
            ((actual == label) & (predicted == label)).sum()
        )

        false_positive = int(
            ((actual != label) & (predicted == label)).sum()
        )

        false_negative = int(
            ((actual == label) & (predicted != label)).sum()
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        output[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    output["total_actual_swings"] = int(
        actual.isin(
            ["TOP_SOON", "BOTTOM_SOON"]
        ).sum()
    )

    output["total_predicted_swings"] = int(
        predicted.isin(
            ["TOP_SOON", "BOTTOM_SOON"]
        ).sum()
    )

    return output


def evaluate(
    results: pd.DataFrame,
    n_bins: int = 10,
) -> dict:
    """
    Chạy toàn bộ quy trình đánh giá.

    Input:
        Output DataFrame từ simulator.run_backtest().

    Returns:
        Dictionary chứa toàn bộ metrics.
    """

    clean = _clean_results(results)

    logger.info(
        "Bắt đầu đánh giá %d mẫu.",
        len(clean),
    )

    classification = calculate_classification_metrics(
        clean
    )

    probability = calculate_probability_metrics(
        clean
    )

    calibration = calculate_calibration_report(
        clean,
        n_bins=n_bins,
    )

    swing = calculate_swing_metrics(
        clean
    )

    report = {
        "sample_count": len(clean),
        "classification": classification,
        "probability": probability,
        "swing": swing,
        "calibration": calibration,
    }

    logger.info("Hoàn thành đánh giá model.")

    return report


def save_metrics_report(
    report: dict,
    output_path: str,
) -> None:
    """
    Lưu báo cáo metrics.

    JSON chứa các chỉ số.
    Calibration được lưu riêng dưới CSV.
    """

    import json
    from pathlib import Path

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration = report.get("calibration")

    serializable = {
        key: value
        for key, value in report.items()
        if key != "calibration"
    }

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable,
            file,
            ensure_ascii=False,
            indent=4,
            allow_nan=False,
        )

    if isinstance(calibration, pd.DataFrame):
        calibration_path = output.with_name(
            output.stem + "_calibration.csv"
        )

        calibration.to_csv(
            calibration_path,
            index=False,
        )

    logger.info(
        "Đã lưu metrics report: %s",
        output,
    )
