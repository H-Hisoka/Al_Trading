
"""
historical_labels.py

Tạo nhãn swing lịch sử từ các swing đã được xác nhận.

Các nhãn:
- TOP
- BOTTOM
- NONE

Hỗ trợ:
- MINOR / MAJOR
- Lưu pivot_time riêng confirmation_time.
- Kiểm tra swing trùng lặp.
- Phát hiện nhãn mâu thuẫn.
- Giữ nguyên dữ liệu OHLC ban đầu.

Lưu ý:
Đây là nhãn lịch sử để huấn luyện và đánh giá,
không phải tín hiệu có thể biết tại pivot_time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


VALID_TYPES = {"TOP", "BOTTOM"}
VALID_SIZES = {"MINOR", "MAJOR"}


@dataclass
class HistoricalLabelConfig:
    timestamp_column: Optional[str] = None

    # Nếu True, yêu cầu mọi swing đều có confirmation_time.
    require_confirmation: bool = True

    # Kiểm tra timestamp của swing phải khớp nến.
    strict_timestamp_matching: bool = True

    # Tên các cột nhãn đầu ra.
    label_column: str = "historical_label"
    size_column: str = "historical_size"

    # Phiên bản nhãn.
    label_version: str = "1.0"


def _normalize_swings(
    swings: Any
) -> pd.DataFrame:
    """
    Chuẩn hóa danh sách swing thành DataFrame.
    """

    if isinstance(swings, pd.DataFrame):
        result = swings.copy()

    elif isinstance(swings, Sequence) and not isinstance(
        swings, (str, bytes)
    ):
        result = pd.DataFrame(list(swings))

    else:
        raise TypeError(
            "swings phải là DataFrame hoặc danh sách dictionary."
        )

    return result


def _validate_candles(
    candles: pd.DataFrame,
    config: HistoricalLabelConfig
) -> pd.DataFrame:
    """
    Kiểm tra và chuẩn hóa dữ liệu nến.
    """

    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles phải là pandas DataFrame.")

    df = candles.copy()

    if config.timestamp_column:
        if config.timestamp_column not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột "
                f"{config.timestamp_column}."
            )

        timestamps = pd.to_datetime(
            df[config.timestamp_column],
            errors="coerce"
        )

    else:
        timestamps = pd.to_datetime(
            df.index,
            errors="coerce"
        )

    if timestamps.isna().any():
        raise ValueError("Timestamp nến chứa giá trị không hợp lệ.")

    if timestamps.has_duplicates:
        raise ValueError("Dữ liệu nến có timestamp trùng lặp.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Nến phải được sắp xếp theo thời gian tăng dần."
        )

    df.index = pd.DatetimeIndex(timestamps)

    return df


def _validate_swings(
    swings: pd.DataFrame,
    config: HistoricalLabelConfig
) -> pd.DataFrame:
    """
    Kiểm tra danh sách swing trước khi tạo nhãn.
    """

    required = {
        "swing_type",
        "pivot_time",
        "swing_size",
    }

    if config.require_confirmation:
        required.add("confirmation_time")

    missing = required - set(swings.columns)

    if missing:
        raise ValueError(
            f"Swing thiếu các trường: {sorted(missing)}"
        )

    result = swings.copy()

    result["swing_type"] = (
        result["swing_type"].astype(str).str.upper()
    )

    result["swing_size"] = (
        result["swing_size"].astype(str).str.upper()
    )

    invalid_types = (
        set(result["swing_type"].dropna()) - VALID_TYPES
    )

    invalid_sizes = (
        set(result["swing_size"].dropna()) - VALID_SIZES
    )

    if invalid_types:
        raise ValueError(
            f"swing_type không hợp lệ: {invalid_types}"
        )

    if invalid_sizes:
        raise ValueError(
            f"swing_size không hợp lệ: {invalid_sizes}"
        )

    result["pivot_time"] = pd.to_datetime(
        result["pivot_time"],
        errors="coerce"
    )

    if result["pivot_time"].isna().any():
        raise ValueError("pivot_time không hợp lệ.")

    if config.require_confirmation:
        result["confirmation_time"] = pd.to_datetime(
            result["confirmation_time"],
            errors="coerce"
        )

        if result["confirmation_time"].isna().any():
            raise ValueError("confirmation_time không hợp lệ.")

        invalid_order = (
            result["confirmation_time"]
            < result["pivot_time"]
        )

        if invalid_order.any():
            raise ValueError(
                "confirmation_time không được trước pivot_time."
            )

    if "status" in result.columns:
        invalid_status = (
            result["status"].astype(str).str.upper()
            != "CONFIRMED"
        )

        if invalid_status.any():
            raise ValueError(
                "Danh sách chứa swing chưa được xác nhận."
            )

    return result


def _resolve_pivot_labels(
    swings: pd.DataFrame
) -> Dict[pd.Timestamp, Dict[str, Any]]:
    """
    Tạo bản đồ pivot_time -> nhãn.

    Nếu cùng một pivot có TOP và BOTTOM:
    báo lỗi vì đây là nhãn mâu thuẫn.

    Nếu cùng loại nhưng khác cấp độ:
    giữ cấp MAJOR nếu tồn tại.
    """

    grouped: Dict[pd.Timestamp, Dict[str, Any]] = {}

    for _, swing in swings.iterrows():

        pivot_time = swing["pivot_time"]
        swing_type = swing["swing_type"]
        swing_size = swing["swing_size"]

        if pivot_time not in grouped:
            grouped[pivot_time] = {
                "types": set(),
                "sizes": set(),
                "swing_ids": [],
                "confirmation_times": [],
            }

        item = grouped[pivot_time]

        item["types"].add(swing_type)
        item["sizes"].add(swing_size)

        if "swing_id" in swings.columns:
            item["swing_ids"].append(swing.get("swing_id"))

        if "confirmation_time" in swings.columns:
            item["confirmation_times"].append(
                swing.get("confirmation_time")
            )

    resolved = {}

    for pivot_time, item in grouped.items():

        if len(item["types"]) != 1:
            raise ValueError(
                f"Pivot {pivot_time} có nhãn TOP/BOTTOM mâu thuẫn."
            )

        swing_type = next(iter(item["types"]))

        swing_size = (
            "MAJOR"
            if "MAJOR" in item["sizes"]
            else "MINOR"
        )

        resolved[pivot_time] = {
            "label": swing_type,
            "size": swing_size,
            "swing_ids": item["swing_ids"],
            "confirmation_times": item["confirmation_times"],
        }

    return resolved


def create_historical_labels(
    candles: pd.DataFrame,
    swings: Any,
    config: Optional[HistoricalLabelConfig] = None
) -> pd.DataFrame:
    """
    Tạo nhãn lịch sử cho từng nến.

    Parameters
    ----------
    candles:
        DataFrame nến M5 hoặc timeframe khác.

    swings:
        Danh sách swing đã xác nhận từ SwingRegistry.

    config:
        Cấu hình tạo nhãn.

    Returns
    -------
    DataFrame:
        historical_label: TOP / BOTTOM / NONE
        historical_size: MINOR / MAJOR / NONE
        is_swing: True / False
        swing_confirmation_time: thời điểm xác nhận
        label_version: phiên bản nhãn
    """

    config = config or HistoricalLabelConfig()

    df = _validate_candles(candles, config)

    swing_df = _normalize_swings(swings)

    swing_df = _validate_swings(
        swing_df,
        config
    )

    pivot_map = _resolve_pivot_labels(swing_df)

    df[config.label_column] = "NONE"
    df[config.size_column] = "NONE"

    df["is_swing"] = False

    df["swing_confirmation_time"] = pd.NaT

    df["label_version"] = config.label_version

    unmatched_pivots = []

    for pivot_time, information in pivot_map.items():

        if pivot_time not in df.index:
            unmatched_pivots.append(pivot_time)
            continue

        df.at[
            pivot_time,
            config.label_column
        ] = information["label"]

        df.at[
            pivot_time,
            config.size_column
        ] = information["size"]

        df.at[
            pivot_time,
            "is_swing"
        ] = True

        confirmation_times = information[
            "confirmation_times"
        ]

        valid_times = [
            value for value in confirmation_times
            if pd.notna(value)
        ]

        if valid_times:
            df.at[
                pivot_time,
                "swing_confirmation_time"
            ] = max(valid_times)

    if unmatched_pivots and config.strict_timestamp_matching:
        raise ValueError(
            f"Có {len(unmatched_pivots)} pivot_time "
            "không khớp timestamp nến. "
            f"Ví dụ: {unmatched_pivots[:5]}"
        )

    return df


def create_historical_label_summary(
    labeled_data: pd.DataFrame,
    label_column: str = "historical_label",
    size_column: str = "historical_size"
) -> pd.DataFrame:
    """
    Thống kê số lượng nhãn lịch sử.
    """

    required = {label_column, size_column}

    missing = required - set(labeled_data.columns)

    if missing:
        raise ValueError(
            f"Thiếu cột: {sorted(missing)}"
        )

    summary = (
        labeled_data
        .groupby([label_column, size_column])
        .size()
        .rename("count")
        .reset_index()
    )

    summary["percentage"] = (
        summary["count"]
        / len(labeled_data)
        * 100
    )

    return summary.sort_values(
        "count",
        ascending=False
    ).reset_index(drop=True)
