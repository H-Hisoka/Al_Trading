
"""
future_event_labels.py

Tạo nhãn dự báo sự kiện swing trong tương lai.

Các lớp:
- TOP_SOON
- BOTTOM_SOON
- NONE

Hỗ trợ:
- Horizon 3 / 6 / 12 nến hoặc tùy chỉnh.
- MINOR / MAJOR.
- Chọn sự kiện gần nhất trong tương lai.
- Không gán NONE cho vùng dữ liệu thiếu tương lai.
- Lưu thời điểm pivot và confirmation riêng biệt.

Lưu ý:
Future labels được phép sử dụng dữ liệu tương lai
để tạo target huấn luyện.

Tuyệt đối không sử dụng các cột target làm feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


VALID_TYPES = {"TOP", "BOTTOM"}
VALID_SIZES = {"MINOR", "MAJOR"}


@dataclass
class FutureLabelConfig:
    # Số nến tương lai được xét.
    horizon_bars: int = 6

    timestamp_column: Optional[str] = None

    # Chỉ sử dụng swing đã xác nhận.
    require_confirmation: bool = True

    # Xử lý nhiều swing tại cùng một pivot.
    prefer_major: bool = True

    # Báo lỗi khi TOP/BOTTOM cùng pivot.
    strict_conflict_check: bool = True

    # Tên các cột đầu ra.
    target_column: str = "future_event_label"
    size_column: str = "future_event_size"

    # Phiên bản nhãn.
    label_version: str = "1.0"


def _normalize_swings(
    swings: Any
) -> pd.DataFrame:
    """
    Chuẩn hóa swing đầu vào.
    """

    if isinstance(swings, pd.DataFrame):
        return swings.copy()

    if isinstance(swings, Sequence) and not isinstance(
        swings, (str, bytes)
    ):
        return pd.DataFrame(list(swings))

    raise TypeError(
        "swings phải là DataFrame hoặc danh sách dictionary."
    )


def _prepare_candles(
    candles: pd.DataFrame,
    config: FutureLabelConfig
) -> pd.DataFrame:
    """
    Chuẩn hóa timestamp và kiểm tra dữ liệu nến.
    """

    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles phải là DataFrame.")

    if config.horizon_bars < 1:
        raise ValueError("horizon_bars phải >= 1.")

    df = candles.copy()

    if config.timestamp_column:
        if config.timestamp_column not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột {config.timestamp_column}."
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
        raise ValueError("Timestamp nến không hợp lệ.")

    if timestamps.has_duplicates:
        raise ValueError("Timestamp nến bị trùng.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Nến phải được sắp xếp tăng dần theo thời gian."
        )

    df.index = pd.DatetimeIndex(timestamps)

    return df


def _prepare_swings(
    swings: Any,
    config: FutureLabelConfig
) -> pd.DataFrame:
    """
    Kiểm tra swing và chỉ giữ các swing hợp lệ.
    """

    df = _normalize_swings(swings)

    required = {
        "swing_type",
        "swing_size",
        "pivot_time",
    }

    if config.require_confirmation:
        required.add("confirmation_time")

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Swing thiếu trường: {sorted(missing)}"
        )

    df["swing_type"] = (
        df["swing_type"].astype(str).str.upper()
    )

    df["swing_size"] = (
        df["swing_size"].astype(str).str.upper()
    )

    invalid_types = (
        set(df["swing_type"].dropna()) - VALID_TYPES
    )

    invalid_sizes = (
        set(df["swing_size"].dropna()) - VALID_SIZES
    )

    if invalid_types:
        raise ValueError(
            f"swing_type không hợp lệ: {invalid_types}"
        )

    if invalid_sizes:
        raise ValueError(
            f"swing_size không hợp lệ: {invalid_sizes}"
        )

    df["pivot_time"] = pd.to_datetime(
        df["pivot_time"],
        errors="coerce"
    )

    if df["pivot_time"].isna().any():
        raise ValueError("pivot_time không hợp lệ.")

    if config.require_confirmation:

        df["confirmation_time"] = pd.to_datetime(
            df["confirmation_time"],
            errors="coerce"
        )

        if df["confirmation_time"].isna().any():
            raise ValueError("confirmation_time không hợp lệ.")

        if (
            df["confirmation_time"] < df["pivot_time"]
        ).any():
            raise ValueError(
                "confirmation_time không được trước pivot_time."
            )

    if "status" in df.columns:
        confirmed = (
            df["status"].astype(str).str.upper()
            == "CONFIRMED"
        )

        if not confirmed.all():
            raise ValueError(
                "Future labels chỉ nhận swing đã xác nhận."
            )

    return df


def _build_event_map(
    swings: pd.DataFrame,
    candles: pd.DataFrame,
    config: FutureLabelConfig
) -> Dict[int, Dict[str, Any]]:
    """
    Ánh xạ swing vào vị trí nến.

    Key:
        Vị trí nến chứa pivot.

    Value:
        Thông tin swing tương ứng.

    Nếu nhiều swing cùng pivot:
        - Cùng hướng: ưu tiên MAJOR nếu được cấu hình.
        - Khác hướng: báo lỗi.
    """

    timestamp_to_position = {
        timestamp: position
        for position, timestamp in enumerate(candles.index)
    }

    grouped: Dict[int, list] = {}

    for _, swing in swings.iterrows():

        pivot_time = swing["pivot_time"]

        if pivot_time not in timestamp_to_position:
            continue

        position = timestamp_to_position[pivot_time]

        grouped.setdefault(position, []).append(
            swing.to_dict()
        )

    event_map = {}

    for position, events in grouped.items():

        types = {
            event["swing_type"]
            for event in events
        }

        if len(types) > 1 and config.strict_conflict_check:
            raise ValueError(
                f"Pivot tại {candles.index[position]} "
                "có cả TOP và BOTTOM."
            )

        if len(types) > 1:
            continue

        if config.prefer_major:
            events.sort(
                key=lambda event: (
                    event["swing_size"] == "MAJOR"
                ),
                reverse=True
            )

        selected = events[0]

        event_map[position] = {
            "label": (
                f"{selected['swing_type']}_SOON"
            ),
            "swing_type": selected["swing_type"],
            "swing_size": selected["swing_size"],
            "pivot_time": selected["pivot_time"],
            "confirmation_time": selected.get(
                "confirmation_time",
                pd.NaT
            ),
            "swing_id": selected.get("swing_id"),
        }

    return event_map


def create_future_event_labels(
    candles: pd.DataFrame,
    swings: Any,
    config: Optional[FutureLabelConfig] = None
) -> pd.DataFrame:
    """
    Tạo target dự báo sự kiện swing tương lai.

    Ví dụ:
        horizon_bars = 6

    Với nến tại vị trí i:
        Xét các pivot tại i+1 đến i+6.

    Nếu sự kiện gần nhất là TOP:
        TOP_SOON

    Nếu sự kiện gần nhất là BOTTOM:
        BOTTOM_SOON

    Nếu không có sự kiện:
        NONE

    Những nến cuối không đủ horizon:
        NaN (chưa đủ thông tin để kết luận).

    Lưu ý:
        Target là nhãn retrospective.
        Không được dùng target làm feature realtime.
    """

    config = config or FutureLabelConfig()

    df = _prepare_candles(
        candles,
        config
    )

    swing_df = _prepare_swings(
        swings,
        config
    )

    event_map = _build_event_map(
        swing_df,
        df,
        config
    )

    n = len(df)
    horizon = config.horizon_bars

    # Sử dụng object để hỗ trợ giá trị thiếu pd.NA.
    targets = pd.Series(
        [pd.NA] * n,
        index=df.index,
        dtype="object"
    )

    sizes = pd.Series(
        [pd.NA] * n,
        index=df.index,
        dtype="object"
    )

    event_times = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="object"
    )

    confirmation_times = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="object"
    )

    bars_to_event = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    event_ids = pd.Series(
        [pd.NA] * n,
        index=df.index,
        dtype="object"
    )

    # Không xử lý phần cuối thiếu horizon tương lai.
    last_valid_position = n - horizon - 1

    for i in range(max(0, last_valid_position + 1)):

        future_events = []

        start = i + 1
        end = min(i + horizon, n - 1)

        for future_position in range(start, end + 1):

            if future_position in event_map:
                event = event_map[future_position]

                future_events.append(
                    (future_position, event)
                )

        if not future_events:
            targets.iloc[i] = "NONE"
            sizes.iloc[i] = "NONE"
            continue

        # Chọn sự kiện xuất hiện sớm nhất.
        future_events.sort(
            key=lambda item: item[0]
        )

        first_position, first_event = future_events[0]

        targets.iloc[i] = first_event["label"]

        sizes.iloc[i] = first_event["swing_size"]

        event_times.iloc[i] = first_event["pivot_time"]

        confirmation_times.iloc[i] = (
            first_event["confirmation_time"]
        )

        bars_to_event.iloc[i] = (
            first_position - i
        )

        event_ids.iloc[i] = first_event["swing_id"]

    df[config.target_column] = targets

    df[config.size_column] = sizes

    df["future_event_time"] = event_times

    df["future_confirmation_time"] = confirmation_times

    df["bars_to_future_event"] = bars_to_event

    df["future_swing_id"] = event_ids

    df["future_horizon_bars"] = horizon

    df["future_label_version"] = config.label_version

    return df


def create_multi_horizon_labels(
    candles: pd.DataFrame,
    swings: Any,
    horizons: tuple = (3, 6, 12),
    config: Optional[FutureLabelConfig] = None
) -> pd.DataFrame:
    """
    Tạo target riêng cho nhiều horizon.

    Ví dụ:
        TOP_SOON_3
        TOP_SOON_6
        TOP_SOON_12

    Mỗi horizon có nhãn riêng.
    """

    if not horizons:
        raise ValueError("Phải cung cấp ít nhất một horizon.")

    if len(set(horizons)) != len(horizons):
        raise ValueError("Horizon không được trùng nhau.")

    if any(h < 1 for h in horizons):
        raise ValueError("Mọi horizon phải >= 1.")

    config = config or FutureLabelConfig()

    result = candles.copy()

    for horizon in horizons:

        current_config = FutureLabelConfig(
            horizon_bars=horizon,
            timestamp_column=config.timestamp_column,
            require_confirmation=config.require_confirmation,
            prefer_major=config.prefer_major,
            strict_conflict_check=config.strict_conflict_check,
            target_column=f"future_event_label_{horizon}",
            size_column=f"future_event_size_{horizon}",
            label_version=config.label_version
        )

        labeled = create_future_event_labels(
            candles,
            swings,
            current_config
        )

        columns = [
            f"future_event_label_{horizon}",
            f"future_event_size_{horizon}",
            "future_event_time",
            "future_confirmation_time",
            "bars_to_future_event",
            "future_swing_id",
        ]

        for column in columns:
            if column in result.columns:
                result.drop(columns=column, inplace=True)

            result[column] = labeled[column]

    return result


def summarize_future_labels(
    labeled_data: pd.DataFrame,
    target_column: str = "future_event_label"
) -> pd.DataFrame:
    """
    Thống kê phân bố các lớp target.

    Các dòng chưa đủ horizon sẽ không được tính.
    """

    if target_column not in labeled_data.columns:
        raise ValueError(
            f"Không tìm thấy cột {target_column}."
        )

    valid_targets = labeled_data[target_column].dropna()

    if valid_targets.empty:
        raise ValueError("Không có target hợp lệ.")

    summary = (
        valid_targets
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"]
        / len(valid_targets)
        * 100
    )

    summary["valid_samples"] = len(valid_targets)

    summary["excluded_samples"] = (
        len(labeled_data) - len(valid_targets)
    )

    return summary
