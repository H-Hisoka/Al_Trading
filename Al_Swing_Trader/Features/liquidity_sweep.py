
# File: features/liquidity_sweep.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LiquiditySweepConfig:
    """Cấu hình nhận diện Liquidity Sweep."""

    atr_column: str = "atr_14"

    # Giá phải xuyên qua swing ít nhất một khoảng ATR nhỏ.
    min_sweep_atr: float = 0.05

    # Giới hạn độ xuyên tối đa để tránh nhầm với breakout mạnh.
    max_sweep_atr: float = 1.0

    # True: yêu cầu giá đóng cửa quay lại phía trong mức swing.
    require_reclaim: bool = True

    # True: dùng high/low để xác định xuyên mức.
    # False: dùng close.
    use_wick: bool = True

    # True: chỉ xét swing đã được xác nhận.
    require_confirmation: bool = True

    # Số swing gần nhất mỗi loại được theo dõi.
    max_active_swings: int = 30

    # Khoảng cách giữa giá đóng cửa và mức sweep được chuẩn hóa theo ATR.
    include_distance_features: bool = True


def _prepare_candles(
    candles: pd.DataFrame,
    config: LiquiditySweepConfig,
) -> pd.DataFrame:
    required = {
        "high",
        "low",
        "close",
        config.atr_column,
    }

    missing = required - set(candles.columns)

    if missing:
        raise ValueError(
            f"Thiếu các cột: {sorted(missing)}"
        )

    df = candles.copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[list(required)].isna().any().any():
        raise ValueError(
            "OHLC hoặc ATR chứa giá trị không hợp lệ"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Nến phải được sắp xếp theo thời gian tăng dần"
        )

    if (df["high"] < df["low"]).any():
        raise ValueError("Có nến high nhỏ hơn low")

    return df


def _prepare_swings(
    swings: pd.DataFrame,
    config: LiquiditySweepConfig,
) -> pd.DataFrame:
    required = {
        "swing_type",
        "pivot_time",
        "pivot_price",
    }

    missing = required - set(swings.columns)

    if missing:
        raise ValueError(
            f"Thiếu các cột swing: {sorted(missing)}"
        )

    df = swings.copy()

    df["swing_type"] = (
        df["swing_type"]
        .astype(str)
        .str.upper()
    )

    df = df[
        df["swing_type"].isin(["TOP", "BOTTOM"])
    ].copy()

    df["pivot_time"] = pd.to_datetime(
        df["pivot_time"]
    )

    df["pivot_price"] = pd.to_numeric(
        df["pivot_price"],
        errors="coerce",
    )

    if "confirmation_time" not in df.columns:
        if config.require_confirmation:
            raise ValueError(
                "Swings cần có confirmation_time"
            )

        df["confirmation_time"] = df["pivot_time"]

    df["confirmation_time"] = pd.to_datetime(
        df["confirmation_time"]
    )

    if "status" in df.columns:
        confirmed = df["status"].astype(str).str.upper().isin(
            ["CONFIRMED", "CONFIRM", "COMPLETED"]
        )
        df = df[confirmed]

    if df["pivot_price"].isna().any():
        raise ValueError("pivot_price không hợp lệ")

    return df.sort_values(
        ["confirmation_time", "pivot_time"]
    ).reset_index(drop=True)


def _make_swing_id(
    swing: dict,
    fallback_id: int,
) -> str:
    """Tạo ID ổn định cho swing nếu dữ liệu chưa có swing_id."""
    if swing.get("swing_id") is not None:
        return str(swing["swing_id"])

    return (
        f'{swing["swing_type"]}_'
        f'{swing["pivot_time"]}_'
        f'{fallback_id}'
    )


def detect_liquidity_sweeps(
    candles: pd.DataFrame,
    swings: pd.DataFrame,
    config: Optional[LiquiditySweepConfig] = None,
) -> pd.DataFrame:
    """
    Phát hiện Buy-side và Sell-side Liquidity Sweep.

    BUY_SIDE_SWEEP:
        High vượt swing TOP.
        Close quay xuống dưới swing TOP.

    SELL_SIDE_SWEEP:
        Low xuyên swing BOTTOM.
        Close quay lên trên swing BOTTOM.

    Kết quả được ghi nhận tại nến phát hiện sweep,
    không ghi ngược về pivot_time.
    """
    if config is None:
        config = LiquiditySweepConfig()

    if config.min_sweep_atr < 0:
        raise ValueError("min_sweep_atr không được âm")

    if config.max_sweep_atr <= config.min_sweep_atr:
        raise ValueError(
            "max_sweep_atr phải lớn hơn min_sweep_atr"
        )

    if config.max_active_swings < 1:
        raise ValueError("max_active_swings phải >= 1")

    df = _prepare_candles(candles, config)
    swing_df = _prepare_swings(swings, config)

    result = pd.DataFrame(index=df.index)

    result["buy_side_sweep"] = 0
    result["sell_side_sweep"] = 0

    result["liquidity_sweep"] = 0
    result["sweep_direction"] = "NONE"

    result["sweep_level"] = np.nan
    result["sweep_depth_atr"] = np.nan
    result["sweep_wick_ratio"] = np.nan

    result["swept_swing_id"] = None
    result["swept_pivot_time"] = pd.NaT

    result["active_buy_side_levels"] = 0
    result["active_sell_side_levels"] = 0

    swing_records = swing_df.to_dict("records")

    pointer = 0

    active_tops = []
    active_bottoms = []

    for candle_time, candle in df.iterrows():

        # Chỉ thêm swing đã được xác nhận tại thời điểm hiện tại.
        while (
            pointer < len(swing_records)
            and swing_records[pointer]["confirmation_time"]
            <= candle_time
        ):
            swing = swing_records[pointer]

            swing["internal_id"] = _make_swing_id(
                swing,
                pointer,
            )

            if swing["swing_type"] == "TOP":
                active_tops.append(swing)

            else:
                active_bottoms.append(swing)

            pointer += 1

        # Giới hạn số mức thanh khoản đang theo dõi.
        active_tops = active_tops[
            -config.max_active_swings:
        ]

        active_bottoms = active_bottoms[
            -config.max_active_swings:
        ]

        result.at[
            candle_time,
            "active_buy_side_levels"
        ] = len(active_tops)

        result.at[
            candle_time,
            "active_sell_side_levels"
        ] = len(active_bottoms)

        atr = float(candle[config.atr_column])

        if atr <= 0:
            continue

        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        upper_price = high if config.use_wick else close
        lower_price = low if config.use_wick else close

        candidates = []

        # Kiểm tra Buy-side sweep.
        for swing in active_tops:
            level = float(swing["pivot_price"])

            penetration = upper_price - level
            penetration_atr = penetration / atr

            crossed = (
                penetration_atr >= config.min_sweep_atr
                and penetration_atr <= config.max_sweep_atr
            )

            reclaimed = (
                close < level
                if config.require_reclaim
                else True
            )

            if crossed and reclaimed:
                candidates.append(
                    {
                        "direction": "BUY_SIDE_SWEEP",
                        "level": level,
                        "depth_atr": penetration_atr,
                        "swing": swing,
                    }
                )

        # Kiểm tra Sell-side sweep.
        for swing in active_bottoms:
            level = float(swing["pivot_price"])

            penetration = level - lower_price
            penetration_atr = penetration / atr

            crossed = (
                penetration_atr >= config.min_sweep_atr
                and penetration_atr <= config.max_sweep_atr
            )

            reclaimed = (
                close > level
                if config.require_reclaim
                else True
            )

            if crossed and reclaimed:
                candidates.append(
                    {
                        "direction": "SELL_SIDE_SWEEP",
                        "level": level,
                        "depth_atr": penetration_atr,
                        "swing": swing,
                    }
                )

        if not candidates:
            continue

        # Nếu nhiều mức bị quét cùng lúc, chọn mức có
        # độ xuyên ATR nhỏ nhất để tránh chọn mức quá xa.
        selected = min(
            candidates,
            key=lambda item: item["depth_atr"],
        )

        direction = selected["direction"]
        swing = selected["swing"]
        level = selected["level"]

        if direction == "BUY_SIDE_SWEEP":
            result.at[candle_time, "buy_side_sweep"] = 1
            result.at[candle_time, "sweep_direction"] = "BUY"

            wick_size = max(0.0, high - max(close, level))

        else:
            result.at[candle_time, "sell_side_sweep"] = 1
            result.at[candle_time, "sweep_direction"] = "SELL"

            wick_size = max(0.0, min(close, level) - low)

        result.at[candle_time, "liquidity_sweep"] = 1

        result.at[
            candle_time,
            "sweep_level"
        ] = level

        result.at[
            candle_time,
            "sweep_depth_atr"
        ] = selected["depth_atr"]

        candle_range = high - low

        result.at[
            candle_time,
            "sweep_wick_ratio"
        ] = (
            wick_size / candle_range
            if candle_range > 0
            else np.nan
        )

        result.at[
            candle_time,
            "swept_swing_id"
        ] = swing["internal_id"]

        result.at[
            candle_time,
            "swept_pivot_time"
        ] = swing["pivot_time"]

    return result


def summarize_liquidity_sweeps(
    sweeps: pd.DataFrame,
) -> dict:
    """Thống kê các sự kiện Liquidity Sweep."""
    if sweeps.empty:
        return {
            "total_sweeps": 0,
            "buy_side_sweeps": 0,
            "sell_side_sweeps": 0,
        }

    return {
        "total_sweeps": int(
            sweeps["liquidity_sweep"].sum()
        ),

        "buy_side_sweeps": int(
            sweeps["buy_side_sweep"].sum()
        ),

        "sell_side_sweeps": int(
            sweeps["sell_side_sweep"].sum()
        ),
    }
