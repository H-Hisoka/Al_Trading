
# File: features/market_structure.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MarketStructureConfig:
    """
    Cấu hình nhận diện cấu trúc thị trường.
    """

    timestamp_column: Optional[str] = None

    # Chỉ sử dụng swing đã xác nhận.
    require_confirmation: bool = True

    # Số swing gần nhất dùng để xác định trạng thái cấu trúc.
    structure_lookback: int = 10

    # Khoảng cách tối thiểu để coi là phá vỡ cấu trúc.
    # 0.0 nghĩa là chỉ cần vượt qua mức swing.
    break_buffer_atr: float = 0.0

    atr_column: str = "atr_14"

    # Nếu True, chỉ xác nhận BOS/CHoCH khi nến đóng cửa vượt mức swing.
    use_close_for_break: bool = True

    # Nếu True, một mức swing chỉ được phá vỡ một lần.
    one_break_per_level: bool = True


def _prepare_candles(
    candles: pd.DataFrame,
    config: MarketStructureConfig,
) -> pd.DataFrame:
    required = {"high", "low", "close"}

    missing = required - set(candles.columns)

    if missing:
        raise ValueError(
            f"Thiếu cột nến: {sorted(missing)}"
        )

    df = candles.copy()

    if config.timestamp_column:
        if config.timestamp_column not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột {config.timestamp_column}"
            )

        df[config.timestamp_column] = pd.to_datetime(
            df[config.timestamp_column]
        )

        df = df.set_index(config.timestamp_column)

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Nến phải được sắp xếp theo thời gian tăng dần"
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[list(required)].isna().any().any():
        raise ValueError("Dữ liệu OHLC không hợp lệ")

    if (df["high"] < df["low"]).any():
        raise ValueError("Có nến high nhỏ hơn low")

    return df


def _prepare_swings(
    swings: pd.DataFrame,
    config: MarketStructureConfig,
) -> pd.DataFrame:
    required = {
        "swing_type",
        "pivot_time",
        "pivot_price",
    }

    missing = required - set(swings.columns)

    if missing:
        raise ValueError(
            f"Thiếu cột swing: {sorted(missing)}"
        )

    df = swings.copy()

    df["pivot_time"] = pd.to_datetime(df["pivot_time"])

    if "confirmation_time" not in df.columns:
        if config.require_confirmation:
            raise ValueError(
                "Swings cần có confirmation_time"
            )

        df["confirmation_time"] = df["pivot_time"]

    df["confirmation_time"] = pd.to_datetime(
        df["confirmation_time"]
    )

    df["pivot_price"] = pd.to_numeric(
        df["pivot_price"],
        errors="coerce",
    )

    df["swing_type"] = (
        df["swing_type"]
        .astype(str)
        .str.upper()
    )

    df = df[
        df["swing_type"].isin(["TOP", "BOTTOM"])
    ].copy()

    if "status" in df.columns:
        confirmed = df["status"].astype(str).str.upper().isin(
            ["CONFIRMED", "CONFIRM", "COMPLETED"]
        )
        df = df[confirmed]

    if df["pivot_price"].isna().any():
        raise ValueError("pivot_price có giá trị không hợp lệ")

    # Quan trọng: cấu trúc chỉ được biết từ confirmation_time.
    df = df.sort_values(
        ["confirmation_time", "pivot_time"]
    ).reset_index(drop=True)

    return df


def _get_atr(
    candle: pd.Series,
    config: MarketStructureConfig,
) -> float:
    value = candle.get(config.atr_column, np.nan)

    if pd.isna(value) or value < 0:
        return 0.0

    return float(value)


def _classify_swing_sequence(
    swings: pd.DataFrame,
) -> dict:
    """
    Phân loại HH, HL, LH, LL dựa trên các swing cùng loại.

    TOP:
        Cao hơn TOP trước = HH
        Thấp hơn TOP trước = LH

    BOTTOM:
        Cao hơn BOTTOM trước = HL
        Thấp hơn BOTTOM trước = LL
    """
    previous_top = None
    previous_bottom = None

    classifications = []

    for _, swing in swings.iterrows():
        swing_type = swing["swing_type"]
        price = float(swing["pivot_price"])

        label = None

        if swing_type == "TOP":
            if previous_top is not None:
                label = "HH" if price > previous_top else "LH"

            previous_top = price

        elif swing_type == "BOTTOM":
            if previous_bottom is not None:
                label = "HL" if price > previous_bottom else "LL"

            previous_bottom = price

        classifications.append(label)

    result = swings.copy()
    result["structure_label"] = classifications

    return result


def _determine_trend(
    recent_swings: pd.DataFrame,
) -> str:
    """
    Xác định xu hướng từ các nhãn swing gần nhất.
    """
    labels = recent_swings["structure_label"].dropna().tolist()

    if len(labels) < 2:
        return "UNKNOWN"

    recent = labels[-4:]

    bullish = recent.count("HH") + recent.count("HL")
    bearish = recent.count("LH") + recent.count("LL")

    if bullish >= 2 and bullish > bearish:
        return "BULLISH"

    if bearish >= 2 and bearish > bullish:
        return "BEARISH"

    return "SIDEWAYS"


def detect_market_structure(
    candles: pd.DataFrame,
    swings: pd.DataFrame,
    config: Optional[MarketStructureConfig] = None,
) -> pd.DataFrame:
    """
    Tạo dữ liệu cấu trúc thị trường theo từng nến.

    Kết quả:
    - market_trend
    - last_swing_high
    - last_swing_low
    - last_swing_high_time
    - last_swing_low_time
    - structure_label
    - bos
    - choch
    - structure_event
    - broken_level

    BOS/CHoCH chỉ được ghi nhận tại nến phá vỡ.
    """

    if config is None:
        config = MarketStructureConfig()

    df = _prepare_candles(candles, config)
    swing_df = _prepare_swings(swings, config)

    result = df.copy()

    result["market_trend"] = "UNKNOWN"
    result["last_swing_high"] = np.nan
    result["last_swing_low"] = np.nan

    result["last_swing_high_time"] = pd.NaT
    result["last_swing_low_time"] = pd.NaT

    result["bos"] = 0
    result["choch"] = 0
    result["structure_event"] = "NONE"
    result["broken_level"] = np.nan

    result["latest_structure_label"] = None

    active_swings = []

    last_high = None
    last_low = None

    previous_trend = "UNKNOWN"

    # Lưu mức đã bị phá để tránh ghi nhận lặp.
    broken_levels = set()

    swing_records = swing_df.to_dict("records")
    swing_pointer = 0

    for candle_time, candle in result.iterrows():

        # Chỉ đưa swing vào hệ thống khi đã đến confirmation_time.
        while (
            swing_pointer < len(swing_records)
            and swing_records[swing_pointer]["confirmation_time"]
            <= candle_time
        ):
            swing = swing_records[swing_pointer]

            active_swings.append(swing)

            if swing["swing_type"] == "TOP":
                last_high = swing

            elif swing["swing_type"] == "BOTTOM":
                last_low = swing

            swing_pointer += 1

        # Cập nhật nhãn HH/HL/LH/LL.
        active_df = pd.DataFrame(active_swings)

        if not active_df.empty:
            active_df = _classify_swing_sequence(active_df)

            recent = active_df.tail(
                config.structure_lookback
            )

            trend = _determine_trend(recent)

            latest_label = (
                active_df["structure_label"].dropna().iloc[-1]
                if active_df["structure_label"].notna().any()
                else None
            )
        else:
            trend = "UNKNOWN"
            latest_label = None

        # Kiểm tra phá vỡ swing high/low đã được xác nhận.
        if last_high is not None:
            result.at[
                candle_time,
                "last_swing_high"
            ] = float(last_high["pivot_price"])

            result.at[
                candle_time,
                "last_swing_high_time"
            ] = last_high["pivot_time"]

        if last_low is not None:
            result.at[
                candle_time,
                "last_swing_low"
            ] = float(last_low["pivot_price"])

            result.at[
                candle_time,
                "last_swing_low_time"
            ] = last_low["pivot_time"]

        atr = _get_atr(candle, config)

        upward_price = (
            float(candle["close"])
            if config.use_close_for_break
            else float(candle["high"])
        )

        downward_price = (
            float(candle["close"])
            if config.use_close_for_break
            else float(candle["low"])
        )

        event = "NONE"
        broken_level = np.nan

        if last_high is not None:
            high_level = float(last_high["pivot_price"])
            high_threshold = (
                high_level + config.break_buffer_atr * atr
            )

            high_key = (
                "TOP",
                last_high["pivot_time"],
                high_level,
            )

            if (
                upward_price > high_threshold
                and (
                    not config.one_break_per_level
                    or high_key not in broken_levels
                )
            ):
                if previous_trend == "BEARISH":
                    event = "BULLISH_CHOCH"
                else:
                    event = "BULLISH_BOS"

                broken_level = high_level
                broken_levels.add(high_key)

        if last_low is not None:
            low_level = float(last_low["pivot_price"])
            low_threshold = (
                low_level - config.break_buffer_atr * atr
            )

            low_key = (
                "BOTTOM",
                last_low["pivot_time"],
                low_level,
            )

            if (
                downward_price < low_threshold
                and (
                    not config.one_break_per_level
                    or low_key not in broken_levels
                )
            ):
                if previous_trend == "BULLISH":
                    event = "BEARISH_CHOCH"
                else:
                    event = "BEARISH_BOS"

                broken_level = low_level
                broken_levels.add(low_key)

        if event == "BULLISH_CHOCH":
            result.at[candle_time, "choch"] = 1
            previous_trend = "BULLISH"

        elif event == "BEARISH_CHOCH":
            result.at[candle_time, "choch"] = -1
            previous_trend = "BEARISH"

        elif event == "BULLISH_BOS":
            result.at[candle_time, "bos"] = 1
            previous_trend = "BULLISH"

        elif event == "BEARISH_BOS":
            result.at[candle_time, "bos"] = -1
            previous_trend = "BEARISH"

        else:
            # Chỉ cập nhật trend nếu chưa có sự kiện phá vỡ.
            if previous_trend == "UNKNOWN":
                previous_trend = trend

        result.at[candle_time, "market_trend"] = (
            previous_trend
        )

        result.at[
            candle_time,
            "structure_event"
        ] = event

        result.at[
            candle_time,
            "broken_level"
        ] = broken_level

        result.at[
            candle_time,
            "latest_structure_label"
        ] = latest_label

    return result


def summarize_market_structure(
    structure: pd.DataFrame,
) -> dict:
    """
    Thống kê các sự kiện cấu trúc thị trường.
    """
    if structure.empty:
        return {
            "total_bos": 0,
            "total_choch": 0,
            "bullish_bos": 0,
            "bearish_bos": 0,
            "bullish_choch": 0,
            "bearish_choch": 0,
        }

    events = structure["structure_event"]

    return {
        "total_bos": int(structure["bos"].ne(0).sum()),
        "total_choch": int(structure["choch"].ne(0).sum()),

        "bullish_bos": int(
            events.eq("BULLISH_BOS").sum()
        ),

        "bearish_bos": int(
            events.eq("BEARISH_BOS").sum()
        ),

        "bullish_choch": int(
            events.eq("BULLISH_CHOCH").sum()
        ),

        "bearish_choch": int(
            events.eq("BEARISH_CHOCH").sum()
        ),
    }
