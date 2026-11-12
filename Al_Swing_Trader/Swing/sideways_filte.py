
"""
sideways_filter.py

Phát hiện trạng thái thị trường:
- TREND
- SIDEWAYS
- TRANSITION

Chỉ sử dụng dữ liệu hiện tại và quá khứ.
Không sử dụng dữ liệu tương lai.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SidewaysConfig:
    lookback: int = 20

    # Các ngưỡng ban đầu, cần kiểm thử trên dữ liệu thực tế.
    range_ratio_threshold: float = 4.0
    efficiency_ratio_threshold: float = 0.25
    overlap_threshold: float = 0.65
    trend_slope_threshold: float = 0.15
    reversal_frequency_threshold: float = 0.35

    # Trọng số đánh giá sideways.
    range_weight: float = 0.25
    efficiency_weight: float = 0.25
    overlap_weight: float = 0.20
    slope_weight: float = 0.15
    reversal_weight: float = 0.15

    sideways_threshold: float = 0.60
    trend_threshold: float = 0.40

    atr_column: str = "atr"


def _validate_data(df: pd.DataFrame, lookback: int) -> None:
    required = {"high", "low", "close"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Thiếu các cột dữ liệu: {sorted(missing)}"
        )

    if len(df) < lookback:
        raise ValueError(
            f"Cần ít nhất {lookback} nến, "
            f"hiện tại chỉ có {len(df)} nến."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError("Dữ liệu phải được sắp xếp tăng dần theo thời gian.")

    if df.index.has_duplicates:
        raise ValueError("Dữ liệu có timestamp trùng lặp.")

    if not np.isfinite(
        df[["high", "low", "close"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("OHLC chứa NaN hoặc giá trị không hợp lệ.")


def _get_atr(
    df: pd.DataFrame,
    atr_column: str
) -> pd.Series:
    """
    Ưu tiên sử dụng ATR đã tính từ atr_thresholds.py.
    Không tự tính lại ATR nếu cột đã tồn tại.
    """

    if atr_column not in df.columns:
        raise ValueError(
            f"Không tìm thấy cột {atr_column}. "
            "Hãy tính ATR bằng atr_thresholds.py trước."
        )

    atr = pd.to_numeric(df[atr_column], errors="coerce")

    return atr.replace([np.inf, -np.inf], np.nan)


def calculate_range_ratio(
    df: pd.DataFrame,
    lookback: int = 20,
    atr_column: str = "atr"
) -> pd.Series:
    """
    Đo độ rộng vùng giá so với ATR.

    Range Ratio = (Highest High - Lowest Low) / ATR
    """

    atr = _get_atr(df, atr_column)

    highest = df["high"].rolling(
        lookback,
        min_periods=lookback
    ).max()

    lowest = df["low"].rolling(
        lookback,
        min_periods=lookback
    ).min()

    ratio = (highest - lowest) / atr.replace(0, np.nan)

    return ratio.replace([np.inf, -np.inf], np.nan)


def calculate_efficiency_ratio(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.Series:
    """
    Đo hiệu suất dịch chuyển giá.

    ER = |Net Displacement| / Total Path Length

    ER gần 1: giá di chuyển có hướng.
    ER gần 0: giá dao động qua lại.
    """

    close = df["close"].astype(float)

    net_change = (
        close - close.shift(lookback)
    ).abs()

    path_length = (
        close.diff().abs()
        .rolling(lookback, min_periods=lookback)
        .sum()
    )

    result = net_change / path_length.replace(0, np.nan)

    return result.clip(0, 1)


def calculate_candle_overlap(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.Series:
    """
    Tính tỷ lệ chồng lấn giữa các nến liên tiếp.

    Overlap = Intersection / Union

    Giá trị cao thể hiện các nến thường xuyên
    nằm trong vùng giá của nhau.
    """

    previous_high = df["high"].shift(1)
    previous_low = df["low"].shift(1)

    current_high = df["high"]
    current_low = df["low"]

    intersection = (
        np.minimum(current_high, previous_high)
        - np.maximum(current_low, previous_low)
    ).clip(lower=0)

    current_range = current_high - current_low
    previous_range = previous_high - previous_low

    union = (
        np.maximum(current_high, previous_high)
        - np.minimum(current_low, previous_low)
    )

    overlap = intersection / union.replace(0, np.nan)

    return overlap.rolling(
        lookback,
        min_periods=lookback
    ).mean().clip(0, 1)


def calculate_trend_slope(
    df: pd.DataFrame,
    lookback: int = 20,
    atr_column: str = "atr"
) -> pd.Series:
    """
    Đo độ dốc xu hướng bằng hồi quy tuyến tính.

    Chuẩn hóa slope theo ATR để giảm ảnh hưởng
    của mức giá và biến động từng sản phẩm.
    """

    close = df["close"].astype(float)
    atr = _get_atr(df, atr_column)

    x = np.arange(lookback, dtype=float)
    x_centered = x - x.mean()

    denominator = np.sum(x_centered ** 2)

    def regression_slope(values):
        if not np.isfinite(values).all():
            return np.nan

        y_centered = values - values.mean()

        return np.sum(
            x_centered * y_centered
        ) / denominator

    raw_slope = close.rolling(
        lookback,
        min_periods=lookback
    ).apply(regression_slope, raw=True)

    normalized_slope = raw_slope / atr.replace(0, np.nan)

    return normalized_slope


def calculate_reversal_frequency(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.Series:
    """
    Đo tần suất đổi hướng của biến động giá.

    Chỉ tính các thay đổi hướng có giá trị khác 0.
    """

    direction = np.sign(df["close"].diff())

    previous_direction = direction.shift(1)

    reversals = (
        (direction * previous_direction < 0)
        .astype(float)
    )

    valid = (
        direction.notna()
        & previous_direction.notna()
    ).astype(float)

    reversal_count = reversals.rolling(
        lookback,
        min_periods=lookback
    ).sum()

    valid_count = valid.rolling(
        lookback,
        min_periods=lookback
    ).sum()

    return (
        reversal_count / valid_count.replace(0, np.nan)
    ).clip(0, 1)


def calculate_sideways_strength(
    range_ratio: pd.Series,
    efficiency_ratio: pd.Series,
    candle_overlap: pd.Series,
    trend_slope: pd.Series,
    reversal_frequency: pd.Series,
    config: Optional[SidewaysConfig] = None
) -> pd.Series:
    """
    Tổng hợp các đặc trưng thành điểm Sideways [0, 1].

    Điểm càng cao thì khả năng thị trường đi ngang càng lớn.
    """

    config = config or SidewaysConfig()

    range_score = (
        1 - range_ratio / config.range_ratio_threshold
    ).clip(0, 1)

    efficiency_score = (
        1 - efficiency_ratio / config.efficiency_ratio_threshold
    ).clip(0, 1)

    overlap_score = (
        candle_overlap / config.overlap_threshold
    ).clip(0, 1)

    slope_score = (
        1 - trend_slope.abs() / config.trend_slope_threshold
    ).clip(0, 1)

    reversal_score = (
        reversal_frequency
        / config.reversal_frequency_threshold
    ).clip(0, 1)

    weights = {
        "range": config.range_weight,
        "efficiency": config.efficiency_weight,
        "overlap": config.overlap_weight,
        "slope": config.slope_weight,
        "reversal": config.reversal_weight,
    }

    total_weight = sum(weights.values())

    if total_weight <= 0:
        raise ValueError("Tổng trọng số phải lớn hơn 0.")

    scores = pd.concat(
        [
            range_score.rename("range"),
            efficiency_score.rename("efficiency"),
            overlap_score.rename("overlap"),
            slope_score.rename("slope"),
            reversal_score.rename("reversal"),
        ],
        axis=1
    )

    weighted_score = sum(
        scores[name] * weight
        for name, weight in weights.items()
    )

    available_weight = scores.notna().mul(
        pd.Series(weights)
    ).sum(axis=1)

    weighted_score = (
        weighted_score / available_weight.replace(0, np.nan)
    )

    return weighted_score.clip(0, 1)


def detect_sideways(
    df: pd.DataFrame,
    config: Optional[SidewaysConfig] = None
) -> pd.DataFrame:
    """
    Phân loại từng nến thành:
    TREND / SIDEWAYS / TRANSITION.

    Các rolling window chỉ sử dụng dữ liệu hiện tại
    và quá khứ.
    """

    config = config or SidewaysConfig()

    _validate_data(df, config.lookback)

    result = pd.DataFrame(index=df.index)

    result["range_ratio"] = calculate_range_ratio(
        df,
        config.lookback,
        config.atr_column
    )

    result["efficiency_ratio"] = calculate_efficiency_ratio(
        df,
        config.lookback
    )

    result["candle_overlap"] = calculate_candle_overlap(
        df,
        config.lookback
    )

    result["trend_slope"] = calculate_trend_slope(
        df,
        config.lookback,
        config.atr_column
    )

    result["reversal_frequency"] = calculate_reversal_frequency(
        df,
        config.lookback
    )

    result["sideways_strength"] = calculate_sideways_strength(
        result["range_ratio"],
        result["efficiency_ratio"],
        result["candle_overlap"],
        result["trend_slope"],
        result["reversal_frequency"],
        config
    )

    result["market_regime"] = "TRANSITION"

    valid = result["sideways_strength"].notna()

    sideways = (
        valid
        & (
            result["sideways_strength"]
            >= config.sideways_threshold
        )
    )

    trend = (
        valid
        & (
            result["sideways_strength"]
            <= config.trend_threshold
        )
    )

    result.loc[sideways, "market_regime"] = "SIDEWAYS"
    result.loc[trend, "market_regime"] = "TREND"

    return result
