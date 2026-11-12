
# File: features/indicators.py

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"high", "low", "close"}


def _validate_ohlcv(
    candles: pd.DataFrame,
    require_volume: bool = False,
) -> pd.DataFrame:
    """
    Kiểm tra và chuẩn hóa dữ liệu OHLCV.

    Dữ liệu đầu vào phải được sắp xếp theo thời gian tăng dần.
    Hàm không tự sắp xếp để tránh vô tình làm sai thứ tự dữ liệu.
    """
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles phải là pandas DataFrame")

    missing = REQUIRED_COLUMNS - set(candles.columns)

    if missing:
        raise ValueError(
            f"Thiếu các cột bắt buộc: {sorted(missing)}"
        )

    if require_volume and "volume" not in candles.columns:
        raise ValueError("Thiếu cột volume")

    df = candles.copy()

    for column in ["high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if "open" in df.columns:
        df["open"] = pd.to_numeric(
            df["open"],
            errors="coerce",
        )

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )

    if df[["high", "low", "close"]].isna().any().any():
        raise ValueError("Dữ liệu OHLC chứa giá trị không hợp lệ")

    if (df["high"] < df["low"]).any():
        raise ValueError("Có nến có high nhỏ hơn low")

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Dữ liệu phải được sắp xếp theo thời gian tăng dần"
        )

    return df


def calculate_true_range(
    candles: pd.DataFrame,
) -> pd.Series:
    """
    True Range = max(
        High - Low,
        abs(High - Previous Close),
        abs(Low - Previous Close)
    )
    """
    df = _validate_ohlcv(candles)

    previous_close = df["close"].shift(1)

    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def calculate_atr(
    candles: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range sử dụng Wilder smoothing.

    ATR phản ánh mức độ biến động của thị trường,
    không xác định hướng tăng hay giảm.
    """
    if period < 1:
        raise ValueError("period phải >= 1")

    true_range = calculate_true_range(candles)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


def calculate_ema(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """Exponential Moving Average."""
    if period < 1:
        raise ValueError("period phải >= 1")

    return series.ewm(
        span=period,
        min_periods=period,
        adjust=False,
    ).mean()


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Relative Strength Index sử dụng Wilder smoothing.
    """
    if period < 1:
        raise ValueError("period phải >= 1")

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # Trường hợp không có biến động giá.
    rsi = rsi.mask(
        (average_gain == 0) & (average_loss == 0),
        50,
    )

    # Trường hợp chỉ tăng, không giảm.
    rsi = rsi.mask(
        (average_gain > 0) & (average_loss == 0),
        100,
    )

    # Trường hợp chỉ giảm, không tăng.
    rsi = rsi.mask(
        (average_gain == 0) & (average_loss > 0),
        0,
    )

    return rsi


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    MACD gồm:
    - MACD line
    - Signal line
    - Histogram
    """
    if fast_period >= slow_period:
        raise ValueError(
            "fast_period phải nhỏ hơn slow_period"
        )

    fast_ema = calculate_ema(close, fast_period)
    slow_ema = calculate_ema(close, slow_period)

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal_period,
        min_periods=signal_period,
        adjust=False,
    ).mean()

    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        },
        index=close.index,
    )


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Bao gồm:
    - Middle band
    - Upper band
    - Lower band
    - Band width
    - %B
    """
    if period < 1:
        raise ValueError("period phải >= 1")

    if std_multiplier <= 0:
        raise ValueError("std_multiplier phải > 0")

    middle = close.rolling(
        window=period,
        min_periods=period,
    ).mean()

    std = close.rolling(
        window=period,
        min_periods=period,
    ).std(ddof=0)

    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std

    width = (
        (upper - lower) /
        middle.replace(0, np.nan)
    )

    percent_b = (
        (close - lower) /
        (upper - lower).replace(0, np.nan)
    )

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_percent_b": percent_b,
        },
        index=close.index,
    )


def calculate_stochastic(
    candles: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator.

    %K = (Close - Lowest Low) /
         (Highest High - Lowest Low) * 100

    %D = SMA(%K, d_period)
    """
    if k_period < 1 or d_period < 1:
        raise ValueError("Các period phải >= 1")

    df = _validate_ohlcv(candles)

    lowest_low = df["low"].rolling(
        k_period,
        min_periods=k_period,
    ).min()

    highest_high = df["high"].rolling(
        k_period,
        min_periods=k_period,
    ).max()

    denominator = (highest_high - lowest_low).replace(
        0,
        np.nan,
    )

    k = (
        (df["close"] - lowest_low) /
        denominator
    ) * 100

    d = k.rolling(
        d_period,
        min_periods=d_period,
    ).mean()

    return pd.DataFrame(
        {
            "stoch_k": k,
            "stoch_d": d,
        },
        index=df.index,
    )


def calculate_adx(
    candles: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index.

    ADX: độ mạnh xu hướng.
    +DI: sức mạnh hướng tăng.
    -DI: sức mạnh hướng giảm.
    """
    if period < 1:
        raise ValueError("period phải >= 1")

    df = _validate_ohlcv(candles)

    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr = calculate_atr(df, period)

    smoothed_plus_dm = plus_dm.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    smoothed_minus_dm = minus_dm.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    plus_di = (
        100 * smoothed_plus_dm /
        atr.replace(0, np.nan)
    )

    minus_di = (
        100 * smoothed_minus_dm /
        atr.replace(0, np.nan)
    )

    di_sum = (plus_di + minus_di).replace(
        0,
        np.nan,
    )

    dx = (
        (plus_di - minus_di).abs() /
        di_sum
    ) * 100

    adx = dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    return pd.DataFrame(
        {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
        },
        index=df.index,
    )


def calculate_indicators(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tính toàn bộ indicator mặc định.

    Giữ nguyên dữ liệu đầu vào và bổ sung các cột chỉ báo.
    """
    df = _validate_ohlcv(candles)

    result = df.copy()

    result["atr_14"] = calculate_atr(df, 14)

    result["rsi_14"] = calculate_rsi(
        df["close"],
        14,
    )

    result["ema_9"] = calculate_ema(
        df["close"],
        9,
    )

    result["ema_21"] = calculate_ema(
        df["close"],
        21,
    )

    result["ema_50"] = calculate_ema(
        df["close"],
        50,
    )

    macd = calculate_macd(df["close"])
    result = result.join(macd)

    bollinger = calculate_bollinger_bands(
        df["close"]
    )
    result = result.join(bollinger)

    stochastic = calculate_stochastic(df)
    result = result.join(stochastic)

    adx = calculate_adx(df)
    result = result.join(adx)

    # Khoảng cách giá so với EMA, chuẩn hóa theo ATR.
    atr_safe = result["atr_14"].replace(0, np.nan)

    result["close_ema9_atr"] = (
        df["close"] - result["ema_9"]
    ) / atr_safe

    result["close_ema21_atr"] = (
        df["close"] - result["ema_21"]
    ) / atr_safe

    result["ema9_ema21_atr"] = (
        result["ema_9"] - result["ema_21"]
    ) / atr_safe

    return result
