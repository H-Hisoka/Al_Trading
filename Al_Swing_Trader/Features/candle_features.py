
# File: features/candle_features.py

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
}


def _validate_candles(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """Kiểm tra dữ liệu OHLC và thứ tự thời gian."""
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles phải là pandas DataFrame")

    missing = REQUIRED_COLUMNS - set(candles.columns)

    if missing:
        raise ValueError(
            f"Thiếu các cột OHLC: {sorted(missing)}"
        )

    df = candles.copy()

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Dữ liệu OHLC có giá trị không hợp lệ")

    if (df["high"] < df["low"]).any():
        raise ValueError("Có nến có high nhỏ hơn low")

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Dữ liệu phải được sắp xếp theo thời gian tăng dần"
        )

    return df


def calculate_single_candle_features(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Đặc trưng hình dạng từng cây nến.

    Tất cả tỷ lệ đều được chuẩn hóa theo biên độ nến
    để có thể sử dụng trên nhiều mức giá khác nhau.
    """
    df = _validate_candles(candles)

    result = pd.DataFrame(index=df.index)

    candle_range = df["high"] - df["low"]

    safe_range = candle_range.replace(0, np.nan)

    body = df["close"] - df["open"]

    upper_wick = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    lower_wick = (
        df[["open", "close"]].min(axis=1) -
        df["low"]
    )

    result["candle_range"] = candle_range
    result["candle_body"] = body.abs()

    result["upper_wick"] = upper_wick
    result["lower_wick"] = lower_wick

    result["body_ratio"] = (
        body.abs() / safe_range
    )

    result["upper_wick_ratio"] = (
        upper_wick / safe_range
    )

    result["lower_wick_ratio"] = (
        lower_wick / safe_range
    )

    result["body_to_range"] = (
        body / safe_range
    )

    result["close_position"] = (
        (df["close"] - df["low"]) /
        safe_range
    )

    result["open_position"] = (
        (df["open"] - df["low"]) /
        safe_range
    )

    # 1: nến tăng, -1: nến giảm, 0: nến không đổi.
    result["candle_direction"] = np.sign(body)

    # Giá đóng cửa nằm gần đỉnh hay đáy nến.
    result["close_near_high"] = (
        1 - result["close_position"]
    )

    result["close_near_low"] = (
        result["close_position"]
    )

    # Tổng độ dài hai râu so với thân nến.
    result["wick_to_body"] = (
        (upper_wick + lower_wick) /
        body.abs().replace(0, np.nan)
    )

    return result


def calculate_price_change_features(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Đặc trưng biến động giá và khoảng trống giá.
    """
    df = _validate_candles(candles)

    result = pd.DataFrame(index=df.index)

    previous_close = df["close"].shift(1)

    result["return_1"] = df["close"].pct_change(1)
    result["return_2"] = df["close"].pct_change(2)
    result["return_3"] = df["close"].pct_change(3)
    result["return_5"] = df["close"].pct_change(5)

    result["log_return_1"] = np.log(
        df["close"] / previous_close
    )

    result["high_change_1"] = (
        df["high"] - df["high"].shift(1)
    )

    result["low_change_1"] = (
        df["low"] - df["low"].shift(1)
    )

    # Gap giữa open hiện tại và close trước đó.
    result["open_gap"] = (
        df["open"] - previous_close
    )

    result["open_gap_pct"] = (
        df["open"] / previous_close - 1
    )

    return result


def calculate_rolling_candle_features(
    candles: pd.DataFrame,
    windows: tuple[int, ...] = (3, 5, 10, 20),
) -> pd.DataFrame:
    """
    Đặc trưng tổng hợp từ các nến trước đó và nến hiện tại.

    Không sử dụng centered rolling window.
    """
    df = _validate_candles(candles)

    result = pd.DataFrame(index=df.index)

    candle_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()

    for window in windows:
        if window < 1:
            raise ValueError("window phải >= 1")

        rolling_high = df["high"].rolling(
            window,
            min_periods=window,
        ).max()

        rolling_low = df["low"].rolling(
            window,
            min_periods=window,
        ).min()

        rolling_range = rolling_high - rolling_low

        result[f"range_{window}"] = rolling_range

        result[f"range_mean_{window}"] = (
            candle_range.rolling(
                window,
                min_periods=window,
            ).mean()
        )

        result[f"body_mean_{window}"] = (
            body.rolling(
                window,
                min_periods=window,
            ).mean()
        )

        result[f"range_std_{window}"] = (
            candle_range.rolling(
                window,
                min_periods=window,
            ).std(ddof=0)
        )

        # Tỷ lệ nến tăng trong cửa sổ.
        result[f"bullish_ratio_{window}"] = (
            (df["close"] > df["open"])
            .rolling(
                window,
                min_periods=window,
            )
            .mean()
        )

        # Tỷ lệ nến giảm trong cửa sổ.
        result[f"bearish_ratio_{window}"] = (
            (df["close"] < df["open"])
            .rolling(
                window,
                min_periods=window,
            )
            .mean()
        )

        # Độ dịch chuyển giá so với tổng biến động.
        net_move = (
            df["close"] - df["close"].shift(window)
        ).abs()

        total_move = (
            df["close"].diff().abs()
            .rolling(
                window,
                min_periods=window,
            )
            .sum()
        )

        result[f"efficiency_ratio_{window}"] = (
            net_move /
            total_move.replace(0, np.nan)
        )

    return result


def calculate_volume_features(
    candles: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Đặc trưng volume.

    MT5 có thể cung cấp tick_volume hoặc real_volume.
    Hàm ưu tiên volume, nếu không có thì dùng tick_volume.
    """
    df = candles.copy()

    volume_column = None

    for candidate in [
        "volume",
        "tick_volume",
        "real_volume",
    ]:
        if candidate in df.columns:
            volume_column = candidate
            break

    result = pd.DataFrame(index=df.index)

    if volume_column is None:
        return result

    volume = pd.to_numeric(
        df[volume_column],
        errors="coerce",
    )

    result["volume"] = volume

    result["volume_change"] = volume.pct_change()

    for window in windows:
        if window < 1:
            raise ValueError("window phải >= 1")

        volume_mean = volume.rolling(
            window,
            min_periods=window,
        ).mean()

        volume_std = volume.rolling(
            window,
            min_periods=window,
        ).std(ddof=0)

        result[f"volume_mean_{window}"] = volume_mean

        result[f"volume_ratio_{window}"] = (
            volume /
            volume_mean.replace(0, np.nan)
        )

        result[f"volume_zscore_{window}"] = (
            (volume - volume_mean) /
            volume_std.replace(0, np.nan)
        )

    return result


def calculate_candle_features(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hàm tổng hợp toàn bộ candle features.
    """
    df = _validate_candles(candles)

    result = df.copy()

    single = calculate_single_candle_features(df)
    price_change = calculate_price_change_features(df)
    rolling = calculate_rolling_candle_features(df)
    volume = calculate_volume_features(df)

    result = result.join(single)
    result = result.join(price_change)
    result = result.join(rolling)
    result = result.join(volume)

    return result
