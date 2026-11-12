
"""
ATR Thresholds
--------------
Tính ATR và xây dựng các ngưỡng thích ứng
dùng trong thuật toán Adaptive Swing Detection.

Không sử dụng dữ liệu tương lai.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class ATRConfig:
    atr_period: int = 14

    # Các mức đảo chiều dùng để thử nghiệm.
    reversal_multipliers: tuple = (
        0.8,
        1.0,
        1.5,
        2.0,
    )

    # Ngưỡng phân loại swing lớn.
    major_swing_multiplier: float = 3.0

    def __post_init__(self):
        if self.atr_period <= 0:
            raise ValueError("atr_period phải lớn hơn 0.")

        if not self.reversal_multipliers:
            raise ValueError("Phải có ít nhất một reversal multiplier.")

        if any(x <= 0 for x in self.reversal_multipliers):
            raise ValueError("Các reversal multiplier phải lớn hơn 0.")

        if self.major_swing_multiplier <= 0:
            raise ValueError("major_swing_multiplier phải lớn hơn 0.")


# ============================================================
# 2. CALCULATE TRUE RANGE
# ============================================================

def calculate_true_range(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Tính True Range.

    TR = max(
        High - Low,
        abs(High - Previous Close),
        abs(Low - Previous Close)
    )
    """

    required = ["high", "low", "close"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Thiếu cột: {missing}")

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return true_range


# ============================================================
# 3. CALCULATE ATR
# ============================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Tính ATR bằng trung bình True Range trong period nến.

    Phiên bản này dùng rolling mean để đồng nhất
    với cách tính ATR trong adaptive_swing.py trước đó.

    ATR tại nến i chỉ sử dụng dữ liệu đến nến i.
    """

    if period <= 0:
        raise ValueError("period phải lớn hơn 0.")

    true_range = calculate_true_range(df)

    atr = true_range.rolling(
        window=period,
        min_periods=period,
    ).mean()

    return atr


# ============================================================
# 4. BUILD REVERSAL THRESHOLDS
# ============================================================

def calculate_reversal_threshold(
    atr: float,
    multiplier: float,
) -> float:
    """
    Tính ngưỡng đảo chiều.

    Reversal threshold = ATR * multiplier
    """

    if not np.isfinite(atr) or atr <= 0:
        return np.nan

    if multiplier <= 0:
        raise ValueError("multiplier phải lớn hơn 0.")

    return float(atr * multiplier)


def calculate_all_reversal_thresholds(
    atr: pd.Series,
    multipliers: tuple = (0.8, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """
    Tạo nhiều ngưỡng đảo chiều để thử nghiệm.

    Ví dụ:
        ATR = 0.0010

        0.8 ATR = 0.0008
        1.0 ATR = 0.0010
        1.5 ATR = 0.0015
        2.0 ATR = 0.0020
    """

    result = pd.DataFrame(index=atr.index)

    for multiplier in multipliers:

        if multiplier <= 0:
            raise ValueError(
                "Multiplier phải lớn hơn 0."
            )

        column_name = (
            f"reversal_{str(multiplier).replace('.', '_')}_atr"
        )

        result[column_name] = atr * multiplier

    return result


# ============================================================
# 5. BUILD MAJOR SWING THRESHOLD
# ============================================================

def calculate_major_swing_threshold(
    atr: pd.Series,
    multiplier: float = 3.0,
) -> pd.Series:
    """
    Tính ngưỡng biên độ phân loại Major Swing.

    Major threshold = ATR * multiplier

    Đây là ngưỡng phân loại độ lớn nhịp,
    không phải điều kiện xác nhận pivot độc lập.
    """

    if multiplier <= 0:
        raise ValueError("multiplier phải lớn hơn 0.")

    return atr * multiplier


# ============================================================
# 6. BUILD ALL ATR THRESHOLDS
# ============================================================

def calculate_atr_thresholds(
    df: pd.DataFrame,
    config: Optional[ATRConfig] = None,
) -> pd.DataFrame:
    """
    Tính toàn bộ ATR và các ngưỡng.

    Returns:
        DataFrame gồm:
            atr
            reversal_0_8_atr
            reversal_1_0_atr
            reversal_1_5_atr
            reversal_2_0_atr
            major_swing_threshold
    """

    if config is None:
        config = ATRConfig()

    atr = calculate_atr(
        df=df,
        period=config.atr_period,
    )

    result = pd.DataFrame(
        index=df.index
    )

    result["atr"] = atr

    reversal_thresholds = (
        calculate_all_reversal_thresholds(
            atr=atr,
            multipliers=config.reversal_multipliers,
        )
    )

    result = pd.concat(
        [result, reversal_thresholds],
        axis=1,
    )

    result["major_swing_threshold"] = (
        calculate_major_swing_threshold(
            atr=atr,
            multiplier=config.major_swing_multiplier,
        )
    )

    return result


# ============================================================
# 7. GET THRESHOLD AT INDEX
# ============================================================

def get_threshold_at(
    thresholds: pd.DataFrame,
    index: int,
    multiplier: float = 1.0,
) -> dict:
    """
    Lấy ATR và các ngưỡng tại một nến cụ thể.

    Hữu ích khi adaptive_swing.py cần lấy
    ngưỡng tại thời điểm cực trị.
    """

    if index not in thresholds.index:
        raise IndexError(
            f"Không tìm thấy index {index}."
        )

    row = thresholds.loc[index]

    atr = float(row["atr"])

    reversal = calculate_reversal_threshold(
        atr=atr,
        multiplier=multiplier,
    )

    return {
        "atr": atr,
        "reversal_threshold": reversal,
        "major_swing_threshold": float(
            row["major_swing_threshold"]
        ),
    }


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":

    sample = pd.DataFrame({
        "high": [
            1.1020, 1.1030, 1.1040,
            1.1050, 1.1040, 1.1060,
            1.1070, 1.1080, 1.1070,
            1.1090, 1.1100, 1.1080,
            1.1110, 1.1120, 1.1130,
        ],
        "low": [
            1.1000, 1.1010, 1.1020,
            1.1030, 1.1020, 1.1040,
            1.1050, 1.1060, 1.1050,
            1.1070, 1.1080, 1.1060,
            1.1090, 1.1100, 1.1110,
        ],
        "close": [
            1.1010, 1.1020, 1.1030,
            1.1040, 1.1030, 1.1050,
            1.1060, 1.1070, 1.1060,
            1.1080, 1.1090, 1.1070,
            1.1100, 1.1110, 1.1120,
        ],
    })

    config = ATRConfig(
        atr_period=5,
        reversal_multipliers=(
            0.8, 1.0, 1.5, 2.0
        ),
        major_swing_multiplier=3.0,
    )

    result = calculate_atr_thresholds(
        sample,
        config,
    )

    print("\n========== ATR THRESHOLD REPORT ==========")

    print(result.tail(5).to_string())

    print("\n========== TEST ONE CANDLE ==========")

    print(
        get_threshold_at(
            thresholds=result,
            index=14,
            multiplier=1.5,
        )
    )
