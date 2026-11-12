
"""
Adaptive Swing Detection
------------------------
Phát hiện Swing High / Swing Low bằng:
- ATR thích ứng.
- Theo dõi cực trị động.
- Xác nhận bằng nhịp đảo chiều tiếp theo.
- Kiểm tra hướng di chuyển và độ nhô.
- Lọc nến có range bất thường.
- So sánh biên độ swing với các swing liền trước.
- Đảm bảo cấu trúc TOP -> BOTTOM -> TOP -> BOTTOM.

Chỉ xử lý dữ liệu nến đã đóng.
Không sử dụng pivot cố định 2 hoặc 3 nến mỗi bên.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class SwingConfig:
    # ATR
    atr_period: int = 14

    # Ngưỡng đảo chiều tính theo ATR.
    # Đây là giá trị thử nghiệm, cần backtest.
    reversal_atr: float = 1.0

    # Ngưỡng phân loại swing lớn.
    major_leg_atr: float = 3.0

    # Kiểm tra hướng di chuyển.
    min_directional_ratio: float = 0.55

    # Cho phép điều chỉnh ngược hướng.
    max_corrections_short: int = 0
    max_corrections_medium: int = 1
    max_corrections_long: int = 2

    # Độ dài nến cực trị so với range trung vị.
    min_pivot_range_ratio: float = 0.20
    max_pivot_range_ratio: float = 3.00

    # Lọc swing có biên độ quá lớn.
    max_swing_amplitude_ratio: float = 3.0
    amplitude_history: int = 3
    min_history_for_amplitude_filter: int = 2

    # Phân loại trạng thái thị trường.
    regime_window: int = 20
    sideways_efficiency_threshold: float = 0.25
    sideways_width_atr_threshold: float = 4.0

    # Metadata
    label_version: str = "adaptive_swing_v1"


# ============================================================
# 2. DATA PREPARATION
# ============================================================

REQUIRED_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu đầu vào.

    Yêu cầu:
        time, open, high, low, close

    Timestamp phải tăng dần và không trùng.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào phải là pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame không có dữ liệu.")

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    result = df.copy()

    # Chuyển timestamp.
    if pd.api.types.is_numeric_dtype(result["time"]):
        result["time"] = pd.to_datetime(
            result["time"],
            unit="s",
            utc=True,
            errors="coerce",
        )
    else:
        result["time"] = pd.to_datetime(
            result["time"],
            utc=True,
            errors="coerce",
        )

    if result["time"].isna().any():
        raise ValueError("Timestamp chứa giá trị không hợp lệ.")

    if result["time"].duplicated().any():
        raise ValueError("Dữ liệu chứa timestamp trùng.")

    if not result["time"].is_monotonic_increasing:
        raise ValueError("Timestamp phải được sắp xếp tăng dần.")

    # Chuyển OHLC thành số.
    for col in ["open", "high", "low", "close"]:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce",
        )

    values = result[
        ["open", "high", "low", "close"]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("OHLC chứa giá trị không hợp lệ.")

    invalid_ohlc = (
        (result["high"] < result["open"])
        | (result["high"] < result["close"])
        | (result["high"] < result["low"])
        | (result["low"] > result["open"])
        | (result["low"] > result["close"])
    )

    if invalid_ohlc.any():
        raise ValueError(
            f"Có {int(invalid_ohlc.sum())} dòng OHLC không hợp lệ."
        )

    result = result.reset_index(drop=True)

    return result


# ============================================================
# 3. CALCULATE ATR
# ============================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Tính ATR bằng True Range và rolling mean.

    Chỉ sử dụng dữ liệu từ hiện tại trở về trước.
    Không sử dụng dữ liệu tương lai.
    """

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        window=period,
        min_periods=period,
    ).mean()

    return atr


# ============================================================
# 4. CHECK DIRECTIONAL MOVEMENT
# ============================================================

def calculate_directional_quality(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    swing_type: str,
) -> dict:
    """
    Đo mức độ di chuyển có hướng của nhịp giá.

    TOP:
        X = High
        U = tổng các đoạn tăng
        D = tổng các đoạn giảm
        Q = U / (U + D)

    BOTTOM:
        X = Low
        D = tổng các đoạn giảm
        U = tổng các đoạn tăng
        Q = D / (U + D)

    Đồng thời đếm số lần điều chỉnh ngược hướng.
    """

    if end_idx <= start_idx:
        return {
            "directional_ratio": 0.0,
            "corrections": 0,
            "movement_count": 0,
        }

    if swing_type == "TOP":
        values = df["high"].iloc[
            start_idx:end_idx + 1
        ].to_numpy(dtype=float)

        changes = np.diff(values)

        upward = float(
            np.maximum(changes, 0).sum()
        )

        downward = float(
            np.maximum(-changes, 0).sum()
        )

        total = upward + downward

        ratio = (
            upward / total
            if total > 0
            else 0.0
        )

        corrections = int(
            np.sum(changes < 0)
        )

    elif swing_type == "BOTTOM":
        values = df["low"].iloc[
            start_idx:end_idx + 1
        ].to_numpy(dtype=float)

        changes = np.diff(values)

        downward = float(
            np.maximum(-changes, 0).sum()
        )

        upward = float(
            np.maximum(changes, 0).sum()
        )

        total = downward + upward

        ratio = (
            downward / total
            if total > 0
            else 0.0
        )

        corrections = int(
            np.sum(changes > 0)
        )

    else:
        raise ValueError(
            "swing_type phải là TOP hoặc BOTTOM."
        )

    return {
        "directional_ratio": ratio,
        "corrections": corrections,
        "movement_count": len(changes),
    }


def allowed_corrections(
    movement_count: int,
    config: SwingConfig,
) -> int:
    """
    Số lần điều chỉnh ngược hướng được chấp nhận.

    Nhịp ngắn:
        Không cho phép điều chỉnh.

    Nhịp trung bình:
        Cho phép 1 lần.

    Nhịp dài:
        Cho phép tối đa 2 lần.
    """

    if movement_count < 3:
        return config.max_corrections_short

    if movement_count < 6:
        return config.max_corrections_medium

    return config.max_corrections_long


# ============================================================
# 5. CHECK PIVOT CANDLE RANGE
# ============================================================

def check_pivot_range(
    df: pd.DataFrame,
    pivot_idx: int,
    previous_pivot_idx: Optional[int],
    config: SwingConfig,
) -> dict:
    """
    Kiểm tra range của nến cực trị.

    Ri = High_i - Low_i

    So sánh range nến cực trị với range trung vị
    của các nến nằm trong nhịp trước đó.

    Nếu không đủ nến trong nhịp, dùng một vùng dữ liệu
    quá khứ ngắn để làm tham chiếu.

    Không sử dụng dữ liệu sau pivot_idx.
    """

    pivot_range = float(
        df.iloc[pivot_idx]["high"]
        - df.iloc[pivot_idx]["low"]
    )

    if previous_pivot_idx is not None:
        start_idx = previous_pivot_idx + 1
        end_idx = pivot_idx
    else:
        start_idx = max(0, pivot_idx - 20)
        end_idx = pivot_idx

    reference = df.iloc[
        start_idx:end_idx
    ]

    if reference.empty:
        reference = df.iloc[
            max(0, pivot_idx - 20):pivot_idx
        ]

    if reference.empty:
        return {
            "valid": True,
            "range_ratio": 1.0,
            "reference_range": pivot_range,
        }

    reference_ranges = (
        reference["high"] - reference["low"]
    ).to_numpy(dtype=float)

    reference_range = float(
        np.median(reference_ranges)
    )

    if reference_range <= 0:
        return {
            "valid": False,
            "range_ratio": np.inf,
            "reference_range": reference_range,
        }

    ratio = pivot_range / reference_range

    valid = (
        config.min_pivot_range_ratio
        <= ratio
        <= config.max_pivot_range_ratio
    )

    return {
        "valid": valid,
        "range_ratio": float(ratio),
        "reference_range": reference_range,
    }


# ============================================================
# 6. CHECK SWING AMPLITUDE
# ============================================================

def check_swing_amplitude(
    amplitude: float,
    previous_amplitudes: list[float],
    config: SwingConfig,
) -> bool:
    """
    So sánh biên độ swing mới với các swing trước.

    Chỉ lọc khi đã có đủ lịch sử so sánh.

    Ví dụ:
        Nếu swing mới lớn gấp 3 lần trung vị các swing
        gần nhất thì xem xét loại bỏ.
    """

    history = previous_amplitudes[
        -config.amplitude_history:
    ]

    if len(history) < config.min_history_for_amplitude_filter:
        return True

    reference = float(np.median(history))

    if reference <= 0:
        return True

    ratio = amplitude / reference

    return ratio <= config.max_swing_amplitude_ratio


# ============================================================
# 7. DETECT MARKET REGIME
# ============================================================

def detect_market_regime(
    df: pd.DataFrame,
    pivot_idx: int,
    atr_values: pd.Series,
    config: SwingConfig,
) -> str:
    """
    Phân loại sơ bộ TREND / SIDEWAYS.

    Chỉ dùng dữ liệu từ quá khứ đến pivot_idx.

    Đây là bộ phân loại đơn giản ban đầu.
    Có thể thay bằng sideways_filter.py sau này.
    """

    start_idx = max(
        0,
        pivot_idx - config.regime_window + 1,
    )

    window = df.iloc[
        start_idx:pivot_idx + 1
    ]

    if len(window) < 3:
        return "UNKNOWN"

    close_values = window["close"].to_numpy(
        dtype=float
    )

    net_movement = abs(
        close_values[-1] - close_values[0]
    )

    total_movement = float(
        np.abs(np.diff(close_values)).sum()
    )

    efficiency = (
        net_movement / total_movement
        if total_movement > 0
        else 0.0
    )

    atr_value = atr_values.iloc[pivot_idx]

    if not np.isfinite(atr_value) or atr_value <= 0:
        return "UNKNOWN"

    width = float(
        window["high"].max()
        - window["low"].min()
    )

    width_atr = width / float(atr_value)

    if (
        efficiency < config.sideways_efficiency_threshold
        and width_atr < config.sideways_width_atr_threshold
    ):
        return "SIDEWAYS"

    return "TREND"


# ============================================================
# 8. BUILD PENDING PIVOT
# ============================================================

def build_pivot_candidate(
    df: pd.DataFrame,
    atr_values: pd.Series,
    pivot_idx: int,
    swing_type: str,
    config: SwingConfig,
    previous_pivot: Optional[dict] = None,
) -> Optional[dict]:
    """
    Tạo pivot ứng viên và kiểm tra chất lượng.

    Pivot chưa được đưa vào danh sách xác nhận.
    """

    pivot = df.iloc[pivot_idx]

    pivot_price = float(
        pivot["high"]
        if swing_type == "TOP"
        else pivot["low"]
    )

    atr_at_pivot = float(
        atr_values.iloc[pivot_idx]
    )

    if not np.isfinite(atr_at_pivot) or atr_at_pivot <= 0:
        return None

    previous_idx = (
        previous_pivot["pivot_idx"]
        if previous_pivot is not None
        else None
    )

    # Kiểm tra range của nến cực trị.
    range_result = check_pivot_range(
        df=df,
        pivot_idx=pivot_idx,
        previous_pivot_idx=previous_idx,
        config=config,
    )

    if not range_result["valid"]:
        return None

    # Các thông số mặc định.
    amplitude = 0.0
    directional_ratio = 1.0
    corrections = 0
    movement_count = 0

    if previous_pivot is not None:

        if previous_pivot["swing_type"] == swing_type:
            return None

        if swing_type == "TOP":
            amplitude = (
                pivot_price
                - previous_pivot["pivot_price"]
            )

        else:
            amplitude = (
                previous_pivot["pivot_price"]
                - pivot_price
            )

        if amplitude <= 0:
            return None

        # Đo hướng di chuyển của nhịp.
        quality = calculate_directional_quality(
            df=df,
            start_idx=previous_pivot["pivot_idx"],
            end_idx=pivot_idx,
            swing_type=swing_type,
        )

        directional_ratio = quality["directional_ratio"]
        corrections = quality["corrections"]
        movement_count = quality["movement_count"]

        if directional_ratio < config.min_directional_ratio:
            return None

        correction_limit = allowed_corrections(
            movement_count,
            config,
        )

        if corrections > correction_limit:
            return None

        # Lọc swing quá lớn so với các swing trước.
        # Chỉ thực hiện khi có đủ dữ liệu lịch sử.
        if not check_swing_amplitude(
            amplitude=amplitude,
            previous_amplitudes=previous_pivot["amplitudes"],
            config=config,
        ):
            return None

    # Độ lớn của nhịp tính theo ATR.
    leg_atr = (
        amplitude / atr_at_pivot
        if amplitude > 0
        else 0.0
    )

    swing_size = (
        "MAJOR"
        if leg_atr >= config.major_leg_atr
        else "MINOR"
    )

    # Điểm prominence ban đầu.
    # Đây là điểm kỹ thuật, KHÔNG phải xác suất.
    directional_score = min(
        directional_ratio,
        1.0,
    )

    amplitude_score = min(
        leg_atr / max(config.major_leg_atr, 0.01),
        1.0,
    )

    range_score = 1.0 / (
        1.0 + abs(
            np.log(
                max(range_result["range_ratio"], 0.000001)
            )
        )
    )

    prominence_score = (
        0.40 * directional_score
        + 0.35 * amplitude_score
        + 0.25 * range_score
    )

    return {
        "pivot_idx": pivot_idx,
        "swing_type": swing_type,
        "pivot_time": pivot["time"],
        "pivot_price": pivot_price,
        "atr_at_pivot": atr_at_pivot,
        "leg_amplitude": amplitude,
        "leg_atr": leg_atr,
        "directional_ratio": directional_ratio,
        "corrections": corrections,
        "range_ratio": range_result["range_ratio"],
        "prominence_score": float(prominence_score),
        "swing_size": swing_size,
    }


# ============================================================
# 9. MAIN ADAPTIVE SWING DETECTION
# ============================================================

def detect_swings(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "M5",
    config: Optional[SwingConfig] = None,
) -> pd.DataFrame:
    """
    Phát hiện swing high / swing low.

    Logic:
        1. Theo dõi cực trị động.
        2. Đợi đảo chiều đủ ngưỡng ATR.
        3. Tạo pivot ứng viên.
        4. Pivot đối diện tiếp theo hình thành.
        5. Xác nhận pivot trước đó.
        6. Giữ cấu trúc TOP / BOTTOM luân phiên.

    Chỉ trả về swing đã xác nhận.

    Swing cuối cùng chưa có pivot đối diện xác nhận
    sẽ không được đưa vào kết quả.
    """

    if config is None:
        config = SwingConfig()

    data = prepare_data(df)

    atr_values = calculate_atr(
        data,
        period=config.atr_period,
    )

    data["_atr"] = atr_values

    # Cần đủ dữ liệu để tính ATR.
    valid_indices = np.flatnonzero(
        np.isfinite(atr_values.to_numpy())
        & (atr_values.to_numpy() > 0)
    )

    if len(valid_indices) == 0:
        return pd.DataFrame()

    first_idx = int(valid_indices[0])

    # Hướng:
    #  0 = chưa xác định
    #  1 = đang tăng, theo dõi TOP
    # -1 = đang giảm, theo dõi BOTTOM
    direction = 0

    # Cực trị đang được theo dõi.
    extreme_idx = first_idx
    extreme_price = float(
        data.iloc[first_idx]["close"]
    )

    # Pivot gần nhất đang chờ pivot đối diện xác nhận.
    pending_pivot = None

    # Danh sách swing đã xác nhận.
    confirmed_swings = []

    # Lịch sử biên độ swing.
    swing_amplitudes = []

    def update_amplitude_history():
        return swing_amplitudes.copy()

    def finalize_previous_pivot(
        previous: dict,
        current_idx: int,
    ):
        """
        Chốt pivot trước đó tại thời điểm pivot đối diện
        đã hình thành.

        confirmation_time không phải pivot_time.
        """

        confirmation_row = data.iloc[current_idx]

        confirmation_price = float(
            confirmation_row["close"]
        )

        reversal = abs(
            previous["pivot_price"]
            - confirmation_price
        )

        reversal_atr = (
            reversal / previous["atr_at_pivot"]
            if previous["atr_at_pivot"] > 0
            else 0.0
        )

        # Chỉ ghi nhận swing nếu có đảo chiều thực tế.
        if reversal_atr < config.reversal_atr:
            return

        previous_amplitudes = (
            update_amplitude_history()
        )

        # Xác định market regime tại pivot.
        market_regime = detect_market_regime(
            df=data,
            pivot_idx=previous["pivot_idx"],
            atr_values=atr_values,
            config=config,
        )

        swing_id = len(confirmed_swings) + 1

        confirmed_swings.append({
            "swing_id": swing_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "swing_type": previous["swing_type"],
            "swing_size": previous["swing_size"],
            "pivot_time": previous["pivot_time"],
            "pivot_price": previous["pivot_price"],
            "confirmation_time": confirmation_row["time"],
            "confirmation_price": confirmation_price,
            "atr_at_pivot": previous["atr_at_pivot"],
            "reversal_atr": float(reversal_atr),
            "prominence_score": previous["prominence_score"],
            "market_regime": market_regime,
            "label_version": config.label_version,

            # Các thông tin bổ sung để phân tích.
            "leg_amplitude": previous["leg_amplitude"],
            "leg_atr": previous["leg_atr"],
            "directional_ratio": previous["directional_ratio"],
            "corrections": previous["corrections"],
            "pivot_range_ratio": previous["range_ratio"],
        })

        if previous["leg_amplitude"] > 0:
            swing_amplitudes.append(
                previous["leg_amplitude"]
            )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    for idx in range(first_idx + 1, len(data)):

        row = data.iloc[idx]

        atr_now = float(
            atr_values.iloc[idx]
        )

        if not np.isfinite(atr_now) or atr_now <= 0:
            continue

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # ----------------------------------------------------
        # INITIALIZE DIRECTION
        # ----------------------------------------------------

        if direction == 0:

            threshold = (
                atr_now * config.reversal_atr
            )

            move_up = high - extreme_price
            move_down = extreme_price - low

            if (
                move_up >= threshold
                or move_down >= threshold
            ):

                if move_up >= move_down:
                    direction = 1
                    extreme_idx = idx
                    extreme_price = high

                else:
                    direction = -1
                    extreme_idx = idx
                    extreme_price = low

            else:
                if high > extreme_price:
                    extreme_price = high
                    extreme_idx = idx

                if low < extreme_price:
                    extreme_price = low
                    extreme_idx = idx

            continue

        # ----------------------------------------------------
        # UP LEG -> TRACK HIGHEST HIGH
        # ----------------------------------------------------

        if direction == 1:

            if high > extreme_price:
                extreme_price = high
                extreme_idx = idx

            atr_at_extreme = float(
                atr_values.iloc[extreme_idx]
            )

            if (
                not np.isfinite(atr_at_extreme)
                or atr_at_extreme <= 0
            ):
                continue

            reversal_threshold = (
                atr_at_extreme
                * config.reversal_atr
            )

            # Giá đóng cửa giảm đủ xa khỏi cực trị.
            if (
                extreme_price - close
                >= reversal_threshold
            ):

                candidate = build_pivot_candidate(
                    df=data,
                    atr_values=atr_values,
                    pivot_idx=extreme_idx,
                    swing_type="TOP",
                    config=config,
                    previous_pivot=pending_pivot,
                )

                if candidate is not None:

                    # Pivot mới xác nhận pivot cũ.
                    if pending_pivot is not None:
                        finalize_previous_pivot(
                            pending_pivot,
                            idx,
                        )

                    candidate["amplitudes"] = (
                        update_amplitude_history()
                    )

                    pending_pivot = candidate

                # Chuyển sang nhịp giảm.
                direction = -1
                extreme_idx = idx
                extreme_price = low

        # ----------------------------------------------------
        # DOWN LEG -> TRACK LOWEST LOW
        # ----------------------------------------------------

        elif direction == -1:

            if low < extreme_price:
                extreme_price = low
                extreme_idx = idx

            atr_at_extreme = float(
                atr_values.iloc[extreme_idx]
            )

            if (
                not np.isfinite(atr_at_extreme)
                or atr_at_extreme <= 0
            ):
                continue

            reversal_threshold = (
                atr_at_extreme
                * config.reversal_atr
            )

            # Giá đóng cửa tăng đủ xa khỏi cực trị.
            if (
                close - extreme_price
                >= reversal_threshold
            ):

                candidate = build_pivot_candidate(
                    df=data,
                    atr_values=atr_values,
                    pivot_idx=extreme_idx,
                    swing_type="BOTTOM",
                    config=config,
                    previous_pivot=pending_pivot,
                )

                if candidate is not None:

                    # Pivot mới xác nhận pivot cũ.
                    if pending_pivot is not None:
                        finalize_previous_pivot(
                            pending_pivot,
                            idx,
                        )

                    candidate["amplitudes"] = (
                        update_amplitude_history()
                    )

                    pending_pivot = candidate

                # Chuyển sang nhịp tăng.
                direction = 1
                extreme_idx = idx
                extreme_price = high

    # Không tự động xác nhận pivot cuối cùng.
    # Phải chờ nhịp đối diện hình thành.

    return pd.DataFrame(confirmed_swings)


# ============================================================
# 10. TEST
# ============================================================

if __name__ == "__main__":

    # Dữ liệu giả lập chỉ để kiểm tra cách chạy module.
    # Không sử dụng dữ liệu này để đánh giá hiệu quả thị trường.

    np.random.seed(42)

    close_prices = [
        100, 101, 102, 104, 106, 108,
        107, 105, 103, 101, 99, 97,
        98, 100, 102, 104, 106, 108,
        107, 105, 103, 101, 99, 97,
        98, 100, 102, 104, 106, 108,
        107, 105, 103, 101, 99, 97,
        98, 100, 102, 104, 106, 108,
    ]

    sample = pd.DataFrame({
        "time": pd.date_range(
            start="2026-01-01",
            periods=len(close_prices),
            freq="5min",
            tz="UTC",
        ),
        "open": close_prices,
        "high": [
            price + 0.5
            for price in close_prices
        ],
        "low": [
            price - 0.5
            for price in close_prices
        ],
        "close": close_prices,
    })

    config = SwingConfig(
        atr_period=5,
        reversal_atr=1.0,
    )

    result = detect_swings(
        df=sample,
        symbol="TEST",
        timeframe="M5",
        config=config,
    )

    print("\n========== ADAPTIVE SWING REPORT ==========")

    if result.empty:
        print("Chưa tìm được swing nào đủ điều kiện xác nhận.")

    else:
        print(
            result[
                [
                    "swing_id",
                    "swing_type",
                    "swing_size",
                    "pivot_time",
                    "pivot_price",
                    "confirmation_time",
                    "reversal_atr",
                    "prominence_score",
                ]
            ].to_string(index=False)
        )

    print("\nTotal confirmed swings:", len(result))
