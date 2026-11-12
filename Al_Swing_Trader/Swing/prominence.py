
from dataclasses import dataclass
from typing import Any, Optional

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# CẤU HÌNH
# ============================================================

@dataclass
class ProminenceConfig:
    """
    Cấu hình đánh giá độ nổi bật của swing.

    Các trọng số là giá trị khởi đầu để thử nghiệm,
    không phải thông số tối ưu đã được chứng minh.
    """

    # Trọng số các thành phần
    weight_reversal: float = 0.25
    weight_prior_leg: float = 0.20
    weight_efficiency: float = 0.15
    weight_overlap: float = 0.15
    weight_momentum: float = 0.15
    weight_failed_breakout: float = 0.10

    # Các mốc chuẩn hóa
    reversal_atr_scale: float = 2.0
    prior_leg_atr_scale: float = 3.0
    momentum_atr_scale: float = 2.0

    # Sai số khi xác định giá thử phá cực trị
    breakout_tolerance_atr: float = 0.10

    # Ngưỡng đánh giá ban đầu, có thể điều chỉnh qua backtest
    minimum_score: float = 45.0

    def __post_init__(self):
        weights = [
            self.weight_reversal,
            self.weight_prior_leg,
            self.weight_efficiency,
            self.weight_overlap,
            self.weight_momentum,
            self.weight_failed_breakout,
        ]

        if any(w < 0 for w in weights):
            raise ValueError("Trọng số không được âm.")

        if sum(weights) <= 0:
            raise ValueError("Tổng trọng số phải lớn hơn 0.")

        if self.reversal_atr_scale <= 0:
            raise ValueError("reversal_atr_scale phải lớn hơn 0.")

        if self.prior_leg_atr_scale <= 0:
            raise ValueError("prior_leg_atr_scale phải lớn hơn 0.")

        if self.momentum_atr_scale <= 0:
            raise ValueError("momentum_atr_scale phải lớn hơn 0.")

        if self.breakout_tolerance_atr < 0:
            raise ValueError("breakout_tolerance_atr không được âm.")

        if not 0 <= self.minimum_score <= 100:
            raise ValueError("minimum_score phải nằm trong khoảng 0–100.")


# ============================================================
# KIỂM TRA VÀ CHUẨN HÓA ĐẦU VÀO
# ============================================================

def _get_pivot_type(pivot: dict) -> str:
    """
    Chuẩn hóa loại pivot thành TOP hoặc BOTTOM.
    """

    raw_type = pivot.get(
        "type",
        pivot.get("kind", pivot.get("pivot_type", ""))
    )

    raw_type = str(raw_type).upper()

    if raw_type in ("TOP", "HIGH", "SWING_HIGH"):
        return "TOP"

    if raw_type in ("BOTTOM", "LOW", "SWING_LOW"):
        return "BOTTOM"

    raise ValueError(
        f"Loại pivot không hợp lệ: {raw_type}"
    )


def _get_pivot_index(pivot: dict) -> int:
    """
    Lấy vị trí pivot theo index dạng số nguyên.

    Lưu ý:
    Đây là vị trí dòng (iloc), không phải nhãn DataFrame index.
    """

    value = pivot.get(
        "pivot_index",
        pivot.get("index")
    )

    if value is None:
        raise ValueError("Pivot thiếu pivot_index.")

    return int(value)


def _get_pivot_price(
    pivot: dict,
    df: pd.DataFrame,
    pivot_type: str,
    pivot_index: int,
) -> float:
    """
    Lấy giá cực trị từ pivot.

    Nếu pivot chưa có price:
    - TOP: lấy High tại pivot_index.
    - BOTTOM: lấy Low tại pivot_index.
    """

    price = pivot.get(
        "price",
        pivot.get("pivot_price")
    )

    if price is not None:
        price = float(price)
    elif pivot_type == "TOP":
        price = float(df.iloc[pivot_index]["high"])
    else:
        price = float(df.iloc[pivot_index]["low"])

    if not np.isfinite(price):
        raise ValueError("Giá pivot không hợp lệ.")

    return price


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0

    return float(numerator / denominator)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


# ============================================================
# ĐO BIÊN ĐỘ ĐẢO CHIỀU
# ============================================================

def calculate_reversal_metrics(
    df: pd.DataFrame,
    pivot_type: str,
    pivot_index: int,
    pivot_price: float,
    confirmation_index: int,
    atr_at_pivot: float,
) -> dict:
    """
    Đo khoảng đảo chiều từ pivot đến thời điểm xác nhận.

    TOP:
        Pivot là High.
        Đo mức giảm từ pivot đến Low thấp nhất sau pivot.

    BOTTOM:
        Pivot là Low.
        Đo mức tăng từ pivot đến High cao nhất sau pivot.

    Chỉ sử dụng dữ liệu từ pivot_index đến confirmation_index.
    """

    if confirmation_index <= pivot_index:
        raise ValueError(
            "confirmation_index phải nằm sau pivot_index."
        )

    section = df.iloc[
        pivot_index:confirmation_index + 1
    ]

    if pivot_type == "TOP":
        reversal_price = float(section["low"].min())
        reversal_distance = pivot_price - reversal_price

    else:
        reversal_price = float(section["high"].max())
        reversal_distance = reversal_price - pivot_price

    reversal_distance = max(0.0, reversal_distance)

    reversal_atr = _safe_ratio(
        reversal_distance,
        atr_at_pivot
    )

    return {
        "reversal_price": reversal_price,
        "reversal_distance": reversal_distance,
        "reversal_atr": reversal_atr,
        "reversal_bars": confirmation_index - pivot_index,
    }


# ============================================================
# ĐO NHỊP GIÁ LIỀN TRƯỚC
# ============================================================

def calculate_prior_leg_metrics(
    previous_price: float,
    pivot_price: float,
    atr_at_pivot: float,
    previous_index: int,
    pivot_index: int,
) -> dict:
    """
    Đo biên độ nhịp giá từ pivot trước đến pivot hiện tại.
    """

    amplitude = abs(pivot_price - previous_price)

    amplitude_atr = _safe_ratio(
        amplitude,
        atr_at_pivot
    )

    duration = pivot_index - previous_index

    return {
        "prior_leg_amplitude": amplitude,
        "prior_leg_atr": amplitude_atr,
        "prior_leg_bars": duration,
    }


# ============================================================
# ĐO HIỆU SUẤT DI CHUYỂN CỦA NHỊP ĐẢO CHIỀU
# ============================================================

def calculate_reversal_efficiency(
    df: pd.DataFrame,
    pivot_index: int,
    confirmation_index: int,
    pivot_type: str,
) -> float:
    """
    Đo mức độ giá di chuyển có định hướng trong nhịp đảo chiều.

    Efficiency = Net movement / Total movement

    Gần 1:
        Giá di chuyển khá một chiều.

    Gần 0:
        Giá dao động qua lại nhiều, có thể là nhiễu.

    Chỉ sử dụng Close từ pivot đến confirmation.
    """

    closes = df.iloc[
        pivot_index:confirmation_index + 1
    ]["close"].to_numpy(dtype=float)

    if len(closes) < 2:
        return 0.0

    changes = np.diff(closes)

    total_movement = float(
        np.abs(changes).sum()
    )

    if pivot_type == "TOP":
        net_movement = max(
            0.0,
            float(closes[0] - closes[-1])
        )
    else:
        net_movement = max(
            0.0,
            float(closes[-1] - closes[0])
        )

    return _clip01(
        _safe_ratio(net_movement, total_movement)
    )


# ============================================================
# ĐO MỨC CHỒNG LẤN GIỮA CÁC NẾN
# ============================================================

def calculate_candle_overlap(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> float:
    """
    Tính tỷ lệ chồng lấn trung bình giữa hai nến liên tiếp.

    overlap = phần giao nhau / range nhỏ hơn của hai nến.

    Giá trị:
        0: Không chồng lấn.
        1: Nến sau nằm hoàn toàn trong vùng nến trước
           hoặc ngược lại.

    Không sử dụng cửa sổ nến cố định.
    """

    section = df.iloc[start_index:end_index + 1]

    if len(section) < 2:
        return 0.0

    highs = section["high"].to_numpy(dtype=float)
    lows = section["low"].to_numpy(dtype=float)

    overlap_values = []

    for i in range(1, len(section)):
        previous_high = highs[i - 1]
        previous_low = lows[i - 1]

        current_high = highs[i]
        current_low = lows[i]

        intersection = max(
            0.0,
            min(previous_high, current_high)
            - max(previous_low, current_low)
        )

        previous_range = max(
            previous_high - previous_low,
            0.0
        )

        current_range = max(
            current_high - current_low,
            0.0
        )

        smaller_range = min(
            previous_range,
            current_range
        )

        if smaller_range <= 0:
            overlap = 0.0
        else:
            overlap = _clip01(
                intersection / smaller_range
            )

        overlap_values.append(overlap)

    if not overlap_values:
        return 0.0

    return float(np.mean(overlap_values))


# ============================================================
# ĐO SỰ THAY ĐỔI ĐỘNG LƯỢNG
# ============================================================

def calculate_momentum_change(
    df: pd.DataFrame,
    pivot_index: int,
    confirmation_index: int,
    atr_at_pivot: float,
) -> dict:
    """
    So sánh tốc độ thay đổi Close trước và sau pivot.

    Nhịp trước:
        previous pivot -> current pivot

    Nhịp sau:
        current pivot -> confirmation

    Slope được chuẩn hóa theo ATR tại pivot.
    """

    # Phần trước pivot cần ít nhất 2 nến
    if pivot_index < 1 or confirmation_index <= pivot_index:
        return {
            "momentum_before": 0.0,
            "momentum_after": 0.0,
            "momentum_change_atr": 0.0,
        }

    before_closes = df.iloc[
        :pivot_index + 1
    ]["close"].to_numpy(dtype=float)

    after_closes = df.iloc[
        pivot_index:confirmation_index + 1
    ]["close"].to_numpy(dtype=float)

    # Chỉ dùng đoạn trước bắt đầu từ pivot trước.
    # Phần trước được giới hạn bởi caller thông qua
    # previous_index ở hàm calculate_prominence.

    return _momentum_from_closes(
        before_closes,
        after_closes,
        atr_at_pivot
    )


def _momentum_from_closes(
    before_closes: np.ndarray,
    after_closes: np.ndarray,
    atr_at_pivot: float,
) -> dict:

    if len(before_closes) < 2 or len(after_closes) < 2:
        return {
            "momentum_before": 0.0,
            "momentum_after": 0.0,
            "momentum_change_atr": 0.0,
        }

    before_slope = (
        before_closes[-1] - before_closes[0]
    ) / (len(before_closes) - 1)

    after_slope = (
        after_closes[-1] - after_closes[0]
    ) / (len(after_closes) - 1)

    momentum_change = abs(
        after_slope - before_slope
    )

    normalized_change = _safe_ratio(
        momentum_change,
        atr_at_pivot
    )

    return {
        "momentum_before": float(before_slope),
        "momentum_after": float(after_slope),
        "momentum_change_atr": normalized_change,
    }


# ============================================================
# ĐẾM CÁC LẦN THỬ PHÁ CỰC TRỊ THẤT BẠI
# ============================================================

def calculate_failed_breakouts(
    df: pd.DataFrame,
    pivot_type: str,
    pivot_index: int,
    confirmation_index: int,
    pivot_price: float,
    atr_at_pivot: float,
    tolerance_atr: float = 0.10,
) -> int:
    """
    Đếm số lần giá thử vượt pivot nhưng đóng cửa quay trở lại.

    TOP:
        High chạm/vượt vùng đỉnh nhưng Close thấp hơn pivot.

    BOTTOM:
        Low chạm/vượt vùng đáy nhưng Close cao hơn pivot.

    Chỉ đếm các nến sau pivot và không vượt quá
    confirmation_index.
    """

    if confirmation_index <= pivot_index:
        return 0

    tolerance = atr_at_pivot * tolerance_atr

    section = df.iloc[
        pivot_index + 1:confirmation_index + 1
    ]

    count = 0

    for _, candle in section.iterrows():

        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        if pivot_type == "TOP":

            touched = high >= pivot_price - tolerance
            rejected = close < pivot_price

        else:

            touched = low <= pivot_price + tolerance
            rejected = close > pivot_price

        if touched and rejected:
            count += 1

    return count


# ============================================================
# HÀM CHẤM ĐIỂM TỔNG HỢP
# ============================================================

def calculate_prominence(
    df: pd.DataFrame,
    pivot: dict,
    previous_pivot: dict,
    confirmation_index: int,
    atr_column: str = "atr",
    config: Optional[ProminenceConfig] = None,
) -> dict:
    """
    Tính prominence_score cho một swing đã được xác nhận.

    Parameters
    ----------
    df:
        DataFrame OHLC đã được chuẩn hóa và có cột ATR.

    pivot:
        Pivot hiện tại, ví dụ:
        {
            "type": "TOP",
            "pivot_index": 100,
            "price": 2500.0
        }

    previous_pivot:
        Pivot ngay trước đó, ví dụ:
        {
            "type": "BOTTOM",
            "pivot_index": 80,
            "price": 2480.0
        }

    confirmation_index:
        Vị trí nến xác nhận swing.
        Chỉ dữ liệu đến vị trí này được sử dụng.

    Returns
    -------
    dict:
        Các chỉ số thành phần và prominence_score.
    """

    if config is None:
        config = ProminenceConfig()

    required_columns = {
        "high",
        "low",
        "close",
        atr_column,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"DataFrame thiếu cột: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("DataFrame rỗng.")

    pivot_type = _get_pivot_type(pivot)
    previous_type = _get_pivot_type(previous_pivot)

    pivot_index = _get_pivot_index(pivot)
    previous_index = _get_pivot_index(previous_pivot)

    if pivot_type == previous_type:
        raise ValueError(
            "Hai pivot liên tiếp phải khác loại TOP/BOTTOM."
        )

    if previous_index >= pivot_index:
        raise ValueError(
            "previous_pivot phải xuất hiện trước pivot."
        )

    if not (
        0 <= previous_index < pivot_index
        < confirmation_index < len(df)
    ):
        raise ValueError(
            "Vị trí pivot hoặc confirmation_index không hợp lệ."
        )

    pivot_price = _get_pivot_price(
        pivot,
        df,
        pivot_type,
        pivot_index,
    )

    previous_price = _get_pivot_price(
        previous_pivot,
        df,
        previous_type,
        previous_index,
    )

    atr_at_pivot = float(
        df.iloc[pivot_index][atr_column]
    )

    if not np.isfinite(atr_at_pivot) or atr_at_pivot <= 0:
        raise ValueError(
            "ATR tại pivot không hợp lệ."
        )

    # --------------------------------------------------------
    # 1. Khoảng đảo chiều
    # --------------------------------------------------------

    reversal = calculate_reversal_metrics(
        df=df,
        pivot_type=pivot_type,
        pivot_index=pivot_index,
        pivot_price=pivot_price,
        confirmation_index=confirmation_index,
        atr_at_pivot=atr_at_pivot,
    )

    # --------------------------------------------------------
    # 2. Độ lớn nhịp giá trước pivot
    # --------------------------------------------------------

    prior_leg = calculate_prior_leg_metrics(
        previous_price=previous_price,
        pivot_price=pivot_price,
        atr_at_pivot=atr_at_pivot,
        previous_index=previous_index,
        pivot_index=pivot_index,
    )

    # --------------------------------------------------------
    # 3. Hiệu suất di chuyển của nhịp đảo chiều
    # --------------------------------------------------------

    efficiency = calculate_reversal_efficiency(
        df=df,
        pivot_index=pivot_index,
        confirmation_index=confirmation_index,
        pivot_type=pivot_type,
    )

    # --------------------------------------------------------
    # 4. Mức chồng lấn nến
    # --------------------------------------------------------

    overlap = calculate_candle_overlap(
        df=df,
        start_index=previous_index,
        end_index=confirmation_index,
    )

    # --------------------------------------------------------
    # 5. Thay đổi động lượng
    # --------------------------------------------------------

    before_closes = df.iloc[
        previous_index:pivot_index + 1
    ]["close"].to_numpy(dtype=float)

    after_closes = df.iloc[
        pivot_index:confirmation_index + 1
    ]["close"].to_numpy(dtype=float)

    momentum = _momentum_from_closes(
        before_closes,
        after_closes,
        atr_at_pivot,
    )

    # --------------------------------------------------------
    # 6. Số lần thử phá cực trị thất bại
    # --------------------------------------------------------

    failed_breakouts = calculate_failed_breakouts(
        df=df,
        pivot_type=pivot_type,
        pivot_index=pivot_index,
        confirmation_index=confirmation_index,
        pivot_price=pivot_price,
        atr_at_pivot=atr_at_pivot,
        tolerance_atr=config.breakout_tolerance_atr,
    )

    # --------------------------------------------------------
    # 7. Chuẩn hóa từng thành phần về 0–1
    # --------------------------------------------------------

    reversal_score = _clip01(
        reversal["reversal_atr"]
        / config.reversal_atr_scale
    )

    prior_leg_score = _clip01(
        prior_leg["prior_leg_atr"]
        / config.prior_leg_atr_scale
    )

    efficiency_score = _clip01(efficiency)

    # Chồng lấn càng cao thì điểm chất lượng càng thấp.
    overlap_score = _clip01(1.0 - overlap)

    momentum_score = _clip01(
        momentum["momentum_change_atr"]
        / config.momentum_atr_scale
    )

    # Tối đa 3 lần thử phá thất bại được tính vào điểm.
    failed_breakout_score = _clip01(
        failed_breakouts / 3.0
    )

    # --------------------------------------------------------
    # 8. Tổng hợp điểm
    # --------------------------------------------------------

    weights = {
        "reversal": config.weight_reversal,
        "prior_leg": config.weight_prior_leg,
        "efficiency": config.weight_efficiency,
        "overlap": config.weight_overlap,
        "momentum": config.weight_momentum,
        "failed_breakout": config.weight_failed_breakout,
    }

    scores = {
        "reversal": reversal_score,
        "prior_leg": prior_leg_score,
        "efficiency": efficiency_score,
        "overlap": overlap_score,
        "momentum": momentum_score,
        "failed_breakout": failed_breakout_score,
    }

    total_weight = sum(weights.values())

    weighted_score = sum(
        scores[name] * weights[name]
        for name in weights
    )

    prominence_score = (
        weighted_score / total_weight
    ) * 100.0

    prominence_score = round(
        float(prominence_score),
        2
    )

    # --------------------------------------------------------
    # 9. Phân loại sơ bộ
    # --------------------------------------------------------

    is_prominent = (
        prominence_score >= config.minimum_score
    )

    if prominence_score >= 70:
        prominence_level = "HIGH"
    elif prominence_score >= config.minimum_score:
        prominence_level = "MEDIUM"
    else:
        prominence_level = "LOW"

    # --------------------------------------------------------
    # 10. Kết quả
    # --------------------------------------------------------

    return {
        "pivot_type": pivot_type,
        "pivot_index": pivot_index,
        "pivot_price": pivot_price,
        "previous_pivot_index": previous_index,
        "confirmation_index": confirmation_index,

        "atr_at_pivot": atr_at_pivot,

        "reversal_price": reversal["reversal_price"],
        "reversal_distance": reversal["reversal_distance"],
        "reversal_atr": round(reversal["reversal_atr"], 4),
        "reversal_bars": reversal["reversal_bars"],

        "prior_leg_amplitude": prior_leg["prior_leg_amplitude"],
        "prior_leg_atr": round(prior_leg["prior_leg_atr"], 4),
        "prior_leg_bars": prior_leg["prior_leg_bars"],

        "reversal_efficiency": round(efficiency, 4),
        "candle_overlap": round(overlap, 4),

        "momentum_before": momentum["momentum_before"],
        "momentum_after": momentum["momentum_after"],
        "momentum_change_atr": round(
            momentum["momentum_change_atr"],
            4
        ),

        "failed_breakouts": failed_breakouts,

        "reversal_score": round(reversal_score * 100, 2),
        "prior_leg_score": round(prior_leg_score * 100, 2),
        "efficiency_score": round(efficiency_score * 100, 2),
        "overlap_score": round(overlap_score * 100, 2),
        "momentum_score": round(momentum_score * 100, 2),
        "failed_breakout_score": round(
            failed_breakout_score * 100,
            2
        ),

        "prominence_score": prominence_score,
        "prominence_level": prominence_level,
        "is_prominent": is_prominent,
    }


# ============================================================
# KIỂM TRA NHANH
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Dữ liệu giả lập để kiểm tra module.
    # Không dùng dữ liệu này để đánh giá hiệu quả giao dịch.

    df = pd.DataFrame({
        "high": [
            100, 102, 104, 106, 108,
            107, 105, 103, 101, 100,
            99, 98, 97, 96, 95,
        ],
        "low": [
            98, 100, 102, 104, 106,
            105, 103, 101, 99, 98,
            97, 96, 95, 94, 93,
        ],
        "close": [
            99, 101, 103, 105, 107,
            106, 104, 102, 100, 99,
            98, 97, 96, 95, 94,
        ],
        "atr": [
            2.0
        ] * 15,
    })

    previous_pivot = {
        "type": "BOTTOM",
        "pivot_index": 0,
        "price": 98.0,
    }

    current_pivot = {
        "type": "TOP",
        "pivot_index": 4,
        "price": 108.0,
    }

    result = calculate_prominence(
        df=df,
        pivot=current_pivot,
        previous_pivot=previous_pivot,
        confirmation_index=14,
    )

    print("\n===== PROMINENCE RESULT =====")

    for key, value in result.items():
        print(f"{key}: {value}")
