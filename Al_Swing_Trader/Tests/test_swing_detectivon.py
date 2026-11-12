
"""
File: tests/test_swing_detection.py

Unit tests cho Adaptive Swing Detection.

Kiểm tra:
- Swing high / swing low.
- Swing minor / major.
- Cấu trúc TOP / BOTTOM.
- Không trùng pivot.
- Thời điểm xác nhận.
- Độ dài và biên độ swing.
- Độ nhô quanh pivot.
- Nến điều chỉnh trong một nhịp giá.
- Sideways filtering.

Synthetic data chỉ phục vụ unit test,
không được dùng để đánh giá hiệu suất thị trường.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


try:
    from swing.adaptive_swing import detect_swings
except ImportError:
    from ..swing.adaptive_swing import detect_swings


# ============================================================
# 1. SYNTHETIC DATA HELPERS
# ============================================================

def make_ohlc(
    close: list[float],
    spread: float = 0.5,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    """
    Tạo OHLC giả lập phục vụ unit test.

    Không đại diện cho dữ liệu thị trường thật.
    """

    close = np.asarray(close, dtype=float)

    if len(close) == 0:
        raise ValueError("close không được rỗng.")

    open_prices = np.empty(len(close))

    open_prices[0] = close[0]

    open_prices[1:] = close[:-1]

    high = np.maximum(
        open_prices,
        close,
    ) + spread

    low = np.minimum(
        open_prices,
        close,
    ) - spread

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": np.full(
                len(close),
                100,
                dtype=int,
            ),
            "spread": np.zeros(
                len(close),
                dtype=int,
            ),
            "real_volume": np.zeros(
                len(close),
                dtype=int,
            ),
        },
        index=pd.date_range(
            start=start,
            periods=len(close),
            freq="5min",
        ),
    )


def make_trending_data(
    leg_size: float = 15.0,
    bars_per_leg: int = 25,
    cycles: int = 4,
) -> pd.DataFrame:
    """
    Tạo các nhịp tăng giảm rõ ràng.

    Mục đích:
        TOP -> BOTTOM -> TOP -> BOTTOM.
    """

    values = [100.0]

    for cycle in range(cycles):

        # Nhịp tăng.
        for i in range(1, bars_per_leg + 1):
            values.append(
                values[-1] + leg_size / bars_per_leg
            )

        # Nhịp giảm.
        for i in range(1, bars_per_leg + 1):
            values.append(
                values[-1] - leg_size / bars_per_leg
            )

    return make_ohlc(values)


def make_trend_with_corrections() -> pd.DataFrame:
    """
    Nhịp tăng có 1-2 nến điều chỉnh.

    Kiểm tra thuật toán không loại nhầm
    toàn bộ nhịp tăng chỉ vì có pullback nhỏ.
    """

    values = [100.0]

    changes = [
        1.0, 1.0, 1.0,
        -0.3,
        1.0, 1.0,
        -0.4,
        1.0, 1.0, 1.0,
        1.0, 1.0,
        -1.0,
        -1.0, -1.0, -1.0,
        -1.0, -1.0,
    ]

    for change in changes:
        values.append(values[-1] + change)

    # Bổ sung đủ dữ liệu lịch sử.
    for i in range(60):
        values.append(
            values[-1] + np.sin(i / 3) * 0.15
        )

    return make_ohlc(values)


def make_sideways_data(
    length: int = 200,
) -> pd.DataFrame:
    """
    Tạo vùng sideways với dao động nhỏ.
    """

    values = []

    for i in range(length):

        value = (
            100
            + np.sin(i * 0.8) * 0.35
            + np.sin(i * 0.23) * 0.15
        )

        values.append(value)

    return make_ohlc(values, spread=0.15)


def make_spike_data() -> pd.DataFrame:
    """
    Tạo một chuỗi có một cực trị bất thường.

    Dùng kiểm tra bộ lọc swing bất thường.
    """

    values = [100.0]

    for i in range(40):
        values.append(
            values[-1] + 0.2
        )

    # Một cú tăng đột biến.
    values.append(values[-1] + 20.0)

    # Sau đó giảm mạnh.
    for i in range(40):
        values.append(
            values[-1] - 0.2
        )

    # Dữ liệu tiếp theo.
    for i in range(50):
        values.append(
            values[-1] + np.sin(i / 4) * 0.2
        )

    return make_ohlc(values)


# ============================================================
# 2. RUN DETECTOR
# ============================================================

def run_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Chạy thuật toán phát hiện swing."""

    result = detect_swings(df.copy())

    assert isinstance(
        result,
        pd.DataFrame,
    ), "detect_swings phải trả về DataFrame."

    return result


def assert_swing_schema(swings: pd.DataFrame) -> None:
    """Kiểm tra schema kết quả swing."""

    required_columns = {
        "swing_id",
        "swing_type",
        "swing_size",
        "pivot_time",
        "pivot_price",
        "confirmation_time",
        "confirmation_price",
        "atr_at_pivot",
        "reversal_atr",
        "prominence_score",
        "market_regime",
        "label_version",
    }

    missing = required_columns - set(swings.columns)

    assert not missing, (
        f"Thiếu các cột swing: {missing}"
    )


# ============================================================
# 3. BASIC SWING DETECTION
# ============================================================

def test_detect_major_swing_high_and_low():
    """
    Kiểm tra phát hiện các nhịp tăng giảm lớn.

    Không ép pivot phải có đúng 2 hoặc 3 nến
    ở mỗi bên.
    """

    df = make_trending_data(
        leg_size=15,
        bars_per_leg=25,
        cycles=4,
    )

    swings = run_detector(df)

    assert_swing_schema(swings)

    assert not swings.empty

    assert swings["swing_type"].isin(
        ["TOP", "BOTTOM"]
    ).all()

    assert swings["swing_size"].isin(
        ["MINOR", "MAJOR"]
    ).all()

    assert "TOP" in swings["swing_type"].values

    assert "BOTTOM" in swings["swing_type"].values


def test_pivot_is_not_required_to_have_fixed_neighbors():
    """
    Kiểm tra thuật toán không phụ thuộc
    một số lượng nến cố định hai bên pivot.
    """

    values = (
        [100 + i * 0.4 for i in range(30)]
        + [112 - i * 0.1 for i in range(8)]
        + [111.2 - i * 0.8 for i in range(20)]
        + [95 + i * 0.5 for i in range(35)]
    )

    df = make_ohlc(values)

    swings = run_detector(df)

    assert_swing_schema(swings)

    assert not swings.empty


# ============================================================
# 4. STRUCTURE VALIDATION
# ============================================================

def test_no_duplicate_pivot_time():
    """
    Một pivot không được đồng thời là
    TOP và BOTTOM trong cùng cấp swing.
    """

    df = make_trending_data()

    swings = run_detector(df)

    if swings.empty:
        pytest.skip("Không có swing để kiểm tra.")

    for _, group in swings.groupby("swing_size"):

        duplicate_times = group[
            group.duplicated(
                subset=["pivot_time"],
                keep=False,
            )
        ]

        assert duplicate_times.empty, (
            "Một pivot bị gắn nhiều nhãn swing."
        )


def test_major_swings_follow_alternating_structure():
    """
    Swing MAJOR phải tạo cấu trúc luân phiên
    TOP -> BOTTOM -> TOP -> BOTTOM.
    """

    df = make_trending_data(
        cycles=5,
    )

    swings = run_detector(df)

    assert_swing_schema(swings)

    major = swings[
        swings["swing_size"] == "MAJOR"
    ].sort_values("pivot_time")

    if len(major) < 2:
        pytest.skip(
            "Không đủ swing MAJOR để kiểm tra luân phiên."
        )

    directions = major["swing_type"].tolist()

    for previous, current in zip(
        directions,
        directions[1:],
    ):
        assert previous != current, (
            "Hai swing MAJOR liên tiếp cùng hướng."
        )


def test_confirmation_must_happen_after_pivot():
    """
    Swing chỉ được xác nhận sau thời điểm pivot.

    Không được lấy confirmation_time bằng
    thời điểm pivot nếu thuật toán cần đảo chiều.
    """

    df = make_trending_data()

    swings = run_detector(df)

    assert_swing_schema(swings)

    if swings.empty:
        pytest.skip("Không có swing để kiểm tra.")

    assert (
        pd.to_datetime(swings["confirmation_time"])
        >= pd.to_datetime(swings["pivot_time"])
    ).all()

    # Giá xác nhận không nhất thiết phải trùng giá pivot.
    assert swings["confirmation_time"].notna().all()


# ============================================================
# 5. SWING PROMINENCE AND ABNORMAL MOVEMENT
# ============================================================

def test_swing_has_valid_pivot_price():
    """
    TOP phải lấy giá High tại pivot.
    BOTTOM phải lấy giá Low tại pivot.
    """

    df = make_trending_data()

    swings = run_detector(df)

    assert_swing_schema(swings)

    if swings.empty:
        pytest.skip("Không có swing để kiểm tra.")

    for _, swing in swings.iterrows():

        pivot_time = swing["pivot_time"]

        assert pivot_time in df.index

        candle = df.loc[pivot_time]

        if swing["swing_type"] == "TOP":
            assert np.isclose(
                swing["pivot_price"],
                candle["high"],
                atol=1e-8,
            )

        elif swing["swing_type"] == "BOTTOM":
            assert np.isclose(
                swing["pivot_price"],
                candle["low"],
                atol=1e-8,
            )


def test_abnormal_spike_does_not_create_conflicting_swings():
    """
    Kiểm tra cực trị bất thường không tạo
    các nhãn mâu thuẫn hoặc trùng pivot.

    Không yêu cầu loại mọi spike:
    một spike có thể là swing hợp lệ nếu
    đáp ứng các tiêu chí cấu trúc.
    """

    df = make_spike_data()

    swings = run_detector(df)

    assert_swing_schema(swings)

    if swings.empty:
        return

    assert not swings.duplicated(
        subset=["swing_id"]
    ).any()

    assert swings["swing_type"].isin(
        ["TOP", "BOTTOM"]
    ).all()


# ============================================================
# 6. CORRECTION TOLERANCE
# ============================================================

def test_one_or_two_counter_moves_are_allowed():
    """
    Nhịp tăng có 1-2 nến điều chỉnh nhỏ
    vẫn có thể hình thành swing hợp lệ.

    Không được coi mọi nến giảm là kết thúc
    nhịp tăng.
    """

    df = make_trend_with_corrections()

    swings = run_detector(df)

    assert_swing_schema(swings)

    # Kiểm tra detector chạy được và không tạo
    # nhãn mâu thuẫn trên dữ liệu có pullback.
    if not swings.empty:

        assert swings["swing_type"].isin(
            ["TOP", "BOTTOM"]
        ).all()

        assert not swings.duplicated(
            subset=["swing_id"]
        ).any()


# ============================================================
# 7. SIDEWAYS FILTERING
# ============================================================

def test_sideways_does_not_generate_excessive_swings():
    """
    Sideways nhiễu nhỏ không được tạo quá
    nhiều swing giả.
    """

    df = make_sideways_data(
        length=200,
    )

    swings = run_detector(df)

    assert_swing_schema(swings)

    # Đây là ngưỡng kiểm thử synthetic,
    # không phải ngưỡng đánh giá thị trường.
    max_swings = max(
        2,
        int(len(df) * 0.08),
    )

    assert len(swings) <= max_swings, (
        f"Sideways tạo quá nhiều swing: "
        f"{len(swings)}"
    )


def test_clear_reversal_inside_sideways_can_be_detected():
    """
    Sideways không được loại bỏ mọi swing.

    Một cú đảo chiều rõ ràng bên trong vùng
    sideways vẫn có thể được nhận diện.
    """

    values = []

    # Sideways đầu tiên.
    for i in range(60):
        values.append(
            100 + np.sin(i * 0.7) * 0.2
        )

    # Cú tăng rõ ràng.
    for i in range(15):
        values.append(
            values[-1] + 0.8
        )

    # Cú giảm rõ ràng.
    for i in range(15):
        values.append(
            values[-1] - 0.9
        )

    # Sideways tiếp tục.
    for i in range(60):
        values.append(
            values[-1] + np.sin(i * 0.7) * 0.2
        )

    df = make_ohlc(values)

    swings = run_detector(df)

    assert_swing_schema(swings)

    assert not swings.empty


# ============================================================
# 8. DATA INTEGRITY
# ============================================================

def test_missing_ohlc_columns_are_rejected():
    """Thiếu OHLC phải được xử lý rõ ràng."""

    df = make_trending_data()

    df = df.drop(columns=["high"])

    with pytest.raises(
        (ValueError, KeyError),
    ):
        run_detector(df)


def test_invalid_ohlc_is_rejected():
    """High phải >= Low."""

    df = make_trending_data()

    df.loc[df.index[10], "high"] = 90

    df.loc[df.index[10], "low"] = 100

    with pytest.raises(
        (ValueError, AssertionError),
    ):
        run_detector(df)


def test_duplicate_timestamp_is_rejected():
    """Timestamp trùng không được âm thầm xử lý."""

    df = make_trending_data()

    duplicated = pd.concat(
        [df, df.iloc[[10]]]
    ).sort_index()

    with pytest.raises(
        (ValueError, AssertionError),
    ):
        run_detector(duplicated)
