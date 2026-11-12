
"""
Kiểm thử labeling:
- Historical labels: nhãn swing đã được xác nhận trong lịch sử.
- Future event labels: nhãn sự kiện TOP_SOON / BOTTOM_SOON
  trong một khoảng thời gian tương lai.

Chạy:
    python -m pytest tests/test_labels.py -v
"""

import numpy as np
import pandas as pd
import pytest

try:
    from labeling.historical_labels import (
        build_historical_labels,
    )
    from labeling.future_event_labels import (
        build_future_event_labels,
    )
except ImportError:
    from ..labeling.historical_labels import (
        build_historical_labels,
    )
    from ..labeling.future_event_labels import (
        build_future_event_labels,
    )


# ============================================================
# FIXTURES
# ============================================================

def make_ohlc(
    n=100,
    start="2026-01-01",
    freq="5min",
    seed=42,
):
    """Tạo OHLC giả lập chỉ dùng trong unit test."""
    rng = np.random.default_rng(seed)

    index = pd.date_range(
        start=start,
        periods=n,
        freq=freq,
    )

    close = 100 + np.cumsum(
        rng.normal(0, 0.3, n)
    )

    open_price = np.empty(n)
    open_price[0] = close[0]

    if n > 1:
        open_price[1:] = close[:-1]

    high = np.maximum(
        open_price,
        close,
    ) + rng.uniform(0.05, 0.3, n)

    low = np.minimum(
        open_price,
        close,
    ) - rng.uniform(0.05, 0.3, n)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(100, 1000, n),
        },
        index=index,
    )


def make_swings(df):
    """
    Tạo hai swing đã xác nhận:
    - TOP tại nến 20, xác nhận tại nến 25.
    - BOTTOM tại nến 40, xác nhận tại nến 45.
    """
    return pd.DataFrame(
        [
            {
                "swing_id": 1,
                "swing_type": "TOP",
                "pivot_time": df.index[20],
                "pivot_price": float(df["High"].iloc[20]),
                "confirmation_time": df.index[25],
                "confirmation_price": float(df["Close"].iloc[25]),
            },
            {
                "swing_id": 2,
                "swing_type": "BOTTOM",
                "pivot_time": df.index[40],
                "pivot_price": float(df["Low"].iloc[40]),
                "confirmation_time": df.index[45],
                "confirmation_price": float(df["Close"].iloc[45]),
            },
        ]
    )


@pytest.fixture
def market_data():
    return make_ohlc()


@pytest.fixture
def swing_data(market_data):
    return make_swings(market_data)


# ============================================================
# HISTORICAL LABELS
# ============================================================

def test_historical_labels_return_dataframe(
    market_data,
    swing_data,
):
    result = build_historical_labels(
        market_data.copy(),
        swing_data.copy(),
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(market_data)


def test_historical_labels_preserve_index(
    market_data,
    swing_data,
):
    result = build_historical_labels(
        market_data.copy(),
        swing_data.copy(),
    )

    assert result.index.equals(market_data.index)


def test_historical_labels_contain_swing_information(
    market_data,
    swing_data,
):
    result = build_historical_labels(
        market_data.copy(),
        swing_data.copy(),
    )

    possible_columns = {
        "historical_label",
        "swing_label",
        "label",
        "swing_type",
    }

    assert possible_columns.intersection(
        result.columns
    ), "Không tìm thấy cột nhãn swing trong kết quả."


def test_historical_top_is_assigned_to_pivot_candle(
    market_data,
    swing_data,
):
    result = build_historical_labels(
        market_data.copy(),
        swing_data.copy(),
    )

    label_columns = [
        col
        for col in (
            "historical_label",
            "swing_label",
            "label",
            "swing_type",
        )
        if col in result.columns
    ]

    assert label_columns

    label_col = label_columns[0]
    top_label = result.loc[
        market_data.index[20],
        label_col,
    ]

    assert str(top_label).upper() in {
        "TOP",
        "1",
        "HIGH",
        "SWING_HIGH",
    }


def test_historical_bottom_is_assigned_to_pivot_candle(
    market_data,
    swing_data,
):
    result = build_historical_labels(
        market_data.copy(),
        swing_data.copy(),
    )

    label_columns = [
        col
        for col in (
            "historical_label",
            "swing_label",
            "label",
            "swing_type",
        )
        if col in result.columns
    ]

    assert label_columns

    label_col = label_columns[0]
    bottom_label = result.loc[
        market_data.index[40],
        label_col,
    ]

    assert str(bottom_label).upper() in {
        "BOTTOM",
        "-1",
        "LOW",
        "SWING_LOW",
    }


def test_historical_labels_do_not_overlap(
    market_data,
    swing_data,
):
    result = build_historical_labels(
        market_data.copy(),
        swing_data.copy(),
    )

    top_columns = [
        col for col in result.columns
        if col.lower() in ("is_top", "top_label")
    ]

    bottom_columns = [
        col for col in result.columns
        if col.lower() in ("is_bottom", "bottom_label")
    ]

    if top_columns and bottom_columns:
        top = result[top_columns[0]].fillna(False).astype(bool)
        bottom = result[bottom_columns[0]].fillna(False).astype(bool)

        assert not (top & bottom).any()


def test_historical_labels_do_not_modify_input(
    market_data,
    swing_data,
):
    original_market = market_data.copy(deep=True)
    original_swings = swing_data.copy(deep=True)

    build_historical_labels(
        market_data,
        swing_data,
    )

    pd.testing.assert_frame_equal(
        market_data,
        original_market,
    )

    pd.testing.assert_frame_equal(
        swing_data,
        original_swings,
    )


def test_empty_swings_are_handled(market_data):
    empty_swings = pd.DataFrame(
        columns=[
            "swing_id",
            "swing_type",
            "pivot_time",
            "pivot_price",
            "confirmation_time",
            "confirmation_price",
        ]
    )

    result = build_historical_labels(
        market_data.copy(),
        empty_swings,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(market_data)


# ============================================================
# FUTURE EVENT LABELS
# ============================================================

def test_future_labels_return_dataframe(market_data):
    result = build_future_event_labels(
        market_data.copy(),
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(market_data)


def test_future_labels_have_target_column(market_data):
    result = build_future_event_labels(
        market_data.copy(),
    )

    possible_columns = {
        "future_label",
        "target",
        "event_label",
        "label",
    }

    assert possible_columns.intersection(
        result.columns
    ), "Không tìm thấy cột nhãn sự kiện tương lai."


def test_future_labels_use_only_valid_classes(market_data):
    result = build_future_event_labels(
        market_data.copy(),
    )

    label_col = next(
        (
            col for col in (
                "future_label",
                "target",
                "event_label",
                "label",
            )
            if col in result.columns
        ),
        None,
    )

    assert label_col is not None

    valid_classes = {
        "TOP_SOON",
        "BOTTOM_SOON",
        "NONE",
        "TOP",
        "BOTTOM",
        0,
        1,
        2,
        -1,
    }

    observed = set(
        result[label_col].dropna().unique()
    )

    assert observed.issubset(valid_classes)


def test_future_label_does_not_confuse_top_and_bottom(
    market_data,
):
    result = build_future_event_labels(
        market_data.copy(),
    )

    top_columns = [
        col for col in result.columns
        if col.lower() in ("future_top", "top_soon")
    ]

    bottom_columns = [
        col for col in result.columns
        if col.lower() in ("future_bottom", "bottom_soon")
    ]

    if top_columns and bottom_columns:
        top = result[top_columns[0]].fillna(False).astype(bool)
        bottom = result[bottom_columns[0]].fillna(False).astype(bool)

        assert not (top & bottom).any()


def test_future_labels_mark_insufficient_horizon_as_missing(
    market_data,
):
    """
    Những nến cuối chuỗi không có đủ dữ liệu tương lai
    không được tự động xem là NONE.

    Nếu implementation dùng một cột validity riêng,
    kiểm tra validity tại các nến cuối.
    """
    result = build_future_event_labels(
        market_data.copy(),
    )

    validity_columns = [
        col for col in result.columns
        if col.lower() in (
            "label_valid",
            "target_valid",
            "future_valid",
        )
    ]

    if validity_columns:
        valid_col = validity_columns[0]

        assert not bool(
            result[valid_col].iloc[-1]
        )


def test_future_labels_do_not_modify_input(market_data):
    original = market_data.copy(deep=True)

    build_future_event_labels(
        market_data,
    )

    pd.testing.assert_frame_equal(
        market_data,
        original,
    )


# ============================================================
# VALIDATION
# ============================================================

def test_historical_labels_reject_missing_swing_columns(
    market_data,
):
    invalid_swings = pd.DataFrame(
        {
            "swing_id": [1],
            "swing_type": ["TOP"],
        }
    )

    with pytest.raises(
        (ValueError, KeyError, TypeError)
    ):
        build_historical_labels(
            market_data.copy(),
            invalid_swings,
        )


def test_labeling_rejects_unsorted_market_data(
    market_data,
    swing_data,
):
    unsorted = market_data.iloc[::-1].copy()

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        build_historical_labels(
            unsorted,
            swing_data,
        )
