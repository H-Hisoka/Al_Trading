
"""
Kiểm thử toàn bộ feature engineering.

Các module:
- indicators.py
- candle_features.py
- market_structure.py
- supply_demand.py
- liquidity_sweep.py
- feature_builder.py

Chạy:
    python -m pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

try:
    from features.indicators import calculate_indicators
    from features.candle_features import calculate_candle_features
    from features.market_structure import (
        detect_market_structure,
        summarize_market_structure,
    )
    from features.supply_demand import (
        create_supply_demand_zones,
        update_zone_status,
        add_supply_demand_features,
        summarize_supply_demand,
    )
    from features.liquidity_sweep import (
        detect_liquidity_sweeps,
        summarize_liquidity_sweeps,
    )
    from features.feature_builder import build_features

except ImportError:
    from ..features.indicators import calculate_indicators
    from ..features.candle_features import calculate_candle_features
    from ..features.market_structure import (
        detect_market_structure,
        summarize_market_structure,
    )
    from ..features.supply_demand import (
        create_supply_demand_zones,
        update_zone_status,
        add_supply_demand_features,
        summarize_supply_demand,
    )
    from ..features.liquidity_sweep import (
        detect_liquidity_sweeps,
        summarize_liquidity_sweeps,
    )
    from ..features.feature_builder import build_features


# ============================================================
# DATA FIXTURES
# ============================================================

def make_ohlc(
    n=300,
    seed=42,
    freq="5min",
):
    rng = np.random.default_rng(seed)

    index = pd.date_range(
        "2026-01-01",
        periods=n,
        freq=freq,
    )

    close = 100 + np.cumsum(
        rng.normal(0, 0.25, n)
    )

    open_price = np.empty(n)
    open_price[0] = close[0]
    open_price[1:] = close[:-1]

    high = (
        np.maximum(open_price, close)
        + rng.uniform(0.05, 0.30, n)
    )

    low = (
        np.minimum(open_price, close)
        - rng.uniform(0.05, 0.30, n)
    )

    volume = rng.integers(100, 2000, n)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def make_swing_data(df):
    """Tạo swing TOP/BOTTOM đã xác nhận để kiểm thử."""

    positions = [
        (30, 40, "TOP"),
        (65, 75, "BOTTOM"),
        (110, 120, "TOP"),
        (155, 165, "BOTTOM"),
        (210, 220, "TOP"),
        (255, 265, "BOTTOM"),
    ]

    records = []

    for swing_id, (pivot_pos, confirm_pos, swing_type) in enumerate(
        positions,
        start=1,
    ):
        price = (
            df["High"].iloc[pivot_pos]
            if swing_type == "TOP"
            else df["Low"].iloc[pivot_pos]
        )

        records.append(
            {
                "swing_id": swing_id,
                "swing_type": swing_type,
                "pivot_time": df.index[pivot_pos],
                "pivot_price": float(price),
                "confirmation_time": df.index[confirm_pos],
                "confirmation_price": float(
                    df["Close"].iloc[confirm_pos]
                ),
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def market_data():
    return make_ohlc()


@pytest.fixture
def swing_data(market_data):
    return make_swing_data(market_data)


# ============================================================
# COMMON VALIDATION
# ============================================================

def assert_valid_ohlc(df):
    assert isinstance(df, pd.DataFrame)

    required = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    assert required.issubset(df.columns)
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique

    assert (df["High"] >= df["Low"]).all()
    assert (df["High"] >= df["Open"]).all()
    assert (df["High"] >= df["Close"]).all()
    assert (df["Low"] <= df["Open"]).all()
    assert (df["Low"] <= df["Close"]).all()


def assert_no_infinite_numeric_values(df):
    numeric = df.select_dtypes(
        include=[np.number]
    )

    assert not np.isinf(
        numeric.to_numpy(dtype=float)
    ).any()


# ============================================================
# INDICATORS
# ============================================================

def test_calculate_indicators_returns_dataframe(market_data):
    result = calculate_indicators(
        market_data.copy()
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(market_data)


def test_indicators_preserve_index(market_data):
    result = calculate_indicators(
        market_data.copy()
    )

    assert result.index.equals(
        market_data.index
    )


def test_indicators_do_not_remove_ohlc(market_data):
    result = calculate_indicators(
        market_data.copy()
    )

    for col in ("Open", "High", "Low", "Close"):
        assert col in result.columns


def test_indicators_contain_expected_indicator_families(
    market_data,
):
    result = calculate_indicators(
        market_data.copy()
    )

    names = [
        col.lower()
        for col in result.columns
    ]

    # Chấp nhận các cách đặt tên khác nhau.
    groups = {
        "ATR": ("atr",),
        "EMA": ("ema",),
        "RSI": ("rsi",),
        "MACD": ("macd",),
        "Bollinger": ("bb_", "bollinger"),
    }

    for group, patterns in groups.items():
        assert any(
            any(pattern in name for pattern in patterns)
            for name in names
        ), f"Thiếu nhóm indicator: {group}"


def test_indicator_values_are_not_infinite(market_data):
    result = calculate_indicators(
        market_data.copy()
    )

    assert_no_infinite_numeric_values(result)


def test_indicators_do_not_modify_input(market_data):
    original = market_data.copy(deep=True)

    calculate_indicators(market_data)

    pd.testing.assert_frame_equal(
        market_data,
        original,
    )


# ============================================================
# CANDLE FEATURES
# ============================================================

def test_candle_features_return_dataframe(market_data):
    result = calculate_candle_features(
        market_data.copy()
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(market_data)


def test_candle_features_preserve_index(market_data):
    result = calculate_candle_features(
        market_data.copy()
    )

    assert result.index.equals(
        market_data.index
    )


def test_candle_features_have_price_structure(
    market_data,
):
    result = calculate_candle_features(
        market_data.copy()
    )

    names = [
        col.lower()
        for col in result.columns
    ]

    expected_patterns = (
        "body",
        "upper_wick",
        "lower_wick",
        "range",
    )

    for pattern in expected_patterns:
        assert any(
            pattern in name
            for name in names
        ), f"Thiếu candle feature: {pattern}"


def test_candle_range_is_non_negative(market_data):
    result = calculate_candle_features(
        market_data.copy()
    )

    range_columns = [
        col for col in result.columns
        if "range" in col.lower()
    ]

    for col in range_columns:
        values = result[col].dropna()
        assert (values >= 0).all()


def test_candle_features_do_not_modify_input(market_data):
    original = market_data.copy(deep=True)

    calculate_candle_features(market_data)

    pd.testing.assert_frame_equal(
        market_data,
        original,
    )


# ============================================================
# MARKET STRUCTURE
# ============================================================

def test_market_structure_returns_dataframe(
    market_data,
    swing_data,
):
    result = detect_market_structure(
        market_data.copy(),
        swing_data.copy(),
    )

    assert isinstance(result, pd.DataFrame)


def test_market_structure_preserves_index(
    market_data,
    swing_data,
):
    result = detect_market_structure(
        market_data.copy(),
        swing_data.copy(),
    )

    assert result.index.equals(
        market_data.index
    )


def test_market_structure_contains_structure_information(
    market_data,
    swing_data,
):
    result = detect_market_structure(
        market_data.copy(),
        swing_data.copy(),
    )

    names = [
        col.lower()
        for col in result.columns
    ]

    assert any(
        "trend" in name
        for name in names
    )

    assert any(
        "bos" in name or "choch" in name
        for name in names
    )


def test_market_structure_summary_is_available(
    market_data,
    swing_data,
):
    result = detect_market_structure(
        market_data.copy(),
        swing_data.copy(),
    )

    summary = summarize_market_structure(result)

    assert summary is not None


# ============================================================
# SUPPLY / DEMAND
# ============================================================

def test_create_supply_demand_zones_returns_dataframe(
    market_data,
    swing_data,
):
    zones = create_supply_demand_zones(
        market_data.copy(),
        swing_data.copy(),
    )

    assert isinstance(zones, pd.DataFrame)


def test_supply_demand_zones_have_price_boundaries(
    market_data,
    swing_data,
):
    zones = create_supply_demand_zones(
        market_data.copy(),
        swing_data.copy(),
    )

    if zones.empty:
        pytest.skip(
            "Không có zone phù hợp trong dữ liệu fixture."
        )

    lower_candidates = (
        "lower",
        "zone_low",
        "low",
        "bottom",
    )

    upper_candidates = (
        "upper",
        "zone_high",
        "high",
        "top",
    )

    columns = {
        col.lower(): col
        for col in zones.columns
    }

    lower_col = next(
        (
            columns[name]
            for name in lower_candidates
            if name in columns
        ),
        None,
    )

    upper_col = next(
        (
            columns[name]
            for name in upper_candidates
            if name in columns
        ),
        None,
    )

    assert lower_col is not None
    assert upper_col is not None

    assert (
        zones[upper_col] >= zones[lower_col]
    ).all()


def test_update_zone_status_returns_dataframe(
    market_data,
    swing_data,
):
    zones = create_supply_demand_zones(
        market_data.copy(),
        swing_data.copy(),
    )

    updated = update_zone_status(
        market_data.copy(),
        zones.copy(),
    )

    assert isinstance(updated, pd.DataFrame)


def test_add_supply_demand_features_returns_dataframe(
    market_data,
    swing_data,
):
    zones = create_supply_demand_zones(
        market_data.copy(),
        swing_data.copy(),
    )

    result = add_supply_demand_features(
        market_data.copy(),
        zones.copy(),
    )

    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(market_data.index)


def test_supply_demand_summary_is_available(
    market_data,
    swing_data,
):
    zones = create_supply_demand_zones(
        market_data.copy(),
        swing_data.copy(),
    )

    summary = summarize_supply_demand(zones)

    assert summary is not None


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def test_liquidity_sweep_returns_dataframe(
    market_data,
    swing_data,
):
    result = detect_liquidity_sweeps(
        market_data.copy(),
        swing_data.copy(),
    )

    assert isinstance(result, pd.DataFrame)


def test_liquidity_sweep_preserves_market_index(
    market_data,
    swing_data,
):
    result = detect_liquidity_sweeps(
        market_data.copy(),
        swing_data.copy(),
    )

    assert result.index.equals(
        market_data.index
    )


def test_liquidity_sweep_summary_is_available(
    market_data,
    swing_data,
):
    result = detect_liquidity_sweeps(
        market_data.copy(),
        swing_data.copy(),
    )

    summary = summarize_liquidity_sweeps(result)

    assert summary is not None


# ============================================================
# FEATURE BUILDER
# ============================================================

def test_feature_builder_returns_dataframe(market_data):
    result = build_features(
        market_data.copy()
    )

    if isinstance(result, tuple):
        result = result[0]

    assert isinstance(result, pd.DataFrame)


def test_feature_builder_preserves_index(market_data):
    result = build_features(
        market_data.copy()
    )

    if isinstance(result, tuple):
        result = result[0]

    assert result.index.equals(
        market_data.index
    )


def test_feature_builder_contains_numeric_features(
    market_data,
):
    result = build_features(
        market_data.copy()
    )

    if isinstance(result, tuple):
        result = result[0]

    numeric = result.select_dtypes(
        include=[np.number]
    )

    assert len(numeric.columns) > 0


def test_feature_builder_has_no_future_label_columns(
    market_data,
):
    data = market_data.copy()

    data["future_label"] = 1
    data["future_top"] = 0
    data["future_bottom"] = 0
    data["target"] = 1

    result = build_features(data)

    if isinstance(result, tuple):
        result = result[0]

    forbidden = [
        col
        for col in result.columns
        if str(col).lower().startswith("future_")
        or str(col).lower() in (
            "target",
            "label",
            "event_label",
        )
    ]

    assert forbidden == []


def test_feature_builder_does_not_modify_input(
    market_data,
):
    original = market_data.copy(deep=True)

    build_features(market_data)

    pd.testing.assert_frame_equal(
        market_data,
        original,
    )


def test_feature_builder_rejects_invalid_ohlc():
    invalid = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [99, 103, 104],
            "Low": [98, 100, 101],
            "Close": [99, 102, 103],
        },
        index=pd.date_range(
            "2026-01-01",
            periods=3,
            freq="5min",
        ),
    )

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        build_features(invalid)


def test_feature_builder_rejects_duplicate_index(
    market_data,
):
    duplicated = pd.concat(
        [
            market_data,
            market_data.iloc[[10]],
        ]
    ).sort_index()

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        build_features(duplicated)


def test_feature_builder_rejects_unsorted_index(
    market_data,
):
    unsorted = market_data.iloc[::-1].copy()

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        build_features(unsorted)
