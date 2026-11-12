
"""
Kiểm thử hệ thống realtime.

Các module:
- realtime/mt5_connector.py
- realtime/candle_manager.py
- realtime/realtime_runner.py
- realtime/signal_engine.py

Các nguyên tắc:
- Chỉ xử lý nến đã đóng.
- Không xử lý trùng timestamp.
- Không dùng dữ liệu tương lai.
- Không tự động đặt lệnh giao dịch.

Chạy:
    python -m pytest tests/test_realtime.py -v
"""

import numpy as np
import pandas as pd
import pytest

try:
    from realtime.candle_manager import CandleManager
    from realtime.signal_engine import SignalEngine
except ImportError:
    from ..realtime.candle_manager import CandleManager
    from ..realtime.signal_engine import SignalEngine


# ============================================================
# FIXTURES
# ============================================================

def make_rates(
    n=100,
    start="2026-01-01",
    freq="5min",
    seed=42,
):
    rng = np.random.default_rng(seed)

    index = pd.date_range(
        start=start,
        periods=n,
        freq=freq,
    )

    close = 100 + np.cumsum(
        rng.normal(0, 0.2, n)
    )

    open_price = np.empty(n)
    open_price[0] = close[0]
    open_price[1:] = close[:-1]

    high = (
        np.maximum(open_price, close)
        + rng.uniform(0.05, 0.25, n)
    )

    low = (
        np.minimum(open_price, close)
        - rng.uniform(0.05, 0.25, n)
    )

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


@pytest.fixture
def rates():
    return make_rates()


# ============================================================
# FAKE CONNECTOR
# ============================================================

class FakeMT5Connector:
    """
    Connector giả lập.

    Chỉ sử dụng trong unit test.
    Không kết nối tài khoản MT5 thật.
    Không gửi lệnh giao dịch.
    """

    def __init__(self, data=None):
        self.data = (
            data.copy()
            if data is not None
            else make_rates()
        )

        self.connected = False
        self.fetch_count = 0
        self.order_count = 0

    def initialize(self):
        self.connected = True
        return True

    def shutdown(self):
        self.connected = False

    def get_rates(
        self,
        symbol,
        timeframe,
        count,
    ):
        self.fetch_count += 1

        return self.data.tail(count).copy()

    def place_order(self, *args, **kwargs):
        self.order_count += 1

        raise AssertionError(
            "Unit test không được phép gửi lệnh giao dịch."
        )


# ============================================================
# FAKE PREDICTOR
# ============================================================

class FakePredictor:
    """Mô hình giả lập để kiểm tra luồng realtime."""

    def __init__(
        self,
        probabilities=None,
    ):
        self.probabilities = (
            probabilities
            or {
                "TOP_SOON": 0.15,
                "BOTTOM_SOON": 0.70,
                "NONE": 0.15,
            }
        )

        self.predict_count = 0
        self.received_data = []

    def predict_proba(self, features):
        self.predict_count += 1

        self.received_data.append(
            features.copy()
        )

        return self.probabilities.copy()


# ============================================================
# HELPERS
# ============================================================

def get_signal_label(signal):
    if isinstance(signal, str):
        return signal.upper()

    if isinstance(signal, dict):
        for key in (
            "signal",
            "label",
            "prediction",
            "class",
        ):
            if key in signal:
                return str(signal[key]).upper()

    if hasattr(signal, "signal"):
        return str(signal.signal).upper()

    if hasattr(signal, "label"):
        return str(signal.label).upper()

    raise AssertionError(
        "Không tìm thấy trường signal/label trong kết quả."
    )


def assert_valid_signal(signal):
    label = get_signal_label(signal)

    assert label in {
        "TOP_SOON",
        "BOTTOM_SOON",
        "NONE",
    }


# ============================================================
# CANDLE MANAGER
# ============================================================

def test_candle_manager_can_be_created():
    manager = CandleManager()

    assert manager is not None


def test_candle_manager_accepts_market_data(rates):
    manager = CandleManager()

    result = manager.update_candles(
        rates.copy()
    )

    assert result is not False


def test_candle_manager_returns_closed_candles(rates):
    manager = CandleManager()

    manager.update_candles(
        rates.copy()
    )

    closed = manager.get_closed_candles()

    assert isinstance(closed, pd.DataFrame)
    assert closed.index.is_monotonic_increasing
    assert closed.index.is_unique


def test_candle_manager_does_not_duplicate_candles(
    rates,
):
    manager = CandleManager()

    manager.update_candles(
        rates.copy()
    )

    first = manager.get_closed_candles()

    manager.update_candles(
        rates.copy()
    )

    second = manager.get_closed_candles()

    assert second.index.is_unique
    assert len(second) == len(first)


def test_candle_manager_accepts_new_candle(rates):
    manager = CandleManager()

    initial = rates.iloc[:-1].copy()
    new_candle = rates.iloc[[-1]].copy()

    manager.update_candles(initial)
    manager.update_candles(new_candle)

    result = manager.get_closed_candles()

    assert result.index.is_unique
    assert result.index[-1] == new_candle.index[-1]


def test_candle_manager_does_not_accept_old_candle_as_new(
    rates,
):
    manager = CandleManager()

    manager.update_candles(rates)

    before = manager.get_closed_candles()

    old_candle = rates.iloc[[0]].copy()

    manager.update_candles(old_candle)

    after = manager.get_closed_candles()

    assert after.index.is_unique
    assert after.index[-1] == before.index[-1]


def test_candle_manager_rejects_invalid_ohlc():
    manager = CandleManager()

    invalid = pd.DataFrame(
        {
            "Open": [100],
            "High": [99],
            "Low": [101],
            "Close": [100],
            "Volume": [100],
        },
        index=pd.date_range(
            "2026-01-01",
            periods=1,
            freq="5min",
        ),
    )

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        manager.update_candles(invalid)


def test_candle_manager_rejects_duplicate_timestamp(
    rates,
):
    manager = CandleManager()

    duplicated = pd.concat(
        [
            rates.iloc[:10],
            rates.iloc[[5]],
        ]
    ).sort_index()

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        manager.update_candles(duplicated)


# ============================================================
# SIGNAL ENGINE
# ============================================================

def test_signal_engine_can_be_created():
    engine = SignalEngine()

    assert engine is not None


def test_signal_engine_returns_valid_signal():
    engine = SignalEngine()

    probabilities = {
        "TOP_SOON": 0.15,
        "BOTTOM_SOON": 0.70,
        "NONE": 0.15,
    }

    signal = engine.generate_signal(
        probabilities
    )

    assert_valid_signal(signal)


def test_signal_engine_detects_bottom_probability():
    engine = SignalEngine()

    probabilities = {
        "TOP_SOON": 0.10,
        "BOTTOM_SOON": 0.80,
        "NONE": 0.10,
    }

    signal = engine.generate_signal(
        probabilities
    )

    assert get_signal_label(signal) == "BOTTOM_SOON"


def test_signal_engine_detects_top_probability():
    engine = SignalEngine()

    probabilities = {
        "TOP_SOON": 0.85,
        "BOTTOM_SOON": 0.05,
        "NONE": 0.10,
    }

    signal = engine.generate_signal(
        probabilities
    )

    assert get_signal_label(signal) == "TOP_SOON"


def test_signal_engine_returns_none_when_no_signal():
    engine = SignalEngine()

    probabilities = {
        "TOP_SOON": 0.20,
        "BOTTOM_SOON": 0.15,
        "NONE": 0.65,
    }

    signal = engine.generate_signal(
        probabilities
    )

    assert get_signal_label(signal) == "NONE"


def test_signal_engine_rejects_invalid_probabilities():
    engine = SignalEngine()

    invalid_probabilities = {
        "TOP_SOON": 0.8,
        "BOTTOM_SOON": 0.7,
        "NONE": 0.1,
    }

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        engine.generate_signal(
            invalid_probabilities
        )


def test_signal_engine_rejects_negative_probability():
    engine = SignalEngine()

    invalid_probabilities = {
        "TOP_SOON": -0.1,
        "BOTTOM_SOON": 0.5,
        "NONE": 0.6,
    }

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        engine.generate_signal(
            invalid_probabilities
        )


def test_signal_engine_rejects_nan_probability():
    engine = SignalEngine()

    invalid_probabilities = {
        "TOP_SOON": np.nan,
        "BOTTOM_SOON": 0.5,
        "NONE": 0.5,
    }

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        engine.generate_signal(
            invalid_probabilities
        )


# ============================================================
# REALTIME DATA SAFETY
# ============================================================

def test_realtime_data_is_chronological(rates):
    manager = CandleManager()

    manager.update_candles(rates)

    result = manager.get_closed_candles()

    assert result.index.is_monotonic_increasing


def test_realtime_data_has_unique_timestamps(rates):
    manager = CandleManager()

    manager.update_candles(rates)

    result = manager.get_closed_candles()

    assert result.index.is_unique


def test_realtime_data_has_valid_ohlc(rates):
    manager = CandleManager()

    manager.update_candles(rates)

    result = manager.get_closed_candles()

    assert (result["High"] >= result["Low"]).all()

    assert (
        result["High"] >= result["Open"]
    ).all()

    assert (
        result["High"] >= result["Close"]
    ).all()

    assert (
        result["Low"] <= result["Open"]
    ).all()

    assert (
        result["Low"] <= result["Close"]
    ).all()


# ============================================================
# REALTIME RUNNER
# ============================================================

def test_realtime_runner_can_be_created():
    """
    Kiểm tra constructor theo giao diện dự kiến.

    Runner nhận connector, candle manager và signal engine.
    """
    try:
        from realtime.realtime_runner import RealtimeRunner
    except ImportError:
        from ..realtime.realtime_runner import RealtimeRunner

    connector = FakeMT5Connector()
    manager = CandleManager()
    engine = SignalEngine()

    runner = RealtimeRunner(
        connector=connector,
        candle_manager=manager,
        signal_engine=engine,
    )

    assert runner is not None


def test_realtime_runner_uses_fake_connector(rates):
    """
    Runner phải có thể nhận connector giả lập,
    không bắt buộc kết nối MT5 thật.
    """
    try:
        from realtime.realtime_runner import RealtimeRunner
    except ImportError:
        from ..realtime.realtime_runner import RealtimeRunner

    connector = FakeMT5Connector(rates)

    manager = CandleManager()
    engine = SignalEngine()

    runner = RealtimeRunner(
        connector=connector,
        candle_manager=manager,
        signal_engine=engine,
    )

    assert runner is not None


def test_realtime_runner_does_not_place_orders(rates):
    """
    Project hiện tại chỉ tạo tín hiệu.
    Không được tự động gửi lệnh mua/bán.
    """
    connector = FakeMT5Connector(rates)

    manager = CandleManager()
    engine = SignalEngine()

    try:
        from realtime.realtime_runner import RealtimeRunner
    except ImportError:
        from ..realtime.realtime_runner import RealtimeRunner

    runner = RealtimeRunner(
        connector=connector,
        candle_manager=manager,
        signal_engine=engine,
    )

    # Nếu runner có phương thức chạy một chu kỳ,
    # kiểm tra nó không gọi place_order.
    if hasattr(runner, "run_once"):
        try:
            runner.run_once()
        except (RuntimeError, ValueError):
            # Lỗi dữ liệu hoặc model trong test không
            # làm thay đổi yêu cầu không gửi lệnh.
            pass

    assert connector.order_count == 0


# ============================================================
# CLOSED CANDLE POLICY
# ============================================================

def test_incomplete_candle_is_not_used_for_prediction(
    rates,
):
    """
    Kiểm tra quy tắc nến chưa đóng.

    Giao diện CandleManager cần loại bỏ nến cuối
    nếu nến đó chưa được đánh dấu là đã đóng.

    Nếu module dùng cột is_closed, test sẽ kiểm tra
    trực tiếp cột này.
    """
    data = rates.copy()

    data["is_closed"] = True
    data.iloc[
        -1,
        data.columns.get_loc("is_closed"),
    ] = False

    manager = CandleManager()
    manager.update_candles(data)

    closed = manager.get_closed_candles()

    if "is_closed" in closed.columns:
        assert closed["is_closed"].all()

    assert rates.index[-1] not in closed.index


def test_closed_candle_is_available_for_prediction(
    rates,
):
    data = rates.copy()
    data["is_closed"] = True

    manager = CandleManager()
    manager.update_candles(data)

    closed = manager.get_closed_candles()

    assert len(closed) > 0
    assert closed["is_closed"].all()


# ============================================================
# CONNECTION SAFETY
# ============================================================

def test_fake_connector_initializes():
    connector = FakeMT5Connector()

    assert connector.initialize() is True
    assert connector.connected is True


def test_fake_connector_shutdown():
    connector = FakeMT5Connector()

    connector.initialize()
    connector.shutdown()

    assert connector.connected is False


def test_fake_connector_returns_requested_number_of_rates():
    connector = FakeMT5Connector(
        make_rates(n=100)
    )

    connector.initialize()

    result = connector.get_rates(
        symbol="EURUSD",
        timeframe="M5",
        count=50,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 50


def test_fake_connector_never_places_order():
    connector = FakeMT5Connector()

    with pytest.raises(AssertionError):
        connector.place_order(
            symbol="EURUSD",
            order_type="BUY",
        )

    assert connector.order_count == 1
