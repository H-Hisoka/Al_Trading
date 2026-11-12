
# File: realtime/mt5_connector.py

import MetaTrader5 as mt5
import pandas as pd

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class MT5Config:

    # Đường dẫn terminal64.exe.
    # None: sử dụng terminal MT5 mặc định.
    terminal_path: Optional[str] = None

    # Symbol giao dịch.
    symbol: str = "EURUSD"

    # Timeframes được hỗ trợ.
    timeframes: tuple = ("M1", "M5", "M15")

    # Timeout kết nối.
    timeout: int = 60000

    # Chế độ kết nối.
    portable: bool = False

    # Kiểm tra dữ liệu sau khi tải.
    validate_data: bool = True


# ============================================================
# 2. TIMEFRAME MAPPING
# ============================================================

TIMEFRAME_MAP = {

    "M1": mt5.TIMEFRAME_M1,

    "M5": mt5.TIMEFRAME_M5,

    "M15": mt5.TIMEFRAME_M15,

}


# ============================================================
# 3. MT5 CONNECTOR
# ============================================================

class MT5Connector:

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else MT5Config()
        )

        self.connected = False

        self.terminal = None

        self.account = None

        self.symbol_info = None

    # --------------------------------------------------------
    # INITIALIZE CONNECTION
    # --------------------------------------------------------

    def connect(self):

        if self.connected:
            return True

        print("Đang kết nối MetaTrader 5...")

        initialize_args = {

            "timeout": self.config.timeout,

            "portable": self.config.portable,

        }

        if self.config.terminal_path:

            success = mt5.initialize(
                self.config.terminal_path,
                **initialize_args
            )

        else:

            success = mt5.initialize(
                **initialize_args
            )

        if not success:

            error = mt5.last_error()

            self.connected = False

            raise ConnectionError(
                f"Không thể kết nối MT5: {error}"
            )

        try:

            self.terminal = mt5.terminal_info()

            self.account = mt5.account_info()

            if self.terminal is None:

                raise ConnectionError(
                    "Không lấy được thông tin terminal"
                )

            if self.account is None:

                raise ConnectionError(
                    "Không lấy được thông tin tài khoản"
                )

            if not self.terminal.connected:

                raise ConnectionError(
                    "Terminal chưa kết nối đến máy chủ"
                )

            if not self.terminal.trade_allowed:

                print(
                    "Cảnh báo: Terminal không cho phép giao dịch"
                )

            print("\n===== MT5 CONNECTION =====")

            print(
                f"Terminal: {self.terminal.name}"
            )

            print(
                f"Company: {self.terminal.company}"
            )

            print(
                f"Login: {self.account.login}"
            )

            print(
                f"Server: {self.account.server}"
            )

            print(
                f"Currency: {self.account.currency}"
            )

            self.connected = True

            return True

        except Exception:

            mt5.shutdown()

            self.connected = False

            raise

    # --------------------------------------------------------
    # DISCONNECT
    # --------------------------------------------------------

    def disconnect(self):

        if self.connected:

            mt5.shutdown()

            self.connected = False

            self.terminal = None

            self.account = None

            self.symbol_info = None

            print("Đã ngắt kết nối MT5.")

    # --------------------------------------------------------
    # CONTEXT MANAGER
    # --------------------------------------------------------

    def __enter__(self):

        self.connect()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):

        self.disconnect()

        return False

    # --------------------------------------------------------
    # CHECK CONNECTION
    # --------------------------------------------------------

    def ensure_connected(self):

        if not self.connected:

            raise RuntimeError(
                "MT5 chưa được kết nối. "
                "Hãy gọi connect() trước."
            )

        terminal = mt5.terminal_info()

        if terminal is None or not terminal.connected:

            self.connected = False

            raise ConnectionError(
                "Kết nối MT5 đã bị mất"
            )

    # --------------------------------------------------------
    # CHECK SYMBOL
    # --------------------------------------------------------

    def select_symbol(self, symbol=None):

        self.ensure_connected()

        if symbol is None:

            symbol = self.config.symbol

        info = mt5.symbol_info(symbol)

        if info is None:

            raise ValueError(
                f"Không tìm thấy symbol: {symbol}. "
                f"MT5 error: {mt5.last_error()}"
            )

        if not info.visible:

            success = mt5.symbol_select(
                symbol,
                True
            )

            if not success:

                raise RuntimeError(
                    f"Không thể bật symbol {symbol}: "
                    f"{mt5.last_error()}"
                )

        self.symbol_info = mt5.symbol_info(
            symbol
        )

        if self.symbol_info is None:

            raise RuntimeError(
                f"Không thể đọc lại thông tin {symbol}"
            )

        return self.symbol_info

    # --------------------------------------------------------
    # GET TIMEFRAME
    # --------------------------------------------------------

    @staticmethod
    def get_timeframe(timeframe):

        timeframe = timeframe.upper()

        if timeframe not in TIMEFRAME_MAP:

            raise ValueError(
                f"Timeframe không được hỗ trợ: {timeframe}. "
                f"Chỉ hỗ trợ {list(TIMEFRAME_MAP.keys())}"
            )

        return TIMEFRAME_MAP[timeframe]

    # --------------------------------------------------------
    # CONVERT TIME
    # --------------------------------------------------------

    @staticmethod
    def to_utc_datetime(value):

        if isinstance(value, pd.Timestamp):

            value = value.to_pydatetime()

        if not isinstance(value, datetime):

            raise TypeError(
                "Thời gian phải là datetime hoặc Timestamp"
            )

        if value.tzinfo is None:

            raise ValueError(
                "Datetime phải có timezone. "
                "Không tự giả định timezone."
            )

        return value.astimezone(
            timezone.utc
        )

    # --------------------------------------------------------
    # CONVERT MT5 RATES TO DATAFRAME
    # --------------------------------------------------------

    def rates_to_dataframe(self, rates):

        if rates is None:

            raise RuntimeError(
                f"MT5 không trả về dữ liệu: {mt5.last_error()}"
            )

        if len(rates) == 0:

            raise ValueError(
                "Không có dữ liệu nến được trả về"
            )

        df = pd.DataFrame(rates)

        df["timestamp"] = pd.to_datetime(
            df["time"],
            unit="s",
            utc=True
        )

        df = df.rename(
            columns={

                "tick_volume": "volume",

            }
        )

        # Giữ lại các trường bổ sung từ MT5.
        columns = [

            "timestamp",

            "open",

            "high",

            "low",

            "close",

            "volume",

            "spread",

            "real_volume",

        ]

        available_columns = [

            col for col in columns
            if col in df.columns

        ]

        df = df[available_columns]

        # Không tự xóa duplicate hoặc tạo nến giả.
        if self.config.validate_data:

            self.validate_data(df)

        return df.reset_index(drop=True)

    # --------------------------------------------------------
    # VALIDATE DATA
    # --------------------------------------------------------

    @staticmethod
    def validate_data(df):

        required_columns = [

            "timestamp",

            "open",

            "high",

            "low",

            "close",

        ]

        missing_columns = [

            col for col in required_columns
            if col not in df.columns

        ]

        if missing_columns:

            raise ValueError(
                f"Thiếu cột dữ liệu: {missing_columns}"
            )

        if df.empty:

            raise ValueError(
                "DataFrame không có dữ liệu"
            )

        if df[required_columns].isnull().any().any():

            raise ValueError(
                "Dữ liệu OHLC chứa giá trị thiếu"
            )

        if df["timestamp"].duplicated().any():

            raise ValueError(
                "Phát hiện timestamp bị trùng"
            )

        if not df["timestamp"].is_monotonic_increasing:

            raise ValueError(
                "Timestamp không tăng dần"
            )

        if not isinstance(
            df["timestamp"].dtype,
            pd.DatetimeTZDtype
        ):

            raise ValueError(
                "Timestamp phải có timezone"
            )

        if str(
            df["timestamp"].dt.tz
        ) != "UTC":

            raise ValueError(
                "Timestamp phải ở UTC"
            )

        invalid_ohlc = (

            (df["high"] < df["low"])

            | (df["high"] < df["open"])

            | (df["high"] < df["close"])

            | (df["low"] > df["open"])

            | (df["low"] > df["close"])

        )

        if invalid_ohlc.any():

            raise ValueError(
                "Phát hiện nến có cấu trúc OHLC không hợp lệ"
            )

        numeric_columns = [

            "open",

            "high",

            "low",

            "close",

        ]

        if not df[numeric_columns].map(
            lambda x: pd.notna(x) and abs(x) != float("inf")
        ).all().all():

            raise ValueError(
                "OHLC chứa infinity hoặc giá trị không hợp lệ"
            )

        return True

    # --------------------------------------------------------
    # GET CANDLES BY POSITION
    # --------------------------------------------------------

    def get_candles(
        self,
        symbol=None,
        timeframe="M5",
        count=500,
        closed_only=True
    ):

        self.ensure_connected()

        if count <= 0:

            raise ValueError(
                "count phải lớn hơn 0"
            )

        if symbol is None:

            symbol = self.config.symbol

        self.select_symbol(symbol)

        mt5_timeframe = self.get_timeframe(
            timeframe
        )

        # MT5:
        # start_pos = 0: nến hiện tại đang chạy.
        # start_pos = 1: nến mới nhất đã đóng.
        start_pos = (
            1 if closed_only else 0
        )

        rates = mt5.copy_rates_from_pos(

            symbol,

            mt5_timeframe,

            start_pos,

            count

        )

        df = self.rates_to_dataframe(
            rates
        )

        return df

    # --------------------------------------------------------
    # GET CANDLES BY TIME RANGE
    # --------------------------------------------------------

    def get_candles_range(
        self,
        start_time,
        end_time,
        symbol=None,
        timeframe="M5"
    ):

        self.ensure_connected()

        if symbol is None:

            symbol = self.config.symbol

        self.select_symbol(symbol)

        start_time = self.to_utc_datetime(
            start_time
        )

        end_time = self.to_utc_datetime(
            end_time
        )

        if start_time >= end_time:

            raise ValueError(
                "start_time phải nhỏ hơn end_time"
            )

        mt5_timeframe = self.get_timeframe(
            timeframe
        )

        rates = mt5.copy_rates_range(

            symbol,

            mt5_timeframe,

            start_time,

            end_time

        )

        return self.rates_to_dataframe(
            rates
        )

    # --------------------------------------------------------
    # GET MULTI TIMEFRAME DATA
    # --------------------------------------------------------

    def get_multi_timeframe(
        self,
        symbol=None,
        count=500,
        closed_only=True
    ):

        if symbol is None:

            symbol = self.config.symbol

        result = {}

        for timeframe in self.config.timeframes:

            result[timeframe] = self.get_candles(

                symbol=symbol,

                timeframe=timeframe,

                count=count,

                closed_only=closed_only

            )

        return result

    # --------------------------------------------------------
    # GET LATEST TICK
    # --------------------------------------------------------

    def get_latest_tick(self, symbol=None):

        self.ensure_connected()

        if symbol is None:

            symbol = self.config.symbol

        self.select_symbol(symbol)

        tick = mt5.symbol_info_tick(
            symbol
        )

        if tick is None:

            raise RuntimeError(
                f"Không lấy được tick: {mt5.last_error()}"
            )

        return {

            "time": datetime.fromtimestamp(
                tick.time,
                tz=timezone.utc
            ),

            "bid": tick.bid,

            "ask": tick.ask,

            "last": tick.last,

            "volume": tick.volume,

        }
