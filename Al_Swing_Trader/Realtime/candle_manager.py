
# File: realtime/candle_manager.py

import pandas as pd

from dataclasses import dataclass, field
from typing import Dict, Optional

from realtime.mt5_connector import (
    MT5Connector,
    MT5Config
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

TIMEFRAME_MINUTES = {

    "M1": 1,

    "M5": 5,

    "M15": 15,

}


@dataclass
class CandleManagerConfig:

    symbol: str = "EURUSD"

    timeframes: tuple = ("M1", "M5", "M15")

    # Số nến tối đa giữ trong bộ nhớ.
    max_candles: int = 2000

    # Số nến tải lại mỗi lần cập nhật.
    refresh_count: int = 5

    # Chỉ quản lý nến đóng nếu True.
    validate_data: bool = True


# ============================================================
# 2. CANDLE MANAGER
# ============================================================

class CandleManager:

    def __init__(
        self,
        connector: MT5Connector,
        config=None
    ):

        self.connector = connector

        self.config = (
            config
            if config is not None
            else CandleManagerConfig()
        )

        # Lịch sử các nến đã đóng.
        self.closed_candles = {

            timeframe: pd.DataFrame()

            for timeframe in self.config.timeframes

        }

        # Nến hiện tại đang chạy.
        self.current_candles = {

            timeframe: None

            for timeframe in self.config.timeframes

        }

        # Timestamp nến đóng gần nhất đã được xử lý.
        self.last_processed = {

            timeframe: None

            for timeframe in self.config.timeframes

        }

    # --------------------------------------------------------
    # CHECK TIMEFRAME
    # --------------------------------------------------------

    def _validate_timeframe(self, timeframe):

        if timeframe not in TIMEFRAME_MINUTES:

            raise ValueError(
                f"Timeframe không hợp lệ: {timeframe}"
            )

        if timeframe not in self.config.timeframes:

            raise ValueError(
                f"Timeframe {timeframe} chưa được bật "
                f"trong CandleManagerConfig"
            )

    # --------------------------------------------------------
    # VALIDATE CANDLES
    # --------------------------------------------------------

    def _validate_candles(self, df):

        if df is None or df.empty:

            raise ValueError(
                "Không có dữ liệu nến"
            )

        required_columns = [

            "timestamp",

            "open",

            "high",

            "low",

            "close",

        ]

        missing = [

            col for col in required_columns
            if col not in df.columns

        ]

        if missing:

            raise ValueError(
                f"Thiếu cột: {missing}"
            )

        if df["timestamp"].duplicated().any():

            raise ValueError(
                "Dữ liệu đầu vào chứa timestamp trùng"
            )

        if not df["timestamp"].is_monotonic_increasing:

            raise ValueError(
                "Timestamp phải tăng dần"
            )

        if df[required_columns].isnull().any().any():

            raise ValueError(
                "Dữ liệu nến chứa giá trị thiếu"
            )

        invalid = (

            (df["high"] < df["low"])

            | (df["high"] < df["open"])

            | (df["high"] < df["close"])

            | (df["low"] > df["open"])

            | (df["low"] > df["close"])

        )

        if invalid.any():

            raise ValueError(
                "Phát hiện cấu trúc OHLC không hợp lệ"
            )

        return True

    # --------------------------------------------------------
    # MERGE CLOSED CANDLES
    # --------------------------------------------------------

    def _merge_closed_candles(
        self,
        timeframe,
        new_candles
    ):

        old_candles = self.closed_candles[
            timeframe
        ]

        if old_candles.empty:

            merged = new_candles.copy()

        else:

            merged = pd.concat(

                [
                    old_candles,
                    new_candles
                ],

                ignore_index=True

            )

            # Các nến cũ được thay bằng dữ liệu mới
            # nếu cùng timestamp.
            merged = merged.drop_duplicates(

                subset=["timestamp"],

                keep="last"

            )

            merged = merged.sort_values(
                "timestamp"
            )

        # Giới hạn dung lượng bộ nhớ.
        if len(merged) > self.config.max_candles:

            merged = merged.tail(
                self.config.max_candles
            )

        merged = merged.reset_index(
            drop=True
        )

        if self.config.validate_data:

            self._validate_candles(
                merged
            )

        self.closed_candles[
            timeframe
        ] = merged

        return merged

    # --------------------------------------------------------
    # UPDATE ONE TIMEFRAME
    # --------------------------------------------------------

    def update_timeframe(self, timeframe):

        self._validate_timeframe(
            timeframe
        )

        if not self.connector.connected:

            raise RuntimeError(
                "MT5 chưa kết nối"
            )

        # Lấy cả nến đang chạy.
        df = self.connector.get_candles(

            symbol=self.config.symbol,

            timeframe=timeframe,

            count=self.config.refresh_count,

            closed_only=False

        )

        if df.empty:

            raise RuntimeError(
                f"Không có dữ liệu {timeframe}"
            )

        if self.config.validate_data:

            self._validate_candles(df)

        # MT5 trả dữ liệu theo thứ tự thời gian tăng dần.
        # Nến cuối cùng là nến đang chạy.
        if len(df) == 1:

            closed_part = df.iloc[0:0].copy()

            current_part = df.iloc[-1].copy()

        else:

            closed_part = df.iloc[:-1].copy()

            current_part = df.iloc[-1].copy()

        # Cập nhật nến đang chạy.
        self.current_candles[
            timeframe
        ] = current_part

        # Cập nhật lịch sử nến đóng.
        if not closed_part.empty:

            self._merge_closed_candles(

                timeframe,

                closed_part

            )

        # Kiểm tra nến đóng mới.
        closed_df = self.closed_candles[
            timeframe
        ]

        if closed_df.empty:

            new_closed_candles = closed_df.copy()

        else:

            last_time = self.last_processed[
                timeframe
            ]

            if last_time is None:

                new_closed_candles = closed_df.tail(
                    1
                ).copy()

            else:

                new_closed_candles = closed_df[
                    closed_df["timestamp"] > last_time
                ].copy()

        return {

            "timeframe": timeframe,

            "current_candle": current_part,

            "closed_candles": closed_df.copy(),

            "new_closed_candles": (
                new_closed_candles.copy()
            ),

        }

    # --------------------------------------------------------
    # UPDATE ALL TIMEFRAMES
    # --------------------------------------------------------

    def update_all(self):

        result = {}

        for timeframe in self.config.timeframes:

            result[timeframe] = (
                self.update_timeframe(
                    timeframe
                )
            )

        return result

    # --------------------------------------------------------
    # MARK CANDLE AS PROCESSED
    # --------------------------------------------------------

    def mark_processed(
        self,
        timeframe,
        timestamp
    ):

        self._validate_timeframe(
            timeframe
        )

        timestamp = pd.Timestamp(
            timestamp
        )

        if timestamp.tzinfo is None:

            raise ValueError(
                "Timestamp phải có timezone"
            )

        timestamp = timestamp.tz_convert(
            "UTC"
        )

        previous = self.last_processed[
            timeframe
        ]

        if previous is not None:

            if timestamp <= previous:

                raise ValueError(
                    "Không thể đánh dấu lại nến cũ "
                    "hoặc nến đã xử lý"
                )

        closed_df = self.closed_candles[
            timeframe
        ]

        if closed_df.empty:

            raise ValueError(
                "Chưa có lịch sử nến đóng"
            )

        if timestamp not in set(
            closed_df["timestamp"]
        ):

            raise ValueError(
                "Timestamp không nằm trong lịch sử "
                "nến đã đóng"
            )

        self.last_processed[
            timeframe
        ] = timestamp

    # --------------------------------------------------------
    # GET CLOSED CANDLES
    # --------------------------------------------------------

    def get_closed_candles(
        self,
        timeframe,
        count=None
    ):

        self._validate_timeframe(
            timeframe
        )

        df = self.closed_candles[
            timeframe
        ].copy()

        if count is not None:

            df = df.tail(count)

        return df.reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # GET CURRENT CANDLE
    # --------------------------------------------------------

    def get_current_candle(self, timeframe):

        self._validate_timeframe(
            timeframe
        )

        candle = self.current_candles[
            timeframe
        ]

        if candle is None:

            return None

        return candle.copy()

    # --------------------------------------------------------
    # GET LATEST CLOSED CANDLE
    # --------------------------------------------------------

    def get_latest_closed_candle(self, timeframe):

        self._validate_timeframe(
            timeframe
        )

        df = self.closed_candles[
            timeframe
        ]

        if df.empty:

            return None

        return df.iloc[-1].copy()

    # --------------------------------------------------------
    # GET NEW CLOSED CANDLES
    # --------------------------------------------------------

    def get_new_closed_candles(self, timeframe):

        self._validate_timeframe(
            timeframe
        )

        df = self.closed_candles[
            timeframe
        ]

        if df.empty:

            return df.copy()

        last_time = self.last_processed[
            timeframe
        ]

        if last_time is None:

            return df.tail(1).copy()

        return df[
            df["timestamp"] > last_time
        ].copy()

    # --------------------------------------------------------
    # GET ALL TIMEFRAME DATA
    # --------------------------------------------------------

    def get_all_closed_candles(self):

        return {

            timeframe: self.get_closed_candles(
                timeframe
            )

            for timeframe in self.config.timeframes

        }

    # --------------------------------------------------------
    # RESET PROCESSING STATE
    # --------------------------------------------------------

    def reset_processing_state(self, timeframe=None):

        if timeframe is not None:

            self._validate_timeframe(
                timeframe
            )

            self.last_processed[
                timeframe
            ] = None

            return

        for key in self.last_processed:

            self.last_processed[key] = None

    # --------------------------------------------------------
    # CLEAR MEMORY
    # --------------------------------------------------------

    def clear(self):

        for timeframe in self.config.timeframes:

            self.closed_candles[
                timeframe
            ] = pd.DataFrame()

            self.current_candles[
                timeframe
            ] = None

            self.last_processed[
                timeframe
            ] = None
