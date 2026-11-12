
# File: realtime/realtime_runner.py

import json
import time
import logging

import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

from realtime.mt5_connector import (
    MT5Connector,
    MT5Config
)

from realtime.candle_manager import (
    CandleManager,
    CandleManagerConfig
)

from realtime.signal_engine import (
    SignalEngine,
    SignalEngineConfig
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class RealtimeRunnerConfig:

    symbol: str = "EURUSD"

    timeframes: tuple = ("M1", "M5", "M15")

    # Thời gian nghỉ giữa hai lần cập nhật.
    poll_interval: float = 2.0

    # Số nến tải lần đầu.
    initial_candles: int = 500

    # Số nến tải lại mỗi lần cập nhật.
    refresh_count: int = 5

    # Số nến giữ trong bộ nhớ.
    max_candles: int = 2000

    # File lưu trạng thái.
    state_path: str = (
        "models_store/realtime_state.json"
    )

    # File ghi log tín hiệu.
    signal_log_path: str = (
        "logs/realtime_signals.jsonl"
    )

    # Bật/tắt hiển thị kết quả NONE.
    print_none_signal: bool = True


# ============================================================
# 2. LOGGER
# ============================================================

def setup_logger():

    logger = logging.getLogger(
        "RealtimeRunner"
    )

    logger.setLevel(
        logging.INFO
    )

    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        console_handler = logging.StreamHandler()

        console_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            console_handler
        )

    return logger


# ============================================================
# 3. REALTIME RUNNER
# ============================================================

class RealtimeRunner:

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else RealtimeRunnerConfig()
        )

        self.logger = setup_logger()

        # MT5 connector.
        self.connector = MT5Connector(

            MT5Config(
                symbol=self.config.symbol,
                timeframes=self.config.timeframes
            )

        )

        # Candle manager.
        self.candle_manager = CandleManager(

            connector=self.connector,

            config=CandleManagerConfig(

                symbol=self.config.symbol,

                timeframes=self.config.timeframes,

                max_candles=self.config.max_candles,

                refresh_count=self.config.refresh_count

            )

        )

        # Signal engine.
        self.signal_engine = SignalEngine(

            SignalEngineConfig()

        )

        self.running = False

        self.state_path = Path(
            self.config.state_path
        )

        self.signal_log_path = Path(
            self.config.signal_log_path
        )

        self.processed_state = {

            "symbol": self.config.symbol,

            "last_processed": {}

        }

        self._load_state()

    # --------------------------------------------------------
    # LOAD PERSISTENT STATE
    # --------------------------------------------------------

    def _load_state(self):

        if not self.state_path.exists():

            self.logger.info(
                "Chưa có realtime_state.json. "
                "Bắt đầu với trạng thái mới."
            )

            return

        try:

            with open(
                self.state_path,
                "r",
                encoding="utf-8"
            ) as file:

                state = json.load(file)

            if state.get("symbol") != self.config.symbol:

                self.logger.warning(
                    "State thuộc symbol khác. "
                    "Không sử dụng trạng thái cũ."
                )

                return

            self.processed_state = state

            self.logger.info(
                "Đã tải trạng thái realtime."
            )

        except Exception as error:

            self.logger.error(
                f"Không đọc được state: {error}"
            )

    # --------------------------------------------------------
    # SAVE PERSISTENT STATE
    # --------------------------------------------------------

    def _save_state(self):

        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_path = self.state_path.with_suffix(
            ".json.tmp"
        )

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.processed_state,
                file,
                ensure_ascii=False,
                indent=4
            )

        temp_path.replace(
            self.state_path
        )

    # --------------------------------------------------------
    # RESTORE LAST PROCESSED TIMESTAMP
    # --------------------------------------------------------

    def _restore_processed_timestamp(self, timeframe):

        saved_time = self.processed_state[
            "last_processed"
        ].get(timeframe)

        if saved_time is None:

            return

        timestamp = pd.Timestamp(
            saved_time
        )

        if timestamp.tzinfo is None:

            raise ValueError(
                "Timestamp trong state thiếu timezone"
            )

        timestamp = timestamp.tz_convert(
            "UTC"
        )

        closed_df = self.candle_manager.get_closed_candles(
            timeframe
        )

        if closed_df.empty:

            return

        # Nếu timestamp đã nằm trong lịch sử hiện tại,
        # khôi phục trạng thái đã xử lý.
        if timestamp in set(
            closed_df["timestamp"]
        ):

            self.candle_manager.mark_processed(
                timeframe,
                timestamp
            )

            self.logger.info(
                f"Đã khôi phục {timeframe}: {timestamp}"
            )

        else:

            # Nếu timestamp không còn trong bộ nhớ,
            # không tự ý đánh dấu một timestamp khác.
            self.logger.warning(
                f"Không tìm thấy timestamp {timestamp} "
                f"trong lịch sử {timeframe}."
            )

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    def connect(self):

        self.logger.info(
            "Đang khởi động Realtime Runner..."
        )

        self.connector.connect()

        # Tải lịch sử ban đầu.
        for timeframe in self.config.timeframes:

            try:

                result = self.connector.get_candles(

                    symbol=self.config.symbol,

                    timeframe=timeframe,

                    count=self.config.initial_candles,

                    closed_only=True

                )

                self.candle_manager._merge_closed_candles(
                    timeframe,
                    result
                )

                self.logger.info(
                    f"Đã tải {len(result)} nến {timeframe}"
                )

            except Exception as error:

                self.logger.error(
                    f"Lỗi tải dữ liệu {timeframe}: {error}"
                )

                raise

        # Khôi phục trạng thái sau khi đã tải lịch sử.
        for timeframe in self.config.timeframes:

            self._restore_processed_timestamp(
                timeframe
            )

        self.logger.info(
            "Kết nối và khởi tạo dữ liệu hoàn tất."
        )

    # --------------------------------------------------------
    # UPDATE TIMEFRAMES
    # --------------------------------------------------------

    def update_market_data(self):

        results = {}

        for timeframe in self.config.timeframes:

            try:

                results[timeframe] = (
                    self.candle_manager.update_timeframe(
                        timeframe
                    )
                )

            except Exception as error:

                self.logger.error(
                    f"Lỗi cập nhật {timeframe}: {error}"
                )

                results[timeframe] = None

        return results

    # --------------------------------------------------------
    # BUILD HISTORICAL SLICE FOR ONE CANDLE
    # --------------------------------------------------------

    def _get_candles_until(self, timeframe, timestamp):

        candles = self.candle_manager.get_closed_candles(
            timeframe
        )

        if candles.empty:

            return candles

        # Chỉ lấy dữ liệu đến thời điểm nến cần xử lý.
        candles = candles[
            candles["timestamp"] <= timestamp
        ].copy()

        return candles.reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # LOG SIGNAL
    # --------------------------------------------------------

    def _log_signal(self, result):

        self.signal_log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.signal_log_path,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------------
    # DISPLAY SIGNAL
    # --------------------------------------------------------

    def display_signal(self, result):

        signal = result["signal"]

        if (
            signal == "NONE"
            and not self.config.print_none_signal
        ):

            return

        self.logger.info(
            "========== M5 PREDICTION =========="
        )

        self.logger.info(
            f"Symbol: {result['symbol']}"
        )

        self.logger.info(
            f"Candle: {result['candle_time']}"
        )

        self.logger.info(
            f"Signal: {signal}"
        )

        self.logger.info(
            f"Predicted class: {result['predicted_class']}"
        )

        self.logger.info(
            f"Confidence: {result['confidence']}"
        )

        self.logger.info(
            f"Probabilities: {result['probabilities']}"
        )

        self.logger.info(
            f"Calibrated: {result['calibrated']}"
        )

        self.logger.info(
            "==================================="
        )

    # --------------------------------------------------------
    # PROCESS ONE CLOSED M5 CANDLE
    # --------------------------------------------------------

    def process_m5_candle(self, candle):

        timestamp = pd.Timestamp(
            candle["timestamp"]
        )

        # Lấy lịch sử M5 đến đúng thời điểm nến cần dự đoán.
        m5_history = self._get_candles_until(
            "M5",
            timestamp
        )

        if m5_history.empty:

            raise ValueError(
                "Không có lịch sử M5 để dự đoán"
            )

        if m5_history.iloc[-1]["timestamp"] != timestamp:

            raise ValueError(
                "Nến cuối không khớp timestamp cần xử lý"
            )

        # Chạy Signal Engine.
        result = self.signal_engine.process(
            m5_history
        )

        result["symbol"] = self.config.symbol

        # Ghi log trước khi đánh dấu đã xử lý.
        self._log_signal(
            result
        )

        self.display_signal(
            result
        )

        # Chỉ đánh dấu khi dự đoán và ghi log thành công.
        self.candle_manager.mark_processed(
            "M5",
            timestamp
        )

        self.processed_state[
            "last_processed"
        ]["M5"] = timestamp.isoformat()

        self._save_state()

        return result

    # --------------------------------------------------------
    # PROCESS NEW M5 CANDLES
    # --------------------------------------------------------

    def process_new_m5_candles(self):

        new_candles = (
            self.candle_manager.get_new_closed_candles(
                "M5"
            )
        )

        if new_candles.empty:

            return []

        results = []

        for _, candle in new_candles.iterrows():

            try:

                result = self.process_m5_candle(
                    candle
                )

                results.append(
                    result
                )

            except Exception as error:

                self.logger.exception(
                    f"Không xử lý được nến M5 "
                    f"{candle['timestamp']}: {error}"
                )

                # Dừng tại nến lỗi.
                # Không bỏ qua để xử lý nến mới hơn.
                break

        return results

    # --------------------------------------------------------
    # RUN ONE LOOP
    # --------------------------------------------------------

    def run_once(self):

        # Cập nhật dữ liệu M1, M5, M15.
        self.update_market_data()

        # Dự đoán trên các nến M5 đã đóng mới.
        return self.process_new_m5_candles()

    # --------------------------------------------------------
    # RUN FOREVER
    # --------------------------------------------------------

    def run_forever(self):

        if not self.connector.connected:

            self.connect()

        self.running = True

        self.logger.info(
            "Realtime Runner đang hoạt động..."
        )

        try:

            while self.running:

                try:

                    self.run_once()

                except Exception as error:

                    self.logger.exception(
                        f"Lỗi trong vòng lặp realtime: {error}"
                    )

                time.sleep(
                    self.config.poll_interval
                )

        except KeyboardInterrupt:

            self.logger.info(
                "Nhận lệnh dừng từ bàn phím."
            )

        finally:

            self.stop()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self):

        self.running = False

        if self.connector.connected:

            self.connector.disconnect()

        self.logger.info(
            "Realtime Runner đã dừng."
        )


# ============================================================
# 4. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    config = RealtimeRunnerConfig(

        symbol="EURUSD",

        timeframes=("M1", "M5", "M15"),

        poll_interval=2.0,

        initial_candles=500,

        refresh_count=5,

        max_candles=2000

    )

    runner = RealtimeRunner(
        config
    )

    runner.run_forever()
