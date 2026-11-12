
# File: realtime/signal_engine.py

import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from features.feature_builder import build_features


# ============================================================
# 1. CONSTANTS
# ============================================================

EXPECTED_CLASSES = [
    "TOP_SOON",
    "BOTTOM_SOON",
    "NONE",
]


# ============================================================
# 2. CONFIGURATION
# ============================================================

@dataclass
class SignalEngineConfig:

    model_path: str = "models_store/swing_model.joblib"

    # Xác suất tối thiểu để phát tín hiệu định hướng.
    min_directional_probability: float = 0.60

    # Số nến tối thiểu cần có để tính feature.
    min_candles: int = 100

    # Số chữ số hiển thị xác suất.
    probability_digits: int = 4

    # Bắt buộc chỉ nhận nến đã đóng.
    require_closed_candles: bool = True


# ============================================================
# 3. SIGNAL ENGINE
# ============================================================

class SignalEngine:

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else SignalEngineConfig()
        )

        self.artifact = None
        self.model = None
        self.calibrator = None

        self.feature_columns = None

        self.load_model()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def load_model(self):

        model_path = Path(
            self.config.model_path
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Không tìm thấy model: {model_path}"
            )

        artifact = joblib.load(
            model_path
        )

        if not isinstance(artifact, dict):

            raise ValueError(
                "Model artifact không đúng định dạng"
            )

        if "model" not in artifact:

            raise ValueError(
                "Artifact không có model"
            )

        if "metadata" not in artifact:

            raise ValueError(
                "Artifact không có metadata"
            )

        metadata = artifact["metadata"]

        feature_columns = metadata.get(
            "feature_columns"
        )

        if not feature_columns:

            raise ValueError(
                "Artifact thiếu feature_columns"
            )

        self.artifact = artifact

        self.model = artifact["model"]

        self.calibrator = artifact.get(
            "calibrator"
        )

        self.feature_columns = list(
            feature_columns
        )

        model_classes = list(
            self.model.classes_
        )

        if set(model_classes) != set(
            EXPECTED_CLASSES
        ):

            raise ValueError(
                f"Model classes không hợp lệ: {model_classes}"
            )

        print(
            "Signal Engine đã tải model thành công."
        )

        if self.calibrator is not None:

            print(
                "Đã tìm thấy calibration model."
            )

        else:

            print(
                "Cảnh báo: Model chưa được calibration."
            )

        return True

    # --------------------------------------------------------
    # RELOAD MODEL
    # --------------------------------------------------------

    def reload_model(self):

        return self.load_model()

    # --------------------------------------------------------
    # VALIDATE CANDLES
    # --------------------------------------------------------

    def _validate_candles(self, candles):

        if not isinstance(candles, pd.DataFrame):

            raise TypeError(
                "candles phải là pandas DataFrame"
            )

        if candles.empty:

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
            if col not in candles.columns
        ]

        if missing:

            raise ValueError(
                f"Thiếu cột: {missing}"
            )

        if len(candles) < self.config.min_candles:

            raise ValueError(
                f"Cần ít nhất {self.config.min_candles} nến, "
                f"hiện có {len(candles)}"
            )

        if candles["timestamp"].isnull().any():

            raise ValueError(
                "Timestamp chứa giá trị thiếu"
            )

        if candles["timestamp"].duplicated().any():

            raise ValueError(
                "Timestamp bị trùng"
            )

        if not candles["timestamp"].is_monotonic_increasing:

            raise ValueError(
                "Nến phải được sắp xếp theo thời gian tăng dần"
            )

        if not isinstance(
            candles["timestamp"].dtype,
            pd.DatetimeTZDtype
        ):

            raise ValueError(
                "Timestamp phải có timezone"
            )

        if str(candles["timestamp"].dt.tz) != "UTC":

            raise ValueError(
                "Timestamp phải được chuẩn hóa UTC"
            )

        return True

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    def build_latest_features(
        self,
        candles,
        swings=None
    ):

        self._validate_candles(
            candles
        )

        # Chỉ sử dụng dữ liệu hiện tại và quá khứ.
        features = build_features(
            candles.copy(),
            swings=swings
        )

        if not isinstance(features, pd.DataFrame):

            raise TypeError(
                "build_features phải trả về DataFrame"
            )

        if features.empty:

            raise ValueError(
                "Feature Builder trả về dữ liệu rỗng"
            )

        missing_columns = [
            col for col in self.feature_columns
            if col not in features.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Feature Builder thiếu các cột: "
                f"{missing_columns}"
            )

        # Giữ đúng thứ tự feature khi train.
        X = features[
            self.feature_columns
        ].copy()

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Lấy feature của nến cuối cùng.
        latest_features = X.tail(1)

        if latest_features.empty:

            raise ValueError(
                "Không tạo được feature cho nến mới nhất"
            )

        return latest_features

    # --------------------------------------------------------
    # MAP MODEL PROBABILITIES
    # --------------------------------------------------------

    def _map_probabilities(self, raw_probabilities):

        model_classes = list(
            self.model.classes_
        )

        raw_probabilities = np.asarray(
            raw_probabilities,
            dtype=float
        )

        if raw_probabilities.ndim != 2:

            raise ValueError(
                "predict_proba phải trả về ma trận 2 chiều"
            )

        if raw_probabilities.shape[1] != len(
            model_classes
        ):

            raise ValueError(
                "Số cột xác suất không khớp model classes"
            )

        class_indices = {
            name: index
            for index, name in enumerate(
                model_classes
            )
        }

        ordered = np.column_stack([

            raw_probabilities[
                :,
                class_indices[class_name]
            ]

            for class_name in EXPECTED_CLASSES

        ])

        if not np.isfinite(ordered).all():

            raise ValueError(
                "Xác suất chứa NaN hoặc infinity"
            )

        if np.any(ordered < 0):

            raise ValueError(
                "Xác suất không được âm"
            )

        row_sums = ordered.sum(
            axis=1,
            keepdims=True
        )

        if np.any(row_sums <= 0):

            raise ValueError(
                "Tổng xác suất không hợp lệ"
            )

        return ordered / row_sums

    # --------------------------------------------------------
    # PREDICT PROBABILITIES
    # --------------------------------------------------------

    def predict_probabilities(self, X):

        raw = self.model.predict_proba(
            X
        )

        probabilities = self._map_probabilities(
            raw
        )

        if self.calibrator is not None:

            probabilities = (
                self.calibrator.predict_proba(
                    probabilities
                )
            )

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if not np.isfinite(probabilities).all():

            raise ValueError(
                "Calibration trả về xác suất không hợp lệ"
            )

        if np.any(probabilities < 0):

            raise ValueError(
                "Calibration trả về xác suất âm"
            )

        row_sums = probabilities.sum(
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-5
        ):

            raise ValueError(
                "Tổng xác suất sau calibration không bằng 1"
            )

        return probabilities

    # --------------------------------------------------------
    # DECIDE SIGNAL
    # --------------------------------------------------------

    def decide_signal(self, probabilities):

        probability_map = {

            class_name: float(
                probabilities[index]
            )

            for index, class_name in enumerate(
                EXPECTED_CLASSES
            )

        }

        predicted_class = max(
            probability_map,
            key=probability_map.get
        )

        confidence = probability_map[
            predicted_class
        ]

        if predicted_class == "NONE":

            signal = "NONE"

        elif confidence < (
            self.config.min_directional_probability
        ):

            signal = "NO_SIGNAL"

        else:

            signal = predicted_class

        return {

            "signal": signal,

            "predicted_class": predicted_class,

            "confidence": confidence,

            "probabilities": probability_map,

        }

    # --------------------------------------------------------
    # PROCESS CANDLES
    # --------------------------------------------------------

    def process(
        self,
        candles,
        swings=None
    ):

        if self.model is None:

            raise RuntimeError(
                "Model chưa được tải"
            )

        self._validate_candles(
            candles
        )

        latest_candle = candles.iloc[-1]

        candle_time = pd.Timestamp(
            latest_candle["timestamp"]
        )

        # Tạo feature.
        X = self.build_latest_features(
            candles,
            swings=swings
        )

        # Dự đoán xác suất.
        probabilities = (
            self.predict_probabilities(X)[0]
        )

        # Xác định tín hiệu.
        decision = self.decide_signal(
            probabilities
        )

        result = {

            "symbol": None,

            "timeframe": "M5",

            "candle_time": candle_time.isoformat(),

            "prediction_time": datetime.now(
                timezone.utc
            ).isoformat(),

            "signal": decision["signal"],

            "predicted_class": (
                decision["predicted_class"]
            ),

            "confidence": round(
                decision["confidence"],
                self.config.probability_digits
            ),

            "probabilities": {

                key: round(
                    value,
                    self.config.probability_digits
                )

                for key, value in (
                    decision["probabilities"].items()
                )

            },

            "calibrated": (
                self.calibrator is not None
            ),

        }

        return result
