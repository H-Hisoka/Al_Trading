
# File: features/supply_demand.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SupplyDemandConfig:
    """
    Cấu hình phát hiện Supply/Demand zones.
    """

    atr_column: str = "atr_14"

    # Số nến tối đa tìm ngược để xác định nến nền.
    base_lookback: int = 5

    # Số nến tối đa để xác nhận một nhịp displacement.
    displacement_lookback: int = 3

    # Độ lớn tối thiểu của nến displacement tính theo ATR.
    min_displacement_atr: float = 1.5

    # Tỷ lệ thân nến / toàn bộ biên độ.
    min_body_ratio: float = 0.60

    # Chỉ cần đóng cửa vượt khỏi vùng nến nền.
    require_close_break: bool = True

    # Giới hạn số vùng được giữ lại.
    max_zones: int = 100

    # Vùng bị coi là phá khi giá đóng cửa vượt khỏi biên zone.
    invalidate_on_close: bool = True

    # Cho phép một vùng được chạm nhiều lần.
    track_multiple_touches: bool = True


def _prepare_candles(
    candles: pd.DataFrame,
    config: SupplyDemandConfig,
) -> pd.DataFrame:
    required = {
        "open",
        "high",
        "low",
        "close",
        config.atr_column,
    }

    missing = required - set(candles.columns)

    if missing:
        raise ValueError(
            f"Thiếu các cột: {sorted(missing)}"
        )

    df = candles.copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[list(required)].isna().any().any():
        raise ValueError(
            "Dữ liệu OHLC hoặc ATR có giá trị không hợp lệ"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Dữ liệu phải được sắp xếp theo thời gian tăng dần"
        )

    if (df["high"] < df["low"]).any():
        raise ValueError("Có nến high nhỏ hơn low")

    return df


def _get_candle_direction(
    candle: pd.Series,
) -> str:
    """
    Xác định hướng nến.
    """
    if candle["close"] > candle["open"]:
        return "BULLISH"

    if candle["close"] < candle["open"]:
        return "BEARISH"

    return "NEUTRAL"


def _get_body_ratio(
    candle: pd.Series,
) -> float:
    candle_range = candle["high"] - candle["low"]

    if candle_range <= 0:
        return 0.0

    body = abs(
        candle["close"] - candle["open"]
    )

    return float(body / candle_range)


def _is_displacement(
    candle: pd.Series,
    direction: str,
    config: SupplyDemandConfig,
) -> bool:
    """
    Kiểm tra một nến có phải nến displacement hay không.

    Điều kiện:
    1. Đúng hướng.
    2. Biên độ nến đủ lớn so với ATR.
    3. Thân nến chiếm tỷ lệ đủ lớn.
    """
    atr = float(candle[config.atr_column])

    if atr <= 0:
        return False

    candle_range = float(
        candle["high"] - candle["low"]
    )

    if candle_range < config.min_displacement_atr * atr:
        return False

    if _get_body_ratio(candle) < config.min_body_ratio:
        return False

    candle_direction = _get_candle_direction(candle)

    if direction == "DEMAND":
        return candle_direction == "BULLISH"

    if direction == "SUPPLY":
        return candle_direction == "BEARISH"

    return False


def _find_base_candle(
    df: pd.DataFrame,
    displacement_position: int,
    direction: str,
    config: SupplyDemandConfig,
) -> Optional[int]:
    """
    Tìm nến nền gần nhất trước displacement.

    DEMAND:
        Ưu tiên nến giảm hoặc nến trung tính.

    SUPPLY:
        Ưu tiên nến tăng hoặc nến trung tính.
    """
    start = max(
        0,
        displacement_position - config.base_lookback,
    )

    for position in range(
        displacement_position - 1,
        start - 1,
        -1,
    ):
        candle = df.iloc[position]

        candle_direction = _get_candle_direction(candle)

        if direction == "DEMAND":
            if candle_direction in ["BEARISH", "NEUTRAL"]:
                return position

        elif direction == "SUPPLY":
            if candle_direction in ["BULLISH", "NEUTRAL"]:
                return position

    return None


def _build_zone(
    df: pd.DataFrame,
    base_position: int,
    displacement_position: int,
    direction: str,
    zone_id: int,
) -> dict:
    """
    Tạo vùng Supply/Demand từ nến nền.
    """
    base = df.iloc[base_position]
    displacement = df.iloc[displacement_position]

    if direction == "DEMAND":
        zone_low = float(base["low"])
        zone_high = float(
            max(base["open"], base["close"])
        )

    else:
        zone_low = float(
            min(base["open"], base["close"])
        )
        zone_high = float(base["high"])

    # Phòng trường hợp nến nền không có thân.
    if zone_high <= zone_low:
        zone_low = float(base["low"])
        zone_high = float(base["high"])

    return {
        "zone_id": zone_id,
        "zone_type": direction,

        "zone_low": zone_low,
        "zone_high": zone_high,

        "base_time": df.index[base_position],
        "zone_created_time": df.index[displacement_position],

        "base_position": base_position,
        "created_position": displacement_position,

        "displacement_atr": (
            abs(
                float(displacement["close"])
                - float(displacement["open"])
            )
            / float(displacement["atr_14"])
            if "atr_14" in df.columns
            and float(displacement["atr_14"]) > 0
            else np.nan
        ),

        "status": "ACTIVE",
        "touch_count": 0,
        "first_touch_time": pd.NaT,
        "last_touch_time": pd.NaT,
        "invalidated_time": pd.NaT,
    }


def create_supply_demand_zones(
    candles: pd.DataFrame,
    config: Optional[SupplyDemandConfig] = None,
) -> pd.DataFrame:
    """
    Phát hiện các vùng Supply/Demand.

    Trả về DataFrame chứa:
    - zone_id
    - zone_type
    - zone_low
    - zone_high
    - base_time
    - zone_created_time
    - status
    - touch_count
    - first_touch_time
    - last_touch_time
    - invalidated_time

    Lưu ý:
    Vùng chỉ được tạo tại thời điểm displacement xảy ra.
    Không được sử dụng zone_created_time để gán vùng
    ngược về các nến trước đó.
    """
    if config is None:
        config = SupplyDemandConfig()

    df = _prepare_candles(candles, config)

    zones = []
    zone_id = 1

    for position in range(len(df)):
        candle = df.iloc[position]

        if position == 0:
            continue

        # Chỉ xét displacement đã hoàn thành.
        directions = ["DEMAND", "SUPPLY"]

        for direction in directions:
            if not _is_displacement(
                candle,
                direction,
                config,
            ):
                continue

            base_position = _find_base_candle(
                df,
                position,
                direction,
                config,
            )

            if base_position is None:
                continue

            base = df.iloc[base_position]

            # Xác nhận displacement thực sự rời khỏi nến nền.
            if config.require_close_break:
                if direction == "DEMAND":
                    if candle["close"] <= base["high"]:
                        continue

                else:
                    if candle["close"] >= base["low"]:
                        continue

            zone = _build_zone(
                df=df,
                base_position=base_position,
                displacement_position=position,
                direction=direction,
                zone_id=zone_id,
            )

            # Đảm bảo không có zone trùng hoàn toàn
            # cùng loại và cùng nến nền.
            duplicate = any(
                existing["zone_type"] == zone["zone_type"]
                and existing["base_time"] == zone["base_time"]
                for existing in zones
            )

            if duplicate:
                continue

            zones.append(zone)
            zone_id += 1

    if not zones:
        return pd.DataFrame(
            columns=[
                "zone_id",
                "zone_type",
                "zone_low",
                "zone_high",
                "base_time",
                "zone_created_time",
                "base_position",
                "created_position",
                "displacement_atr",
                "status",
                "touch_count",
                "first_touch_time",
                "last_touch_time",
                "invalidated_time",
            ]
        )

    return pd.DataFrame(zones).tail(
        config.max_zones
    ).reset_index(drop=True)


def update_zone_status(
    candles: pd.DataFrame,
    zones: pd.DataFrame,
    config: Optional[SupplyDemandConfig] = None,
) -> pd.DataFrame:
    """
    Theo dõi trạng thái các vùng Supply/Demand.

    ACTIVE:
        Chưa bị chạm hoặc phá.

    MITIGATED:
        Giá đã quay lại vùng.

    INVALIDATED:
        Giá đã phá vùng.

    Một vùng đã bị phá sẽ không được kích hoạt lại.
    """
    if config is None:
        config = SupplyDemandConfig()

    df = _prepare_candles(candles, config)

    if zones.empty:
        return zones.copy()

    result = zones.copy()

    for index, zone in result.iterrows():
        zone_low = float(zone["zone_low"])
        zone_high = float(zone["zone_high"])

        zone_type = zone["zone_type"]

        created_time = pd.Timestamp(
            zone["zone_created_time"]
        )

        status = "ACTIVE"
        touch_count = 0

        first_touch_time = pd.NaT
        last_touch_time = pd.NaT
        invalidated_time = pd.NaT

        # Chỉ theo dõi từ thời điểm vùng được tạo.
        future_candles = df.loc[
            df.index > created_time
        ]

        for candle_time, candle in future_candles.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            close = float(candle["close"])

            # Kiểm tra giá có đi vào vùng hay không.
            touched = (
                high >= zone_low
                and low <= zone_high
            )

            # Kiểm tra vùng bị phá.
            if config.invalidate_on_close:
                if zone_type == "DEMAND":
                    invalidated = close < zone_low
                else:
                    invalidated = close > zone_high

            else:
                if zone_type == "DEMAND":
                    invalidated = low < zone_low
                else:
                    invalidated = high > zone_high

            # Ưu tiên kiểm tra phá vùng trước.
            if invalidated:
                status = "INVALIDATED"
                invalidated_time = candle_time
                break

            if touched:
                if status == "ACTIVE":
                    status = "MITIGATED"
                    first_touch_time = candle_time

                touch_count += 1
                last_touch_time = candle_time

                if not config.track_multiple_touches:
                    break

        result.at[index, "status"] = status
        result.at[index, "touch_count"] = touch_count

        result.at[index, "first_touch_time"] = (
            first_touch_time
        )

        result.at[index, "last_touch_time"] = (
            last_touch_time
        )

        result.at[index, "invalidated_time"] = (
            invalidated_time
        )

    return result


def add_supply_demand_features(
    candles: pd.DataFrame,
    zones: pd.DataFrame,
    config: Optional[SupplyDemandConfig] = None,
) -> pd.DataFrame:
    """
    Bổ sung đặc trưng Supply/Demand vào từng nến.

    Chỉ sử dụng những vùng đã được tạo tại hoặc trước
    thời điểm của nến đang xét.

    Không dùng trạng thái được tính từ toàn bộ tương lai
    để gán ngược vào dữ liệu quá khứ.
    """
    if config is None:
        config = SupplyDemandConfig()

    df = _prepare_candles(candles, config)

    result = df.copy()

    result["nearest_demand_distance_atr"] = np.nan
    result["nearest_supply_distance_atr"] = np.nan

    result["inside_demand_zone"] = 0
    result["inside_supply_zone"] = 0

    result["demand_zone_count"] = 0
    result["supply_zone_count"] = 0

    result["nearest_demand_zone_id"] = np.nan
    result["nearest_supply_zone_id"] = np.nan

    if zones.empty:
        return result

    for candle_time, candle in result.iterrows():
        current_price = float(candle["close"])

        atr = float(candle[config.atr_column])

        if atr <= 0:
            continue

        # Chỉ dùng zone đã được tạo tại thời điểm hiện tại.
        known_zones = zones[
            pd.to_datetime(zones["zone_created_time"])
            <= candle_time
        ]

        if "invalidated_time" in known_zones.columns:
            invalidated = pd.to_datetime(
                known_zones["invalidated_time"],
                errors="coerce",
            )

            known_zones = known_zones[
                invalidated.isna()
                | (invalidated > candle_time)
            ]

        demands = known_zones[
            known_zones["zone_type"] == "DEMAND"
        ]

        supplies = known_zones[
            known_zones["zone_type"] == "SUPPLY"
        ]

        result.at[
            candle_time,
            "demand_zone_count"
        ] = len(demands)

        result.at[
            candle_time,
            "supply_zone_count"
        ] = len(supplies)

        # Vùng Demand gần nhất.
        if not demands.empty:
            distances = []

            for _, zone in demands.iterrows():
                low = float(zone["zone_low"])
                high = float(zone["zone_high"])

                if low <= current_price <= high:
                    distance = 0.0
                    result.at[
                        candle_time,
                        "inside_demand_zone"
                    ] = 1

                elif current_price > high:
                    distance = current_price - high

                else:
                    distance = low - current_price

                distances.append(
                    (distance, zone["zone_id"])
                )

            distance, nearest_id = min(
                distances,
                key=lambda item: item[0],
            )

            result.at[
                candle_time,
                "nearest_demand_distance_atr"
            ] = distance / atr

            result.at[
                candle_time,
                "nearest_demand_zone_id"
            ] = nearest_id

        # Vùng Supply gần nhất.
        if not supplies.empty:
            distances = []

            for _, zone in supplies.iterrows():
                low = float(zone["zone_low"])
                high = float(zone["zone_high"])

                if low <= current_price <= high:
                    distance = 0.0
                    result.at[
                        candle_time,
                        "inside_supply_zone"
                    ] = 1

                elif current_price > high:
                    distance = current_price - high

                else:
                    distance = low - current_price

                distances.append(
                    (distance, zone["zone_id"])
                )

            distance, nearest_id = min(
                distances,
                key=lambda item: item[0],
            )

            result.at[
                candle_time,
                "nearest_supply_distance_atr"
            ] = distance / atr

            result.at[
                candle_time,
                "nearest_supply_zone_id"
            ] = nearest_id

    return result


def summarize_supply_demand(
    zones: pd.DataFrame,
) -> dict:
    """
    Thống kê các vùng Supply/Demand.
    """
    if zones.empty:
        return {
            "total_zones": 0,
            "demand_zones": 0,
            "supply_zones": 0,
            "active_zones": 0,
            "mitigated_zones": 0,
            "invalidated_zones": 0,
        }

    return {
        "total_zones": len(zones),

        "demand_zones": int(
            zones["zone_type"].eq("DEMAND").sum()
        ),

        "supply_zones": int(
            zones["zone_type"].eq("SUPPLY").sum()
        ),

        "active_zones": int(
            zones["status"].eq("ACTIVE").sum()
        ),

        "mitigated_zones": int(
            zones["status"].eq("MITIGATED").sum()
        ),

        "invalidated_zones": int(
            zones["status"].eq("INVALIDATED").sum()
        ),
    }
