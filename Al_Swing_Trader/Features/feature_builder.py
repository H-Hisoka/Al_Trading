
# File: features/feature_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from features.indicators import calculate_indicators
from features.candle_features import calculate_candle_features

from features.market_structure import (
    MarketStructureConfig,
    detect_market_structure,
)

from features.supply_demand import (
    SupplyDemandConfig,
    create_supply_demand_zones,
    add_supply_demand_features,
)

from features.liquidity_sweep import (
    LiquiditySweepConfig,
    detect_liquidity_sweeps,
)


@dataclass
class FeatureBuilderConfig:
    """
    Cấu hình tổng hợp features.
    """

    # Tên cột thời gian nếu chưa đặt làm DataFrame index.
    timestamp_column: Optional[str] = None

    # Bật/tắt từng nhóm feature.
    use_indicators: bool = True
    use_candle_features: bool = True
    use_market_structure: bool = True
    use_supply_demand: bool = True
    use_liquidity_sweep: bool = True

    # Chỉ giữ các cột số trong đầu vào ML.
    numeric_only: bool = True

    # Loại bỏ các feature không đủ dữ liệu.
    drop_all_nan_columns: bool = True

    # Có loại bỏ hàng đầu tiên chưa đủ dữ liệu rolling hay không.
    drop_rows_with_nan: bool = False

    # Giữ lại OHLCV trong kết quả.
    include_raw_ohlcv: bool = True

    # Cấu hình riêng từng module.
    market_structure: MarketStructureConfig = (
        MarketStructureConfig()
    )

    supply_demand: SupplyDemandConfig = (
        SupplyDemandConfig()
    )

    liquidity_sweep: LiquiditySweepConfig = (
        LiquiditySweepConfig()
    )


def _prepare_input(
    candles: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu đầu vào.

    Index sau bước này phải là thời gian tăng dần.
    """
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles phải là DataFrame")

    df = candles.copy()

    if config.timestamp_column:
        if config.timestamp_column not in df.columns:
            raise ValueError(
                f"Không tìm thấy {config.timestamp_column}"
            )

        df[config.timestamp_column] = pd.to_datetime(
            df[config.timestamp_column]
        )

        df = df.set_index(config.timestamp_column)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame cần có DatetimeIndex hoặc timestamp_column"
        )

    if df.index.has_duplicates:
        raise ValueError(
            "Dữ liệu có timestamp trùng lặp"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Dữ liệu phải được sắp xếp theo thời gian tăng dần"
        )

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Thiếu cột OHLC: {sorted(missing)}"
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[list(required)].isna().any().any():
        raise ValueError("OHLC có dữ liệu không hợp lệ")

    if (df["high"] < df["low"]).any():
        raise ValueError("Có nến high nhỏ hơn low")

    return df


def _new_columns(
    source: pd.DataFrame,
    original_columns: set,
) -> pd.DataFrame:
    """
    Lấy riêng các cột mới được module tạo ra.
    """
    columns = [
        column
        for column in source.columns
        if column not in original_columns
    ]

    return source[columns].copy()


def _join_features(
    base: pd.DataFrame,
    new_features: pd.DataFrame,
    group_name: str,
) -> pd.DataFrame:
    """
    Ghép features và kiểm tra trùng tên cột.
    """
    if not base.index.equals(new_features.index):
        raise ValueError(
            f"Index không đồng nhất khi ghép nhóm {group_name}"
        )

    duplicated = (
        set(base.columns)
        & set(new_features.columns)
    )

    if duplicated:
        raise ValueError(
            f"Trùng tên cột ở nhóm {group_name}: "
            f"{sorted(duplicated)}"
        )

    return base.join(new_features)


def _get_market_structure_features(
    candles: pd.DataFrame,
    swings: pd.DataFrame,
    config: MarketStructureConfig,
) -> pd.DataFrame:
    """
    Chỉ lấy các feature cần thiết từ market structure.

    Không đưa timestamp, event ID hoặc giá pivot
    vào mô hình mặc định.
    """
    structure = detect_market_structure(
        candles,
        swings,
        config,
    )

    columns = [
        "bos",
        "choch",
        "last_swing_high",
        "last_swing_low",
    ]

    available = [
        column
        for column in columns
        if column in structure.columns
    ]

    features = structure[available].copy()

    # Mã hóa xu hướng thành số.
    trend_mapping = {
        "UNKNOWN": 0,
        "SIDEWAYS": 1,
        "BULLISH": 2,
        "BEARISH": -2,
    }

    features["market_trend_code"] = (
        structure["market_trend"]
        .map(trend_mapping)
        .fillna(0)
        .astype("int8")
    )

    # Mã hóa nhãn cấu trúc.
    label_mapping = {
        "HH": 2,
        "HL": 1,
        "LH": -1,
        "LL": -2,
    }

    features["structure_label_code"] = (
        structure["latest_structure_label"]
        .map(label_mapping)
        .fillna(0)
        .astype("int8")
    )

    return features


def _get_supply_demand_features(
    candles: pd.DataFrame,
    config: SupplyDemandConfig,
) -> pd.DataFrame:
    """
    Tạo zones và trích xuất features theo thời điểm.

    Không đưa status được tính từ toàn bộ tương lai
    vào features.
    """
    zones = create_supply_demand_zones(
        candles,
        config,
    )

    features = add_supply_demand_features(
        candles,
        zones,
        config,
    )

    selected_columns = [
        "nearest_demand_distance_atr",
        "nearest_supply_distance_atr",
        "inside_demand_zone",
        "inside_supply_zone",
        "demand_zone_count",
        "supply_zone_count",
    ]

    return features[selected_columns].copy()


def _get_liquidity_features(
    candles: pd.DataFrame,
    swings: pd.DataFrame,
    config: LiquiditySweepConfig,
) -> pd.DataFrame:
    """
    Trích xuất feature Liquidity Sweep.
    """
    sweeps = detect_liquidity_sweeps(
        candles,
        swings,
        config,
    )

    selected_columns = [
        "buy_side_sweep",
        "sell_side_sweep",
        "liquidity_sweep",
        "sweep_depth_atr",
        "sweep_wick_ratio",
        "active_buy_side_levels",
        "active_sell_side_levels",
    ]

    return sweeps[selected_columns].copy()


def build_features(
    candles: pd.DataFrame,
    swings: Optional[pd.DataFrame] = None,
    config: Optional[FeatureBuilderConfig] = None,
) -> pd.DataFrame:
    """
    Xây dựng feature matrix cho mô hình ML.

    Input:
        candles:
            DataFrame OHLCV có DatetimeIndex.

        swings:
            DataFrame swing đã xác nhận, cần có:
            swing_type, pivot_time, pivot_price,
            confirmation_time.

    Output:
        DataFrame features theo thời gian.

    Hàm này không tạo nhãn và không thực hiện train/test split.
    """
    if config is None:
        config = FeatureBuilderConfig()

    df = _prepare_input(candles, config)

    if (
        config.use_market_structure
        or config.use_liquidity_sweep
    ) and swings is None:
        raise ValueError(
            "Cần truyền swings khi bật market_structure "
            "hoặc liquidity_sweep"
        )

    # Bắt đầu với OHLCV nếu người dùng muốn giữ lại.
    if config.include_raw_ohlcv:
        base_columns = [
            column
            for column in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tick_volume",
                "real_volume",
            ]
            if column in df.columns
        ]

        result = df[base_columns].copy()

    else:
        result = pd.DataFrame(index=df.index)

    # 1. Technical indicators.
    if config.use_indicators:
        indicator_df = calculate_indicators(df)

        indicator_features = _new_columns(
            indicator_df,
            set(df.columns),
        )

        result = _join_features(
            result,
            indicator_features,
            "indicators",
        )

    # 2. Candle features.
    if config.use_candle_features:
        candle_df = calculate_candle_features(df)

        candle_features = _new_columns(
            candle_df,
            set(df.columns),
        )

        result = _join_features(
            result,
            candle_features,
            "candle_features",
        )

    # Các module phía sau cần ATR.
    # Nếu tắt indicators thì tính ATR riêng.
    if "atr_14" not in result.columns:
        from features.indicators import calculate_atr

        atr = calculate_atr(df, 14)

        if config.use_indicators:
            result["atr_14"] = atr
        else:
            result["atr_14"] = atr

    # Tạo bản dữ liệu có ATR để sử dụng ở các module sau.
    analysis_candles = df.copy()

    if "atr_14" not in analysis_candles.columns:
        analysis_candles["atr_14"] = result["atr_14"]

    # 3. Market structure.
    if config.use_market_structure:
        structure_features = _get_market_structure_features(
            analysis_candles,
            swings,
            config.market_structure,
        )

        result = _join_features(
            result,
            structure_features,
            "market_structure",
        )

    # 4. Supply/Demand.
    if config.use_supply_demand:
        supply_demand_features = _get_supply_demand_features(
            analysis_candles,
            config.supply_demand,
        )

        result = _join_features(
            result,
            supply_demand_features,
            "supply_demand",
        )

    # 5. Liquidity sweep.
    if config.use_liquidity_sweep:
        liquidity_features = _get_liquidity_features(
            analysis_candles,
            swings,
            config.liquidity_sweep,
        )

        result = _join_features(
            result,
            liquidity_features,
            "liquidity_sweep",
        )

    # Loại bỏ cột có toàn bộ giá trị NaN.
    if config.drop_all_nan_columns:
        result = result.dropna(
            axis=1,
            how="all",
        )

    # Chỉ giữ feature số.
    if config.numeric_only:
        result = result.select_dtypes(
            include=[np.number, "bool"]
        )

    # Kiểm tra infinity.
    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if config.drop_rows_with_nan:
        result = result.dropna(axis=0)

    return result


def validate_feature_matrix(
    features: pd.DataFrame,
) -> dict:
    """
    Kiểm tra sơ bộ feature matrix trước khi train.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features phải là DataFrame")

    if features.empty:
        raise ValueError("Feature matrix đang rỗng")

    duplicated_columns = features.columns[
        features.columns.duplicated()
    ].tolist()

    nan_counts = features.isna().sum()

    infinity_count = int(
        np.isinf(
            features.select_dtypes(include=[np.number])
        ).sum().sum()
    )

    return {
        "rows": len(features),
        "columns": len(features.columns),
        "duplicated_columns": duplicated_columns,
        "columns_with_nan": {
            column: int(count)
            for column, count in nan_counts.items()
            if count > 0
        },
        "infinity_count": infinity_count,
        "numeric_columns": len(
            features.select_dtypes(include=[np.number]).columns
        ),
    }
