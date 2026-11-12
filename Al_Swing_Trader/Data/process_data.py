
"""
File: data/validate_data.py

Purpose:
    Kiểm tra chất lượng dữ liệu nến OHLCV lấy từ MetaTrader 5.

Responsibilities:
    1. Kiểm tra cấu trúc DataFrame.
    2. Kiểm tra các cột bắt buộc.
    3. Phát hiện giá trị thiếu.
    4. Phát hiện timestamp trùng lặp.
    5. Kiểm tra tính hợp lệ của OHLC.
    6. Kiểm tra thứ tự thời gian.
    7. Tổng hợp báo cáo chất lượng dữ liệu.

Note:
    File này chỉ kiểm tra, KHÔNG tự ý sửa dữ liệu đầu vào.
"""

import logging
import numpy as np
import pandas as pd

from typing import Any


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

REQUIRED_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close"
]

OPTIONAL_COLUMNS = [
    "tick_volume",
    "spread",
    "real_volume"
]

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close"
]


# =========================================================
# 1. CHECK DATAFRAME
# =========================================================

def check_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Kiểm tra dữ liệu đầu vào có phải Pandas DataFrame hợp lệ không.
    """

    if not isinstance(df, pd.DataFrame):
        return {
            "passed": False,
            "message": "Dữ liệu đầu vào không phải Pandas DataFrame."
        }

    if df.empty:
        return {
            "passed": False,
            "message": "DataFrame không có dữ liệu."
        }

    return {
        "passed": True,
        "message": "DataFrame hợp lệ.",
        "rows": len(df),
        "columns": list(df.columns)
    }


# =========================================================
# 2. CHECK REQUIRED COLUMNS
# =========================================================

def check_required_columns(df: pd.DataFrame) -> dict[str, Any]:
    """
    Kiểm tra các cột bắt buộc có tồn tại hay không.
    """

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        return {
            "passed": False,
            "message": "Thiếu các cột bắt buộc.",
            "missing_columns": missing_columns
        }

    return {
        "passed": True,
        "message": "Đầy đủ các cột bắt buộc."
    }


# =========================================================
# 3. CHECK MISSING VALUES
# =========================================================

def check_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Phát hiện giá trị thiếu trong các cột bắt buộc.
    """

    missing_counts = df[REQUIRED_COLUMNS].isna().sum()

    missing_columns = {
        col: int(count)
        for col, count in missing_counts.items()
        if count > 0
    }

    total_missing = int(missing_counts.sum())

    return {
        "passed": total_missing == 0,
        "message": (
            "Không có giá trị thiếu."
            if total_missing == 0
            else "Phát hiện giá trị thiếu."
        ),
        "total_missing": total_missing,
        "missing_columns": missing_columns
    }


# =========================================================
# 4. CHECK DUPLICATE TIMESTAMPS
# =========================================================

def check_duplicate_timestamps(df: pd.DataFrame) -> dict[str, Any]:
    """
    Phát hiện các nến có timestamp bị trùng lặp.
    """

    timestamps = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce"
    )

    duplicate_mask = timestamps.duplicated(keep=False)

    duplicate_count = int(duplicate_mask.sum())

    duplicate_examples = (
        timestamps[duplicate_mask]
        .drop_duplicates()
        .head(5)
        .astype(str)
        .tolist()
    )

    return {
        "passed": duplicate_count == 0,
        "message": (
            "Không có timestamp trùng lặp."
            if duplicate_count == 0
            else "Phát hiện timestamp trùng lặp."
        ),
        "duplicate_count": duplicate_count,
        "examples": duplicate_examples
    }


# =========================================================
# 5. CHECK OHLC VALIDITY
# =========================================================

def check_ohlc_validity(df: pd.DataFrame) -> dict[str, Any]:
    """
    Kiểm tra tính hợp lệ của dữ liệu Open, High, Low, Close.

    Điều kiện:
        High >= Open
        High >= Close
        Low <= Open
        Low <= Close
        High >= Low

    Đồng thời kiểm tra giá phải là số hữu hạn.
    """

    prices = df[PRICE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Kiểm tra dữ liệu không phải số hoặc NaN
    invalid_numeric = prices.isna().any(axis=1)

    # Kiểm tra giá trị vô cực
    invalid_finite = ~np.isfinite(prices).all(axis=1)

    # Kiểm tra OHLC logic
    invalid_ohlc = (
        (prices["high"] < prices["open"]) |
        (prices["high"] < prices["close"]) |
        (prices["low"] > prices["open"]) |
        (prices["low"] > prices["close"]) |
        (prices["high"] < prices["low"])
    )

    # Tổng hợp các dòng không hợp lệ
    invalid_mask = (
        invalid_numeric |
        invalid_finite |
        invalid_ohlc
    )

    invalid_count = int(invalid_mask.sum())

    invalid_examples = (
        df.loc[invalid_mask, ["time"] + PRICE_COLUMNS]
        .head(5)
        .to_dict(orient="records")
    )

    return {
        "passed": invalid_count == 0,
        "message": (
            "Dữ liệu OHLC hợp lệ."
            if invalid_count == 0
            else "Phát hiện dữ liệu OHLC không hợp lệ."
        ),
        "invalid_count": invalid_count,
        "invalid_numeric_count": int(invalid_numeric.sum()),
        "invalid_finite_count": int(invalid_finite.sum()),
        "invalid_ohlc_count": int(invalid_ohlc.sum()),
        "examples": invalid_examples
    }


# =========================================================
# 6. CHECK TIMESTAMP ORDER
# =========================================================

def check_timestamp_order(df: pd.DataFrame) -> dict[str, Any]:
    """
    Kiểm tra timestamp có hợp lệ và tăng dần hay không.
    """

    timestamps = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce"
    )

    # Kiểm tra timestamp không chuyển đổi được
    invalid_timestamp = timestamps.isna()

    invalid_count = int(invalid_timestamp.sum())

    if invalid_count > 0:
        return {
            "passed": False,
            "message": "Có timestamp không hợp lệ.",
            "invalid_count": invalid_count
        }

    # Kiểm tra thứ tự thời gian
    is_monotonic = timestamps.is_monotonic_increasing

    return {
        "passed": is_monotonic,
        "message": (
            "Timestamp tăng dần."
            if is_monotonic
            else "Timestamp không tăng dần."
        ),
        "is_monotonic": is_monotonic,
        "first_timestamp": str(timestamps.iloc[0]),
        "last_timestamp": str(timestamps.iloc[-1])
    }


# =========================================================
# 7. MAIN VALIDATION FUNCTION
# =========================================================

def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Thực hiện toàn bộ quá trình kiểm tra dữ liệu.

    Returns:
        dict: Báo cáo tổng hợp kết quả kiểm tra.
    """

    logger.info("Bắt đầu kiểm tra chất lượng dữ liệu.")

    # Kiểm tra DataFrame
    dataframe_result = check_dataframe(df)

    if not dataframe_result["passed"]:
        logger.error(dataframe_result["message"])

        return {
            "valid": False,
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 1,
            "results": {
                "dataframe": dataframe_result
            }
        }

    # Kiểm tra các cột bắt buộc
    columns_result = check_required_columns(df)

    if not columns_result["passed"]:
        logger.error(columns_result["message"])

        return {
            "valid": False,
            "total_checks": 1,
            "passed_checks": 0,
            "failed_checks": 1,
            "results": {
                "dataframe": dataframe_result,
                "required_columns": columns_result
            }
        }

    # Thực hiện các kiểm tra chi tiết
    results = {
        "dataframe": dataframe_result,
        "required_columns": columns_result,
        "missing_values": check_missing_values(df),
        "duplicate_timestamps": check_duplicate_timestamps(df),
        "ohlc_validity": check_ohlc_validity(df),
        "timestamp_order": check_timestamp_order(df)
    }

    # Tổng hợp kết quả
    total_checks = len(results)

    passed_checks = sum(
        1 for result in results.values()
        if result["passed"]
    )

    failed_checks = total_checks - passed_checks

    valid = failed_checks == 0

    report = {
        "valid": valid,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "total_rows": len(df),
        "results": results
    }

    # Logging kết quả
    logger.info(
        f"Hoàn thành kiểm tra: "
        f"{passed_checks}/{total_checks} kiểm tra đạt."
    )

    if valid:
        logger.info("Dữ liệu đạt tất cả điều kiện kiểm tra.")
    else:
        logger.warning(
            f"Dữ liệu chưa hợp lệ. "
            f"Có {failed_checks} kiểm tra thất bại."
        )

    return report


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Dữ liệu giả lập chỉ để kiểm thử module
    test_data = pd.DataFrame({
        "time": pd.date_range(
            start="2026-01-01 00:00:00",
            periods=5,
            freq="5min",
            tz="UTC"
        ),
        "open": [1.1000, 1.1010, 1.1020, 1.1015, 1.1030],
        "high": [1.1020, 1.1030, 1.1040, 1.1030, 1.1050],
        "low": [1.0990, 1.1000, 1.1010, 1.1005, 1.1020],
        "close": [1.1010, 1.1020, 1.1015, 1.1025, 1.1040],
        "tick_volume": [100, 120, 150, 130, 180]
    })

    report = validate_data(test_data)

    print("\n========== VALIDATION REPORT ==========")
    print(f"Valid: {report['valid']}")
    print(f"Total checks: {report['total_checks']}")
    print(f"Passed: {report['passed_checks']}")
    print(f"Failed: {report['failed_checks']}")

    for name, result in report["results"].items():
        status = "PASS" if result["passed"] else "FAIL"

        print(f"\n[{status}] {name}")
        print(result["message"])
