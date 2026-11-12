
import logging
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
]

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]

OPTIONAL_COLUMNS = [
    "tick_volume",
    "spread",
    "real_volume",
]


# ============================================================
# 1. CHECK DATAFRAME
# ============================================================

def check_dataframe(df: Any) -> tuple[bool, str]:
    """
    Kiểm tra dữ liệu đầu vào có phải DataFrame hợp lệ không.
    """

    if not isinstance(df, pd.DataFrame):
        return False, "Dữ liệu đầu vào không phải pandas DataFrame."

    if df.empty:
        return False, "DataFrame không có dữ liệu."

    return True, "DataFrame hợp lệ."


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

def check_required_columns(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Kiểm tra các cột bắt buộc có tồn tại hay không.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        return (
            False,
            f"Thiếu các cột bắt buộc: {missing_columns}"
        )

    return True, "Đầy đủ các cột bắt buộc."


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

def check_missing_values(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Kiểm tra giá trị bị thiếu trong các cột bắt buộc.
    """

    missing_report = {}

    for column in REQUIRED_COLUMNS:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            missing_report[column] = missing_count

    if missing_report:
        return (
            False,
            f"Phát hiện giá trị thiếu: {missing_report}"
        )

    return True, "Không có giá trị thiếu trong các cột bắt buộc."


# ============================================================
# 4. CHECK DUPLICATE TIMESTAMPS
# ============================================================

def check_duplicate_timestamps(
    df: pd.DataFrame
) -> tuple[bool, str]:
    """
    Kiểm tra timestamp có bị trùng hay không.
    """

    timestamps = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce"
    )

    invalid_count = int(timestamps.isna().sum())

    if invalid_count > 0:
        return (
            False,
            f"Có {invalid_count} timestamp không hợp lệ."
        )

    duplicate_count = int(
        timestamps.duplicated(keep=False).sum()
    )

    if duplicate_count > 0:
        return (
            False,
            f"Phát hiện {duplicate_count} dòng có timestamp trùng."
        )

    return True, "Không có timestamp trùng."


# ============================================================
# 5. CHECK OHLC VALIDITY
# ============================================================

def check_ohlc_validity(
    df: pd.DataFrame
) -> tuple[bool, str]:
    """
    Kiểm tra tính hợp lệ của dữ liệu Open, High, Low, Close.

    Điều kiện:
        High >= Open
        High >= Close
        High >= Low

        Low <= Open
        Low <= Close
        Low <= High
    """

    data = df[PRICE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    values = data.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        return (
            False,
            "OHLC chứa giá trị không phải số hoặc vô hạn."
        )

    open_price = data["open"]
    high_price = data["high"]
    low_price = data["low"]
    close_price = data["close"]

    invalid_mask = (
        (high_price < open_price)
        | (high_price < close_price)
        | (high_price < low_price)
        | (low_price > open_price)
        | (low_price > close_price)
    )

    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        return (
            False,
            f"Phát hiện {invalid_count} dòng có cấu trúc OHLC không hợp lệ."
        )

    return True, "Dữ liệu OHLC hợp lệ."


# ============================================================
# 6. CHECK TIMESTAMP ORDER
# ============================================================

def check_timestamp_order(
    df: pd.DataFrame
) -> tuple[bool, str]:
    """
    Kiểm tra timestamp có được sắp xếp tăng dần không.
    """

    timestamps = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce"
    )

    if timestamps.isna().any():
        return False, "Timestamp chứa giá trị không hợp lệ."

    if not timestamps.is_monotonic_increasing:
        return (
            False,
            "Timestamp không được sắp xếp theo thứ tự tăng dần."
        )

    return True, "Timestamp được sắp xếp đúng thứ tự."


# ============================================================
# 7. MAIN VALIDATION FUNCTION
# ============================================================

def validate_data(df: Any) -> dict:
    """
    Chạy toàn bộ quy trình kiểm tra dữ liệu.

    Returns:
        dict: Báo cáo kết quả kiểm tra.
    """

    results = []

    # Kiểm tra DataFrame
    is_valid, message = check_dataframe(df)

    results.append({
        "check": "DataFrame",
        "passed": is_valid,
        "message": message
    })

    if not is_valid:
        return {
            "valid": False,
            "total_checks": len(results),
            "passed_checks": 0,
            "failed_checks": 1,
            "total_rows": 0,
            "results": results
        }

    # Kiểm tra các cột bắt buộc
    is_valid, message = check_required_columns(df)

    results.append({
        "check": "Required columns",
        "passed": is_valid,
        "message": message
    })

    if not is_valid:
        return {
            "valid": False,
            "total_checks": len(results),
            "passed_checks": sum(r["passed"] for r in results),
            "failed_checks": sum(not r["passed"] for r in results),
            "total_rows": len(df),
            "results": results
        }

    # Các bước kiểm tra còn lại
    validation_functions = [
        ("Missing values", check_missing_values),
        ("Duplicate timestamps", check_duplicate_timestamps),
        ("OHLC validity", check_ohlc_validity),
        ("Timestamp order", check_timestamp_order),
    ]

    for check_name, check_function in validation_functions:

        is_valid, message = check_function(df)

        results.append({
            "check": check_name,
            "passed": is_valid,
            "message": message
        })

    passed_checks = sum(
        result["passed"]
        for result in results
    )

    failed_checks = sum(
        not result["passed"]
        for result in results
    )

    return {
        "valid": failed_checks == 0,
        "total_checks": len(results),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "total_rows": len(df),
        "results": results
    }


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Dữ liệu mẫu để kiểm tra
    sample_data = pd.DataFrame({
        "time": [
            1794000000,
            1794000300,
            1794000600,
            1794000900,
            1794001200,
        ],
        "open": [1.1000, 1.1010, 1.1020, 1.1015, 1.1030],
        "high": [1.1020, 1.1030, 1.1040, 1.1040, 1.1050],
        "low": [1.0990, 1.1000, 1.1010, 1.1000, 1.1020],
        "close": [1.1010, 1.1020, 1.1015, 1.1030, 1.1040],
        "tick_volume": [100, 120, 150, 130, 160],
        "spread": [10, 10, 12, 10, 11],
        "real_volume": [0, 0, 0, 0, 0],
    })

    report = validate_data(sample_data)

    print("\n========== DATA VALIDATION REPORT ==========")

    for result in report["results"]:

        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"[{status}] {result['check']}: "
            f"{result['message']}"
        )

    print("\n========== SUMMARY ==========")

    print(f"Valid: {report['valid']}")
    print(f"Total rows: {report['total_rows']}")
    print(f"Passed: {report['passed_checks']}")
    print(f"Failed: {report['failed_checks']}")
