
"""
File: data/storage.py

Purpose:
    Lưu trữ và truy xuất dữ liệu nến OHLCV.

Responsibilities:
    1. Tạo đường dẫn lưu trữ.
    2. Lưu DataFrame thành Parquet hoặc CSV.
    3. Đọc dữ liệu từ Parquet hoặc CSV.
    4. Kiểm tra file tồn tại.
    5. Xóa file theo yêu cầu.
    6. Thống kê thông tin dữ liệu.

Note:
    Không tự động ghi đè dữ liệu cũ.
"""

import logging
import re
import os
import pandas as pd

from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_BASE_DIR = Path("data")

SUPPORTED_FORMATS = [
    "parquet",
    "csv"
]


# =========================================================
# 1. GET FILE PATH
# =========================================================

def get_file_path(
    symbol: str,
    timeframe: str,
    data_type: str = "processed",
    file_format: str = "parquet",
    base_dir: Optional[Path] = None
) -> Path:
    """
    Tạo đường dẫn lưu trữ dữ liệu.

    Ví dụ:
        data/processed/EURUSD_M5.parquet
    """

    if not symbol or not timeframe:
        raise ValueError("Symbol và timeframe không được để trống.")

    if data_type not in ["raw", "processed"]:
        raise ValueError(
            "data_type chỉ được là 'raw' hoặc 'processed'."
        )

    if file_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Định dạng không được hỗ trợ: {file_format}"
        )

    # Ngăn ký tự không an toàn trong tên file
    safe_symbol = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        symbol
    )

    safe_timeframe = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        timeframe
    )

    root = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR

    directory = root / data_type

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"{safe_symbol}_{safe_timeframe}.{file_format}"
    )

    return directory / filename


# =========================================================
# 2. SAVE DATA
# =========================================================

def save_data(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    data_type: str = "processed",
    file_format: str = "parquet",
    overwrite: bool = False,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Lưu DataFrame thành Parquet hoặc CSV.

    overwrite:
        False: Không ghi đè file cũ.
        True: Cho phép thay thế file cũ.

    Returns:
        Path: Đường dẫn file đã lưu.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dữ liệu phải là Pandas DataFrame.")

    if df.empty:
        raise ValueError("Không thể lưu DataFrame rỗng.")

    file_path = get_file_path(
        symbol=symbol,
        timeframe=timeframe,
        data_type=data_type,
        file_format=file_format,
        base_dir=base_dir
    )

    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"File đã tồn tại: {file_path}. "
            "Sử dụng overwrite=True nếu muốn ghi đè."
        )

    # Tạo file tạm để tránh ghi trực tiếp vào file đích
    temp_path = file_path.with_name(
        f"{file_path.stem}.tmp{file_path.suffix}"
    )

    try:
        if file_format == "parquet":
            df.to_parquet(
                temp_path,
                index=False,
                engine="pyarrow"
            )

        elif file_format == "csv":
            df.to_csv(
                temp_path,
                index=False,
                encoding="utf-8"
            )

        # Kiểm tra file tạm đã được tạo
        if not temp_path.exists():
            raise IOError("Không tạo được file dữ liệu.")

        # Chuyển file tạm sang file chính
        if file_path.exists() and not overwrite:
            raise FileExistsError(
                f"File đã tồn tại: {file_path}"
            )

        os.replace(temp_path, file_path)

        logger.info(f"Đã lưu dữ liệu: {file_path}")

        return file_path

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise


# =========================================================
# 3. LOAD DATA
# =========================================================

def load_data(
    symbol: str,
    timeframe: str,
    data_type: str = "processed",
    file_format: str = "parquet",
    base_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Đọc dữ liệu từ Parquet hoặc CSV.

    Returns:
        DataFrame chứa dữ liệu đã lưu.
    """

    file_path = get_file_path(
        symbol=symbol,
        timeframe=timeframe,
        data_type=data_type,
        file_format=file_format,
        base_dir=base_dir
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        )

    if file_format == "parquet":
        df = pd.read_parquet(
            file_path,
            engine="pyarrow"
        )

    elif file_format == "csv":
        df = pd.read_csv(
            file_path,
            parse_dates=["time"]
        )

        # Đảm bảo timestamp được đọc dưới dạng UTC
        df["time"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="raise"
        )

    else:
        raise ValueError(
            f"Định dạng không được hỗ trợ: {file_format}"
        )

    logger.info(
        f"Đã đọc dữ liệu: {file_path} ({len(df)} dòng)"
    )

    return df


# =========================================================
# 4. CHECK DATA EXISTS
# =========================================================

def data_exists(
    symbol: str,
    timeframe: str,
    data_type: str = "processed",
    file_format: str = "parquet",
    base_dir: Optional[Path] = None
) -> bool:
    """
    Kiểm tra dữ liệu đã được lưu hay chưa.
    """

    file_path = get_file_path(
        symbol=symbol,
        timeframe=timeframe,
        data_type=data_type,
        file_format=file_format,
        base_dir=base_dir
    )

    return file_path.is_file()


# =========================================================
# 5. DELETE DATA
# =========================================================

def delete_data(
    symbol: str,
    timeframe: str,
    data_type: str = "processed",
    file_format: str = "parquet",
    base_dir: Optional[Path] = None
) -> bool:
    """
    Xóa file dữ liệu theo yêu cầu.

    Returns:
        True nếu xóa thành công.
        False nếu file không tồn tại.
    """

    file_path = get_file_path(
        symbol=symbol,
        timeframe=timeframe,
        data_type=data_type,
        file_format=file_format,
        base_dir=base_dir
    )

    if not file_path.is_file():
        logger.warning(f"Không tìm thấy file: {file_path}")
        return False

    file_path.unlink()

    logger.info(f"Đã xóa file: {file_path}")

    return True


# =========================================================
# 6. GET DATA INFO
# =========================================================

def get_data_info(
    symbol: str,
    timeframe: str,
    data_type: str = "processed",
    file_format: str = "parquet",
    base_dir: Optional[Path] = None
) -> dict:
    """
    Thống kê thông tin file dữ liệu.
    """

    file_path = get_file_path(
        symbol=symbol,
        timeframe=timeframe,
        data_type=data_type,
        file_format=file_format,
        base_dir=base_dir
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        )

    df = load_data(
        symbol=symbol,
        timeframe=timeframe,
        data_type=data_type,
        file_format=file_format,
        base_dir=base_dir
    )

    info = {
        "file_path": str(file_path),
        "file_size_mb": round(
            file_path.stat().st_size / (1024 ** 2),
            4
        ),
        "total_rows": len(df),
        "columns": list(df.columns),
        "first_timestamp": (
            str(df["time"].min())
            if "time" in df.columns else None
        ),
        "last_timestamp": (
            str(df["time"].max())
            if "time" in df.columns else None
        )
    }

    return info


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_data = pd.DataFrame({
        "time": pd.date_range(
            start="2026-01-01",
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

    # Lưu dữ liệu thử nghiệm
    saved_path = save_data(
        df=test_data,
        symbol="TEST",
        timeframe="M5",
        data_type="processed",
        file_format="parquet",
        overwrite=True,
        base_dir=Path("data/test_storage")
    )

    # Đọc dữ liệu
    loaded_data = load_data(
        symbol="TEST",
        timeframe="M5",
        data_type="processed",
        file_format="parquet",
        base_dir=Path("data/test_storage")
    )

    # Thống kê dữ liệu
    info = get_data_info(
        symbol="TEST",
        timeframe="M5",
        data_type="processed",
        file_format="parquet",
        base_dir=Path("data/test_storage")
    )

    print("\n========== STORAGE REPORT ==========")
    print(info)
    print("\nDữ liệu đã đọc:")
    print(loaded_data.head())
