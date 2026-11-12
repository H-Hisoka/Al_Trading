
"""
swing_registry.py

Quản lý vòng đời Swing Candidate và Confirmed Swing.

Không thực hiện phát hiện swing.
Đầu vào là candidate được tạo bởi adaptive_swing.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


VALID_TYPES = {"TOP", "BOTTOM"}
VALID_SIZES = {"MINOR", "MAJOR"}

REQUIRED_FIELDS = {
    "swing_type",
    "swing_size",
    "pivot_time",
    "pivot_price",
    "confirmation_time",
    "confirmation_price",
    "atr_at_pivot",
    "reversal_atr",
    "prominence_score",
}


@dataclass
class RegistryConfig:
    """
    Cấu hình quản lý swing.

    allow_same_direction_nested:
        Cho phép các swing cùng hướng tồn tại ở
        cấp độ khác nhau, ví dụ MINOR nằm trong MAJOR.

    strict_alternation:
        Nếu True, swing cùng cấp phải luân phiên TOP/BOTTOM.
    """

    allow_same_direction_nested: bool = True
    strict_alternation: bool = True

    duplicate_time_tolerance_seconds: float = 0.0
    duplicate_price_tolerance: float = 0.0

    require_positive_atr: bool = True
    min_prominence_score: float = 0.0


@dataclass
class SwingRegistry:
    """
    Registry lưu trữ swing.

    confirmed:
        Danh sách swing đã xác nhận.

    candidates:
        Các swing đang chờ xác nhận.

    Mọi swing được quản lý dưới dạng dictionary
    để tương thích với dữ liệu DataFrame.
    """

    config: RegistryConfig = field(
        default_factory=RegistryConfig
    )

    confirmed: List[Dict[str, Any]] = field(
        default_factory=list
    )

    candidates: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )

    _next_id: int = 1

    def _generate_id(self) -> str:
        swing_id = f"SWING_{self._next_id:08d}"
        self._next_id += 1
        return swing_id

    @staticmethod
    def _to_timestamp(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            raise ValueError("Timestamp không hợp lệ.")

        return timestamp

    @staticmethod
    def _normalize_swing(swing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chuẩn hóa dữ liệu swing nhưng không tự suy diễn
        những trường bắt buộc còn thiếu.
        """

        normalized = dict(swing)

        if "swing_type" in normalized:
            normalized["swing_type"] = str(
                normalized["swing_type"]
            ).upper()

        if "swing_size" in normalized:
            normalized["swing_size"] = str(
                normalized["swing_size"]
            ).upper()

        return normalized

    def validate_swing(
        self,
        swing: Dict[str, Any],
        require_confirmation: bool = True
    ) -> bool:
        """
        Kiểm tra tính hợp lệ của swing.

        Candidate có thể chưa có confirmation_time
        và confirmation_price.
        """

        if not isinstance(swing, dict):
            raise TypeError("Swing phải là dictionary.")

        missing = REQUIRED_FIELDS - set(swing)

        if not require_confirmation:
            missing -= {
                "confirmation_time",
                "confirmation_price",
            }

        if missing:
            raise ValueError(
                f"Swing thiếu trường: {sorted(missing)}"
            )

        swing_type = str(
            swing["swing_type"]
        ).upper()

        swing_size = str(
            swing["swing_size"]
        ).upper()

        if swing_type not in VALID_TYPES:
            raise ValueError(
                f"swing_type không hợp lệ: {swing_type}"
            )

        if swing_size not in VALID_SIZES:
            raise ValueError(
                f"swing_size không hợp lệ: {swing_size}"
            )

        pivot_time = self._to_timestamp(
            swing["pivot_time"]
        )

        pivot_price = float(swing["pivot_price"])
        atr = float(swing["atr_at_pivot"])
        reversal_atr = float(swing["reversal_atr"])
        prominence = float(swing["prominence_score"])

        if not all(
            pd.notna(value)
            for value in [
                pivot_price,
                atr,
                reversal_atr,
                prominence,
            ]
        ):
            raise ValueError("Swing chứa giá trị NaN.")

        if atr < 0 or reversal_atr < 0:
            raise ValueError("ATR và reversal không được âm.")

        if self.config.require_positive_atr and atr <= 0:
            raise ValueError("ATR phải lớn hơn 0.")

        if not 0 <= prominence <= 100:
            raise ValueError(
                "prominence_score phải nằm trong [0, 100]."
            )

        if prominence < self.config.min_prominence_score:
            raise ValueError(
                "Prominence không đạt ngưỡng cấu hình."
            )

        if require_confirmation:
            confirmation_time = self._to_timestamp(
                swing["confirmation_time"]
            )

            confirmation_price = float(
                swing["confirmation_price"]
            )

            if pd.isna(confirmation_price):
                raise ValueError(
                    "confirmation_price không hợp lệ."
                )

            if confirmation_time < pivot_time:
                raise ValueError(
                    "confirmation_time không thể trước pivot_time."
                )

        return True

    def check_duplicate(
        self,
        swing: Dict[str, Any],
        include_candidates: bool = True
    ) -> bool:
        """
        Kiểm tra swing trùng với swing đã tồn tại.

        Trả về True nếu phát hiện trùng lặp.
        """

        pivot_time = self._to_timestamp(
            swing["pivot_time"]
        )

        pivot_price = float(swing["pivot_price"])

        existing_swings = list(self.confirmed)

        if include_candidates:
            existing_swings.extend(
                self.candidates.values()
            )

        for existing in existing_swings:
            if (
                existing["swing_type"] != swing["swing_type"]
                or existing["swing_size"] != swing["swing_size"]
            ):
                continue

            existing_time = self._to_timestamp(
                existing["pivot_time"]
            )

            existing_price = float(
                existing["pivot_price"]
            )

            time_difference = abs(
                (pivot_time - existing_time).total_seconds()
            )

            price_difference = abs(
                pivot_price - existing_price
            )

            if (
                time_difference
                <= self.config.duplicate_time_tolerance_seconds
                and price_difference
                <= self.config.duplicate_price_tolerance
            ):
                return True

        return False

    def check_alternation(
        self,
        swing: Dict[str, Any]
    ) -> bool:
        """
        Kiểm tra TOP/BOTTOM luân phiên.

        So sánh theo cùng cấp độ MINOR hoặc MAJOR.
        Điều này cho phép swing nhỏ nằm bên trong
        swing lớn mà không bị loại bỏ chỉ vì khác cấp.
        """

        if not self.config.strict_alternation:
            return True

        same_level = [
            item for item in self.confirmed
            if item["swing_size"] == swing["swing_size"]
        ]

        if not same_level:
            return True

        latest = max(
            same_level,
            key=lambda item: self._to_timestamp(
                item["confirmation_time"]
            )
        )

        latest_type = latest["swing_type"]

        return latest_type != swing["swing_type"]

    def register_candidate(
        self,
        swing: Dict[str, Any]
    ) -> str:
        """
        Đăng ký một swing candidate.

        Candidate chưa được đưa vào confirmed.
        """

        normalized = self._normalize_swing(swing)

        self.validate_swing(
            normalized,
            require_confirmation=False
        )

        if self.check_duplicate(normalized):
            raise ValueError(
                "Candidate đã tồn tại trong registry."
            )

        candidate_id = normalized.get("swing_id")

        if not candidate_id:
            candidate_id = self._generate_id()

        if candidate_id in self.candidates:
            raise ValueError(
                f"Candidate ID đã tồn tại: {candidate_id}"
            )

        normalized["swing_id"] = candidate_id
        normalized["status"] = "CANDIDATE"

        self.candidates[candidate_id] = normalized

        return candidate_id

    def update_swing(
        self,
        swing_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cập nhật candidate chưa xác nhận.

        Không cho phép sửa candidate đã được xác nhận.
        """

        if swing_id not in self.candidates:
            raise KeyError(
                f"Không tìm thấy candidate: {swing_id}"
            )

        updated = dict(self.candidates[swing_id])
        updated.update(updates)

        updated = self._normalize_swing(updated)

        if updated.get("swing_id") != swing_id:
            raise ValueError(
                "Không được thay đổi swing_id."
            )

        self.validate_swing(
            updated,
            require_confirmation=False
        )

        for other_id, other in self.candidates.items():
            if other_id == swing_id:
                continue

            if (
                other["swing_type"] == updated["swing_type"]
                and other["swing_size"] == updated["swing_size"]
                and self._to_timestamp(other["pivot_time"])
                == self._to_timestamp(updated["pivot_time"])
                and abs(
                    float(other["pivot_price"])
                    - float(updated["pivot_price"])
                ) <= self.config.duplicate_price_tolerance
            ):
                raise ValueError(
                    "Cập nhật tạo ra candidate trùng lặp."
                )

        self.candidates[swing_id] = updated

        return dict(updated)

    def confirm_swing(
        self,
        swing_id: str,
        confirmation_time: Optional[Any] = None,
        confirmation_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Xác nhận một candidate.

        Chỉ chuyển sang confirmed khi:
        - Dữ liệu hợp lệ.
        - Không trùng lặp.
        - Thỏa mãn quy tắc luân phiên.
        """

        if swing_id not in self.candidates:
            raise KeyError(
                f"Không tìm thấy candidate: {swing_id}"
            )

        swing = dict(self.candidates[swing_id])

        if confirmation_time is not None:
            swing["confirmation_time"] = confirmation_time

        if confirmation_price is not None:
            swing["confirmation_price"] = confirmation_price

        self.validate_swing(
            swing,
            require_confirmation=True
        )

        if self.check_duplicate(
            swing,
            include_candidates=False
        ):
            raise ValueError(
                "Swing đã tồn tại trong danh sách confirmed."
            )

        if not self.check_alternation(swing):
            raise ValueError(
                "Swing không thỏa mãn TOP/BOTTOM alternation."
            )

        swing["status"] = "CONFIRMED"

        self.confirmed.append(swing)

        self.confirmed.sort(
            key=lambda item: self._to_timestamp(
                item["confirmation_time"]
            )
        )

        del self.candidates[swing_id]

        return dict(swing)

    def get_last_swing(
        self,
        swing_type: Optional[str] = None,
        swing_size: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Lấy swing được xác nhận gần nhất.
        Có thể lọc theo TOP/BOTTOM và MINOR/MAJOR.
        """

        results = self.get_swings(
            swing_type=swing_type,
            swing_size=swing_size
        )

        if not results:
            return None

        return max(
            results,
            key=lambda item: self._to_timestamp(
                item["confirmation_time"]
            )
        )

    def get_swings(
        self,
        swing_type: Optional[str] = None,
        swing_size: Optional[str] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Truy xuất các swing đã xác nhận.
        """

        results = list(self.confirmed)

        if swing_type is not None:
            swing_type = swing_type.upper()

            if swing_type not in VALID_TYPES:
                raise ValueError("swing_type không hợp lệ.")

            results = [
                item for item in results
                if item["swing_type"] == swing_type
            ]

        if swing_size is not None:
            swing_size = swing_size.upper()

            if swing_size not in VALID_SIZES:
                raise ValueError("swing_size không hợp lệ.")

            results = [
                item for item in results
                if item["swing_size"] == swing_size
            ]

        if start_time is not None:
            start = self._to_timestamp(start_time)

            results = [
                item for item in results
                if self._to_timestamp(item["pivot_time"]) >= start
            ]

        if end_time is not None:
            end = self._to_timestamp(end_time)

            results = [
                item for item in results
                if self._to_timestamp(item["pivot_time"]) <= end
            ]

        return sorted(
            results,
            key=lambda item: self._to_timestamp(
                item["pivot_time"]
            )
        )

    def get_swings_between(
        self,
        start_time: Any,
        end_time: Any,
        swing_type: Optional[str] = None,
        swing_size: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy swing có pivot_time nằm trong khoảng thời gian.
        """

        start = self._to_timestamp(start_time)
        end = self._to_timestamp(end_time)

        if start > end:
            raise ValueError(
                "start_time phải nhỏ hơn hoặc bằng end_time."
            )

        return self.get_swings(
            swing_type=swing_type,
            swing_size=swing_size,
            start_time=start,
            end_time=end
        )

    def clear(self) -> None:
        """
        Xóa trạng thái registry trong bộ nhớ.

        Không xóa file dữ liệu trên ổ cứng.
        """

        self.confirmed.clear()
        self.candidates.clear()
        self._next_id = 1

    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển các swing đã xác nhận thành DataFrame.
        """

        return pd.DataFrame(self.confirmed)
