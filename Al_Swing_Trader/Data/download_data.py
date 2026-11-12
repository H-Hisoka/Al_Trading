
import MetaTrader5 as mt5
import logging


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# 1. INITIALIZE MT5
# =========================================================

def initialize_mt5() -> bool:
    """
    Khởi tạo kết nối với MetaTrader 5.

    Returns:
        bool: True nếu kết nối thành công, False nếu thất bại.
    """

    # Kiểm tra MT5 đã được khởi tạo trước đó chưa
    if mt5.terminal_info() is not None:
        logger.info("MT5 đã được kết nối.")
        return True

    # Khởi tạo kết nối MT5
    if not mt5.initialize():
        error_code, error_message = mt5.last_error()

        logger.error(
            f"Không thể kết nối MT5. "
            f"Error code: {error_code}, "
            f"Message: {error_message}"
        )

        return False

    # Kiểm tra thông tin terminal
    terminal = mt5.terminal_info()

    if terminal is None:
        logger.error("Không thể lấy thông tin MT5 Terminal.")
        mt5.shutdown()
        return False

    # Kiểm tra tài khoản đang đăng nhập
    account = mt5.account_info()

    if account is None:
        logger.error("MT5 chưa đăng nhập tài khoản.")
        mt5.shutdown()
        return False

    logger.info("Kết nối MT5 thành công.")
    logger.info(f"Account Login: {account.login}")
    logger.info(f"Server: {account.server}")
    logger.info(f"Company: {account.company}")

    return True


# =========================================================
# 2. SHUTDOWN MT5
# =========================================================

def shutdown_mt5() -> None:
    """
    Đóng kết nối MetaTrader 5 và giải phóng tài nguyên.
    """

    try:
        mt5.shutdown()
        logger.info("Đã đóng kết nối MT5.")

    except Exception as error:
        logger.error(f"Lỗi khi đóng kết nối MT5: {error}")


# =========================================================
# TEST CONNECTION
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        if initialize_mt5():
            print("\nMT5 CONNECTION SUCCESSFUL!")

    finally:
        shutdown_mt5()
