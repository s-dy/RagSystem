import uvicorn

from config import LoggingConfig
from server import app
from src.observability.logger import setup_logger

if __name__ == "__main__":
    # 初始化日志系统
    log_config = LoggingConfig()
    setup_logger(
        name="rag_system",
        log_level=log_config.log_level,
        log_dir=log_config.log_dir,
        enable_file_logging=log_config.enable_file_logging,
        enable_console_logging=log_config.enable_console_logging,
        max_bytes=log_config.max_bytes,
        backup_count=log_config.backup_count,
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
