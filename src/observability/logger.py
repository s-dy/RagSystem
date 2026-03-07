"""
结构化日志系统
提供 JSON 格式日志（文件）+ 人类可读格式（控制台）、请求ID追踪、错误日志分离
"""
import json
import logging
import logging.handlers
import os
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 请求ID上下文变量
_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> None:
    """设置当前请求的ID。"""
    _request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """获取当前请求的ID。"""
    return _request_id_var.get()


class StructuredFormatter(logging.Formatter):
    """
    结构化日志格式化器
    将日志输出为 JSON 格式，便于日志收集和分析
    """

    def format(self, record: logging.LogRecord) -> str:
        request_id = _request_id_var.get()

        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if request_id:
            log_data["request_id"] = request_id

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        if hasattr(record, "extra_context"):
            log_data["context"] = record.extra_context

        log_data["thread"] = record.thread
        log_data["process"] = record.process

        return json.dumps(log_data, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    人类可读的日志格式化器
    用于控制台输出，格式更友好
    """

    def format(self, record: logging.LogRecord) -> str:
        request_id = _request_id_var.get()
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(8)
        logger_name = record.name
        message = record.getMessage()

        if request_id:
            log_line = f"[{timestamp}] {level} [{logger_name}] [RequestID: {request_id}] {message}"
        else:
            log_line = f"[{timestamp}] {level} [{logger_name}] {message}"

        log_line += f" | {record.module}.{record.funcName}:{record.lineno}"

        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        if hasattr(record, "extra_context"):
            context_str = json.dumps(record.extra_context, ensure_ascii=False, indent=2)
            log_line += f"\n上下文信息:\n{context_str}"

        return log_line


def setup_logger(
    name: str = "rag_system",
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    设置并配置日志记录器。

    Args:
        name: 日志记录器名称
        log_level: 日志级别
        log_dir: 日志文件目录
        enable_file_logging: 是否启用文件日志
        enable_console_logging: 是否启用控制台日志
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    log_path = None
    if enable_file_logging:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

    # 控制台处理器（人类可读格式）
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(HumanReadableFormatter())
        logger.addHandler(console_handler)

    # 文件处理器 - 全量日志（JSON 格式）
    if enable_file_logging and log_path:
        all_log_file = log_path / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            all_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    # 文件处理器 - 错误日志（JSON 格式，仅 ERROR 及以上）
    if enable_file_logging and log_path:
        error_log_file = log_path / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        logger.addHandler(error_handler)

    return logger


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any,
) -> None:
    """
    记录带上下文的日志。

    Args:
        logger: 日志记录器
        level: 日志级别
        message: 日志消息
        **context: 额外的上下文信息
    """
    extra = {"extra_context": context}
    logger.log(level, message, extra=extra)


# 全局日志器实例（延迟初始化）
_logger: Optional[logging.Logger] = None


def get_logger(name: str | None = None) -> logging.Logger:
    """
    获取项目统一的日志记录器。

    返回 rag_system 的子日志器，自动继承根日志器的 handlers 和 formatters，
    同时保留模块级别的日志名称便于定位问题。

    Args:
        name: 模块名称，通常传入 __name__。为 None 时返回根日志器。

    Returns:
        配置好的日志记录器
    """
    global _logger
    root_name = "rag_system"

    if _logger is None:
        existing_logger = logging.getLogger(root_name)
        if existing_logger.handlers:
            # 已有配置，直接使用
            _logger = existing_logger
        else:
            log_level = os.getenv("LOG_LEVEL", "INFO")
            enable_file = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
            _logger = setup_logger(
                name=root_name,
                log_level=log_level,
                enable_file_logging=enable_file,
            )

    if name:
        return logging.getLogger(f"{root_name}.{name}")
    return _logger
