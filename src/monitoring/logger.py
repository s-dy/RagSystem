import functools
import json
import inspect
import sys
import time
from loguru import logger
from datetime import datetime
from pathlib import Path

logger.remove()

# 获取当前文件所在目录
current_file_path = Path(__file__).resolve()
monitor_dir = current_file_path.parent  # monitor文件夹路径

#  在monitor文件夹下创建logs目录
log_dir = monitor_dir / "logs"
log_dir.mkdir(exist_ok=True)

# 按日期生成日志文件名
current_date = datetime.now().strftime("%Y-%m-%d")
log_file = log_dir / f"app_{current_date}.log"

# 添加控制台 handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{extra[caller_file]}</cyan>:"
           "<cyan>{extra[caller_func]}</cyan>:"
           "<cyan>{extra[caller_line]}</cyan> - "
           "<level>{message}</level>",
    colorize=True,
    level="INFO"  # 添加级别过滤
)

# 添加文件 handler
logger.add(
    str(log_file),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
           "{level: <8} | "
           "{extra[caller_file]}:"
           "{extra[caller_func]}:"
           "{extra[caller_line]} - "
           "{message}",
    level="INFO",
    rotation="00:00",
    retention="30 days",
    encoding="utf-8",
    enqueue=True,
    compression="zip"
)


def monitor_task_status(title, desc=None, level='INFO'):
    """
    监控任务状态并记录日志
    Args:
        title: 日志标题
        desc: 日志描述/内容
        level: 日志级别 (INFO, WARNING, ERROR, DEBUG)
    """
    # 获取调用者信息 - 需要向上追溯两层
    # 第一层是当前函数本身，第二层是调用这个函数的代码
    stack = inspect.stack()
    if len(stack) > 1:
        caller_frame = stack[1]  # 获取调用者的帧信息
        # 提取文件名（不含路径）
        caller_file = Path(caller_frame.filename).name
        caller_func = caller_frame.function
        caller_line = caller_frame.lineno
    else:
        # 备用方案，如果无法获取调用者信息
        caller_file = "unknown"
        caller_func = "unknown"
        caller_line = 0

    # 清理栈帧引用
    del stack

    # 确保标题和描述是字符串
    if hasattr(title, '__str__'):
        title = str(title)
    if hasattr(desc, '__str__'):
        desc = str(desc)

    # 格式化消息
    if desc is not None:
        try:
            # 尝试将描述转换为JSON字符串
            if isinstance(desc, (dict, list, tuple)):
                message = f"{title}: {json.dumps(desc, ensure_ascii=False, default=str)}"
            else:
                message = f"{title}: {desc}"
        except (TypeError, ValueError):
            # 如果JSON转换失败，使用字符串表示
            message = f"{title}: {desc}"
    else:
        message = f"{title}"

    # 使用patch方法添加上下文信息
    bound_logger = logger.bind(
        caller_file=caller_file,
        caller_func=caller_func,
        caller_line=caller_line
    )

    # 根据级别记录日志
    level_upper = level.upper()
    if level_upper == "INFO":
        bound_logger.info(message)
    elif level_upper == "WARNING":
        bound_logger.warning(message)
    elif level_upper == "ERROR":
        bound_logger.error(message)
    elif level_upper == "DEBUG":
        bound_logger.debug(message)
    else:
        bound_logger.info(message)


def timer(func):
    """
    函数执行时间计时器装饰器（同步函数）
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            end = time.time()
            monitor_task_status(
                f"{func.__module__}.{func.__name__}",
                f"执行成功，耗时: {end - start:.2f}秒",
                level="INFO"
            )
            return result
        except Exception as e:
            end = time.time()
            monitor_task_status(
                f"{func.__module__}.{func.__name__}",
                f"执行失败，耗时: {end - start:.2f}秒，错误: {str(e)}",
                level="ERROR"
            )
            raise

    return wrapper
