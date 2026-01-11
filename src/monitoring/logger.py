import functools
import json
import inspect
import sys
import time
from loguru import logger

# 移除默认 handler
logger.remove()

# 重新添加控制台 handler，保留颜色 + 自定义格式
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)

def monitor_task_status(title, desc=None, level='INFO'):
    # 获取调用者文件名
    caller_frame = inspect.currentframe().f_back
    caller_filename = caller_frame.f_code.co_filename.split('/')[-1].split('\\')[-1]
    del caller_frame

    if hasattr(title,'__str__'):
        title = title.__str__()
    if hasattr(desc,'__str__'):
        desc = desc.__str__()
    # 格式化消息
    if desc is not None:
        message = f"[{caller_filename}] {title}: {json.dumps(desc, ensure_ascii=False)}"
    else:
        message = f"[{caller_filename}] {title}"

    # 使用 loguru 直接输出
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "DEBUG":
        logger.debug(message)
    else:
        logger.debug(message)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        monitor_task_status(f"{func.__module__}.{func.__name__}", f"{end-start}")
        return result
    return wrapper