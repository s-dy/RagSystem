import asyncio
from typing import List, Callable, Any, Dict, Union, Tuple, Coroutine, Optional
import threading
from queue import Queue
import time
from multiprocessing import cpu_count
import concurrent.futures
import pickle
import traceback
import inspect


def is_picklable(obj: Any) -> bool:
    """检查对象是否可序列化"""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, TypeError):
        return False

class AsyncParallelExecutor:
    """异步并行执行器（兼容已有事件循环）"""

    @staticmethod
    async def execute_tasks_async(
            tasks: List[Union[Callable, Coroutine]],
            task_args: List[tuple] = None,
            task_kwargs: List[dict] = None,
            max_concurrent: int = None
    ) -> List[Any]:
        """异步并行执行任务（内部实现）"""
        # ... 保持之前的实现不变 ...
        n_tasks = len(tasks)
        task_args = task_args or [()] * n_tasks
        task_kwargs = task_kwargs or [{}] * n_tasks

        async def execute_single_task(task, args, kwargs):
            if inspect.iscoroutinefunction(task):
                return await task(*args, **kwargs)
            elif inspect.isawaitable(task):
                return await task
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: task(*args, **kwargs))

        # 使用信号量控制并发
        semaphore = None
        if max_concurrent:
            semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task, args, kwargs):
            if semaphore:
                async with semaphore:
                    return await execute_single_task(task, args, kwargs)
            return await execute_single_task(task, args, kwargs)

        # 创建所有协程
        coroutines = [
            run_with_semaphore(task, args, kwargs)
            for task, args, kwargs in zip(tasks, task_args, task_kwargs)
        ]

        # 并行执行
        return await asyncio.gather(*coroutines, return_exceptions=True)

    @staticmethod
    def execute_tasks(
            tasks: List[Union[Callable, Coroutine]],
            task_args: List[tuple] = None,
            task_kwargs: List[dict] = None,
            max_concurrent: int = None
    ) -> List[Any]:
        """
        同步接口：智能处理事件循环

        Args:
            tasks: 要执行的任务函数或协程列表
            task_args: 每个任务的位置参数列表
            task_kwargs: 每个任务的关键字参数列表
            max_concurrent: 最大并发数

        Returns:
            任务执行结果列表
        """
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()

            # 检查事件循环是否已经在运行
            if loop.is_running():
                # 在Jupyter/IPython等环境中，使用nest_asyncio或创建新线程
                return AsyncParallelExecutor._run_in_new_thread(
                    tasks, task_args, task_kwargs, max_concurrent
                )
            else:
                # 标准情况：运行事件循环
                return loop.run_until_complete(
                    AsyncParallelExecutor.execute_tasks_async(
                        tasks, task_args, task_kwargs, max_concurrent
                    )
                )
        except RuntimeError:
            # 没有事件循环的情况
            return asyncio.run(
                AsyncParallelExecutor.execute_tasks_async(
                    tasks, task_args, task_kwargs, max_concurrent
                )
            )

    @staticmethod
    def _run_in_new_thread(
            tasks: List[Union[Callable, Coroutine]],
            task_args: List[tuple],
            task_kwargs: List[dict],
            max_concurrent: int
    ) -> List[Any]:
        """在新线程中运行事件循环"""
        import threading
        from concurrent.futures import Future

        result_future = Future()

        def run_in_thread():
            try:
                # 在新线程中创建新的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    AsyncParallelExecutor.execute_tasks_async(
                        tasks, task_args, task_kwargs, max_concurrent
                    )
                )
                result_future.set_result(result)
            except Exception as e:
                result_future.set_exception(e)
            finally:
                loop.close()

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        return result_future.result()

# 异步任务执行器
def parallel_execute(
        tasks: List[Callable],
        task_args: List[tuple] = None,
        task_kwargs: List[dict] = None,
        max_workers: int = None,
        timeout: float = None,
        use_async: bool = True
) -> List[Any]:
    """
    并行执行快捷函数

    Args:
        tasks: 任务函数列表
        task_args: 任务参数列表
        task_kwargs: 任务关键字参数列表
        max_workers: 最大工作线程/协程数
        timeout: 超时时间（仅同步模式有效）
        use_async: 是否使用异步模式

    Returns:
        执行结果列表
    """
    # if use_async:
    return AsyncParallelExecutor.execute_tasks(
        tasks, task_args, task_kwargs, max_workers
    )
    # else:
    #     return ParallelExecutor.execute_tasks(
    #         tasks, task_args, task_kwargs, max_workers, timeout
    #     )

# 多线程执行器
def multi_thread_executor(
        func_list: List[Callable],
        task_args: Optional[List[Tuple]] = None,
        task_kwargs: Optional[List[Dict]] = None,
        max_workers: int = 5,
        timeout: Optional[float] = None
) -> List[Any]:
    """
    使用 threading 模块实现的多线程执行器
    """
    if task_args is None:
        task_args = [()] * len(func_list)
    if task_kwargs is None:
        task_kwargs = [{}] * len(func_list)

    if not (len(func_list) == len(task_args) == len(task_kwargs)):
        raise ValueError("函数列表、参数列表和关键字参数列表长度必须一致")

    n_tasks = len(func_list)
    results = [None] * n_tasks
    errors = [None] * n_tasks

    # 创建任务队列
    task_queue = Queue()
    for i in range(n_tasks):
        task_queue.put((i, func_list[i], task_args[i], task_kwargs[i]))

    # 锁用于安全地访问结果列表
    lock = threading.Lock()

    def worker():
        """工作线程函数"""
        while True:
            try:
                idx, func, args, kwargs = task_queue.get_nowait()
            except:
                # 队列为空，结束线程
                break
            try:
                result = func(*args, **kwargs)
                with lock:
                    results[idx] = result
            except Exception as e:
                with lock:
                    errors[idx] = e
            finally:
                task_queue.task_done()

    # 创建并启动工作线程
    threads = []
    for _ in range(min(max_workers, n_tasks)):
        thread = threading.Thread(target=worker)
        thread.daemon = True  # 设置为守护线程
        thread.start()
        threads.append(thread)

    # 等待所有任务完成或超时
    start_time = time.time()
    task_queue.join()

    # 检查是否有任务未完成
    if timeout is not None and time.time() - start_time > timeout:
        # 清空队列取消剩余任务
        while not task_queue.empty():
            task_queue.get()
            task_queue.task_done()
        raise TimeoutError("任务执行超时")

    # 等待所有线程结束
    for thread in threads:
        thread.join(timeout=1)

    # 如果有错误，可以选择抛出或返回
    if any(errors):
        # 这里可以选择抛出第一个错误或返回错误列表
        # 为了保持接口一致，我们将错误作为结果返回
        for i, error in enumerate(errors):
            if error is not None:
                results[i] = error

    return results

# 多进程执行器
def multiprocess_executor(
        func_list: List[Callable],
        task_args: Optional[List[Tuple]] = None,
        task_kwargs: Optional[List[Dict]] = None,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        chunksize: int = 1,
        use_process_pool: bool = True,
        context: str = 'spawn'  # 'spawn', 'fork', or 'forkserver'
) -> List[Union[Any, Exception]]:
    """
    多进程执行器，适用于CPU密集型任务

    参数:
        func_list: 函数列表
        task_args: 参数列表
        task_kwargs: 关键字参数字典列表
        max_workers: 最大进程数，默认使用CPU核心数
        timeout: 超时时间（秒）
        chunksize: 任务块大小，用于优化性能
        use_process_pool: 使用multiprocessing.Pool还是ProcessPoolExecutor
        context: 进程启动方式

    返回:
        结果列表，包含函数返回值或异常
    """
    if task_args is None:
        task_args = [()] * len(func_list)
    if task_kwargs is None:
        task_kwargs = [{}] * len(func_list)

    if not (len(func_list) == len(task_args) == len(task_kwargs)):
        raise ValueError("函数列表、参数列表和关键字参数列表长度必须一致")

    # 设置默认最大进程数
    if max_workers is None:
        max_workers = cpu_count()

    # 验证函数和参数是否可序列化
    for i, (func, args, kwargs) in enumerate(zip(func_list, task_args, task_kwargs)):
        if not is_picklable(func):
            raise ValueError(f"函数 {i} 不可序列化")
        if not is_picklable(args):
            raise ValueError(f"函数 {i} 的参数不可序列化")
        if not is_picklable(kwargs):
            raise ValueError(f"函数 {i} 的关键字参数不可序列化")

    # 包装函数以捕获异常
    def worker_wrapper(func, args, kwargs, task_id):
        """包装函数，捕获异常并返回"""
        try:
            result = func(*args, **kwargs)
            return {'status': 'success', 'result': result, 'task_id': task_id}
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            return {'status': 'error', 'error': error_info, 'task_id': task_id}

    # 准备任务
    tasks = []
    for i, (func, args, kwargs) in enumerate(zip(func_list, task_args, task_kwargs)):
        # 使用偏函数或lambda包装（但要注意序列化问题）
        from functools import partial
        task_func = partial(worker_wrapper, func, args, kwargs, i)
        tasks.append(task_func)

    results = [None] * len(func_list)

    return _use_processpoolexecutor(tasks, max_workers, timeout)

def _use_processpoolexecutor(tasks, max_workers, timeout):
    """使用 ProcessPoolExecutor 的实现"""
    results = [None] * len(tasks)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {}
            for i, task in enumerate(tasks):
                future = executor.submit(task)
                future_to_index[future] = i

            # 收集结果
            try:
                for future in concurrent.futures.as_completed(future_to_index, timeout=timeout):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        if result['status'] == 'success':
                            results[result['task_id']] = result['result']
                        else:
                            error_info = result['error']
                            exc = Exception(f"{error_info['type']}: {error_info['message']}")
                            results[idx] = exc
                    except Exception as e:
                        results[idx] = e
            except concurrent.futures.TimeoutError:
                raise TimeoutError("任务执行超时")
    except Exception as e:
        return [e] * len(tasks)

    return results

def async_run(func):
    return asyncio.run(func)