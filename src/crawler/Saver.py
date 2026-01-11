import os
import json
import uuid
import threading
from typing import Any, Dict, Optional
from collections import deque


class MetaDataSaver:
    """通用元数据存储器，支持批量保存和立即保存"""
    def __init__(
            self,
            data_dir: str = None,
            batch_size: int = 10,  # 批量阈值
            auto_flush_interval: float = 5.0  # 自动 flush 间隔（秒）
    ):
        """
        Args:
            data_dir: 数据存储目录,默认当前目录下的data文件夹
            batch_size: 缓冲区大小，达到后自动 flush
            auto_flush_interval: 后台自动 flush 间隔（设为 None 则禁用）
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.metadata_file = os.path.join(self.data_dir, "metadata.json")

        # 线程安全
        self._lock = threading.Lock()
        self._pending_writes = deque()  # 待写入队列: [(item_id, metadata), ...]

        # 批量控制
        self.batch_size = batch_size
        self.auto_flush_interval = auto_flush_interval

        # 加载现有数据
        self._load_metadata()

        # 启动后台 flush 线程（如果启用）
        if self.auto_flush_interval is not None:
            self._flush_thread = threading.Thread(target=self._background_flush, daemon=True)
            self._flush_thread.start()

    def _load_metadata(self):
        """加载现有 metadata.json"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data if isinstance(data, dict) else {}
            except Exception as e:
                print(f"⚠️ Failed to load metadata.json: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """原子性保存 metadata.json"""
        temp_file = self.metadata_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, self.metadata_file)
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

    def save_item(
            self,
            metadata: Dict[str, Any],
            item_id: Optional[str] = None,
            immediate: bool = False
    ):
        """
        保存一个数据项

        Args:
            metadata: 元数据字典
            item_id: 可选唯一ID
            immediate: 是否立即写入磁盘（默认 False，加入缓冲区）
        """
        if item_id is None:
            item_id = str(uuid.uuid4())

        with self._lock:
            # 更新内存数据
            self.metadata[item_id] = metadata

            if immediate:
                # 立即写入
                self._save_metadata()
            else:
                # 加入待写入队列
                self._pending_writes.append((item_id, metadata))

                # 检查是否达到批量阈值
                if len(self._pending_writes) >= self.batch_size:
                    self.flush()

    def flush(self):
        """强制将所有待写入数据持久化到磁盘"""
        with self._lock:
            if not self._pending_writes:
                return

            # 清空队列（数据已在 self.metadata 中）
            self._pending_writes.clear()

            # 写入磁盘
            self._save_metadata()
            print(f"💾 Flushed {len(self.metadata)} items to {self.metadata_file}")

    def _background_flush(self):
        """后台自动 flush 线程"""
        import time
        while True:
            time.sleep(self.auto_flush_interval)
            if self._pending_writes:
                self.flush()

    def get_all_items(self) -> Dict[str, Dict[str, Any]]:
        """获取所有已保存项的元数据"""
        with self._lock:
            return self.metadata.copy()

    def get_item_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取元数据"""
        with self._lock:
            return self.metadata.get(item_id)

    def __del__(self):
        """析构时确保数据落盘"""
        if hasattr(self, '_pending_writes') and self._pending_writes:
            self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pending_writes:
            self.flush()


    @property
    def data_dir_path(self):
        return self.data_dir