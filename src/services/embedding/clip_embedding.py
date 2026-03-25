"""CLIP 跨模态 Embedding 模块

提供图片和文字在同一向量空间中的 Embedding，用于多模态 RAG 的跨模态检索。
文字查询可以直接检索图片，图片查询也可以检索相关文本。

依赖：
    pip install transformers torch pillow

配置：
    CLIP_MODEL_PATH：CLIP 模型路径，支持本地路径或 HuggingFace 模型名
                     （默认 openai/clip-vit-base-patch32）
"""

import io
import threading
from typing import List

from src.observability.logger import get_logger

logger = get_logger(__name__)


def _default_clip_model_path() -> str:
    """从 MultimodalConfig 读取默认 CLIP 模型路径，支持 CLIP_MODEL_PATH 环境变量"""
    try:
        from config import MultimodalConfig
        return MultimodalConfig().clip_model_path
    except Exception:
        return "openai/clip-vit-base-patch32"


class CLIPEmbedding:
    """CLIP 跨模态 Embedding，支持图片和文字在同一向量空间中表示。

    使用单例模式缓存模型，避免重复加载。
    支持 HuggingFace 模型名（如 openai/clip-vit-base-patch32）和本地模型路径，
    优先读取 CLIP_MODEL_PATH 环境变量，也可通过 model_name 参数显式指定。
    """

    _instances: dict[str, "CLIPEmbedding"] = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "") -> "CLIPEmbedding":
        resolved_name = model_name or _default_clip_model_path()
        with cls._lock:
            if resolved_name not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[resolved_name] = instance
            return cls._instances[resolved_name]

    def __init__(self, model_name: str = "") -> None:
        if self._initialized:
            return
        self.model_name = model_name or _default_clip_model_path()
        self._model = None
        self._processor = None
        self._initialized = True

    def _ensure_loaded(self) -> None:
        """懒加载 CLIP 模型，首次调用时才加载，避免启动时占用资源。

        支持本地路径（生产环境离线部署）和 HuggingFace Hub 在线下载。
        """
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"[CLIPEmbedding] 加载 CLIP 模型: {self.model_name}")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()
            logger.info("[CLIPEmbedding] CLIP 模型加载完成")
        except ImportError as exc:
            raise ImportError(
                "CLIPEmbedding 需要安装 transformers 和 torch。"
                "请执行: pip install transformers torch pillow"
            ) from exc

    @property
    def embedding_dim(self) -> int:
        """返回 CLIP 向量维度（clip-vit-base-patch32 为 512）"""
        return 512

    def embed_image_bytes(self, image_bytes: bytes) -> List[float]:
        """将图片字节编码为 CLIP 向量。

        Args:
            image_bytes: 图片的原始字节数据

        Returns:
            512 维的浮点向量列表
        """
        import torch
        from PIL import Image

        self._ensure_loaded()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
            # L2 归一化，使余弦相似度等价于内积
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def embed_text(self, text: str) -> List[float]:
        """将文字编码为 CLIP 向量（与图片向量在同一空间）。

        Args:
            text: 查询文本

        Returns:
            512 维的浮点向量列表
        """
        import torch

        self._ensure_loaded()
        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量将文字编码为 CLIP 向量。

        Args:
            texts: 查询文本列表

        Returns:
            向量列表，每项为 512 维浮点列表
        """
        import torch

        self._ensure_loaded()
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.tolist()


def get_clip_embedding(model_name: str = "") -> CLIPEmbedding:
    """获取 CLIPEmbedding 单例实例。

    model_name 为空时自动读取 CLIP_MODEL_PATH 环境变量（通过 MultimodalConfig），
    支持本地路径和 HuggingFace 模型名。

    Args:
        model_name: CLIP 模型名称或本地路径，为空则使用 CLIP_MODEL_PATH 环境变量

    Returns:
        CLIPEmbedding 实例
    """
    return CLIPEmbedding(model_name)
