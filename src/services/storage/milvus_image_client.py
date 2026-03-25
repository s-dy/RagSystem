"""图片专用 Milvus Collection 客户端

为多模态 RAG 提供图片向量的存储和检索能力。
使用 CLIP 向量（512 维）作为图片 Embedding，支持：
- 图片入库（存储 CLIP 向量 + 图片 base64 + 描述文字）
- 文字查询图片（CLIP 文字向量 → 检索相似图片）
- 图片查询图片（CLIP 图片向量 → 检索相似图片）
"""

import base64
import threading
from dataclasses import dataclass
from typing import List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config import MilvusConfig
from src.observability.logger import get_logger
from src.services.embedding.clip_embedding import get_clip_embedding

logger = get_logger(__name__)

# 图片 Collection 的固定名称后缀，与文本 Collection 区分
IMAGE_COLLECTION_SUFFIX = "_images"

# CLIP 向量维度（clip-vit-base-patch32）
CLIP_DIM = 512


@dataclass
class RetrievedImage:
    """图片检索结果的结构化表示"""

    image_id: str
    caption: str
    source: str
    page_idx: int
    score: float
    image_base64: str


class MilvusImageClient:
    """图片专用 Milvus 客户端，按 collection_name 缓存实例（单例池）。

    Collection Schema：
        - image_id (VARCHAR, PK)：图片唯一标识
        - clip_vector (FLOAT_VECTOR, 512)：CLIP 图片向量
        - caption (VARCHAR)：VLM 生成的图片描述（用于文本检索辅助）
        - source (VARCHAR)：来源文件路径
        - page_idx (INT64)：来源页码
        - image_base64 (VARCHAR)：图片 base64 编码（存储原始图片）
    """

    _instances: dict[str, "MilvusImageClient"] = {}
    _lock = threading.Lock()

    def __new__(cls, config: MilvusConfig = None) -> "MilvusImageClient":
        if config is None:
            config = MilvusConfig()
        collection_name = config.collection_name + IMAGE_COLLECTION_SUFFIX

        with cls._lock:
            if collection_name not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[collection_name] = instance
            return cls._instances[collection_name]

    def __init__(self, config: MilvusConfig = None) -> None:
        if self._initialized:
            return
        if config is None:
            config = MilvusConfig()
        self.config = config
        self.collection_name = config.collection_name + IMAGE_COLLECTION_SUFFIX
        self.clip = get_clip_embedding()
        self._collection: Optional[Collection] = None
        self._connect()
        self._ensure_collection()
        self._initialized = True

    def _connect(self) -> None:
        """建立 Milvus 连接"""
        try:
            connections.connect(
                alias="image_client",
                host=self.config.host,
                port=self.config.port,
                token=self.config.token,
                db_name=self.config.db_name,
            )
            logger.info(f"[MilvusImageClient] 连接成功: {self.config.host}:{self.config.port}")
        except Exception as exc:
            logger.error(f"[MilvusImageClient] 连接失败: {exc}")
            raise

    def _ensure_collection(self) -> None:
        """确保图片 Collection 存在，不存在则自动创建"""
        if utility.has_collection(self.collection_name, using="image_client"):
            self._collection = Collection(self.collection_name, using="image_client")
            self._collection.load()
            logger.info(f"[MilvusImageClient] 复用已有 Collection: {self.collection_name}")
            return

        fields = [
            FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="clip_vector", dtype=DataType.FLOAT_VECTOR, dim=CLIP_DIM),
            FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page_idx", dtype=DataType.INT64),
            FieldSchema(name="image_base64", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields, description="多模态 RAG 图片向量库")
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
            using="image_client",
        )

        # 为 CLIP 向量创建 HNSW 索引
        self._collection.create_index(
            field_name="clip_vector",
            index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}},
        )
        self._collection.load()
        logger.info(f"[MilvusImageClient] 创建新 Collection: {self.collection_name}")

    async def insert_images_async(self, image_records: List[dict]) -> int:
        """批量插入图片向量记录（异步版本，CLIP 推理在线程池中执行）。

        Args:
            image_records: 图片记录列表，每项需包含：
                - image_id (str)：唯一标识
                - image_bytes (bytes)：图片原始字节
                - caption (str)：图片描述文字
                - source (str)：来源文件
                - page_idx (int)：来源页码

        Returns:
            成功插入的记录数
        """
        import asyncio
        return await asyncio.to_thread(self.insert_images, image_records)

    def insert_images(self, image_records: List[dict]) -> int:
        """批量插入图片向量记录（同步版本）。

        Args:
            image_records: 图片记录列表，每项需包含：
                - image_id (str)：唯一标识
                - image_bytes (bytes)：图片原始字节
                - caption (str)：图片描述文字
                - source (str)：来源文件
                - page_idx (int)：来源页码

        Returns:
            成功插入的记录数
        """
        if not image_records:
            return 0

        image_ids, clip_vectors, captions, sources, page_idxs, image_base64s = (
            [], [], [], [], [], [],
        )

        for record in image_records:
            image_bytes = record["image_bytes"]
            try:
                # CLIP 推理为 CPU 密集型，在调用方通过 asyncio.to_thread 异步化
                vector = self.clip.embed_image_bytes(image_bytes)
            except Exception as exc:
                logger.warning(f"[MilvusImageClient] 图片向量化失败，跳过: {exc}")
                continue

            image_ids.append(record["image_id"])
            clip_vectors.append(vector)
            captions.append(record.get("caption", ""))
            sources.append(record.get("source", ""))
            page_idxs.append(int(record.get("page_idx", 0)))
            image_base64s.append(base64.b64encode(image_bytes).decode("utf-8"))

        if not image_ids:
            return 0

        self._collection.insert(
            [image_ids, clip_vectors, captions, sources, page_idxs, image_base64s]
        )
        self._collection.flush()
        logger.info(f"[MilvusImageClient] 插入图片向量: count={len(image_ids)}")
        return len(image_ids)

    def search_by_text(self, query_text: str, top_k: int = 3) -> List[RetrievedImage]:
        """用文字查询检索相关图片（CLIP 跨模态检索）。

        将查询文字编码为 CLIP 向量，在图片向量空间中检索最相似的图片。

        Args:
            query_text: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的图片列表，按相似度降序排列
        """
        try:
            query_vector = self.clip.embed_text(query_text)
            return self._search(query_vector, top_k)
        except Exception as exc:
            logger.warning(f"[MilvusImageClient] 文字检索图片失败: {exc}")
            return []

    def search_by_image_bytes(self, image_bytes: bytes, top_k: int = 3) -> List[RetrievedImage]:
        """用图片查询检索相似图片。

        Args:
            image_bytes: 查询图片的原始字节
            top_k: 返回结果数量

        Returns:
            检索到的相似图片列表
        """
        try:
            query_vector = self.clip.embed_image_bytes(image_bytes)
            return self._search(query_vector, top_k)
        except Exception as exc:
            logger.warning(f"[MilvusImageClient] 图片检索图片失败: {exc}")
            return []

    def _search(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedImage]:
        """执行向量检索并返回结构化结果，支持相似度阈值过滤和 image_id 去重。

        Args:
            query_vector: 查询向量（CLIP 空间，512 维）
            top_k: 返回结果数量上限
            score_threshold: 相似度阈值，低于此值的结果将被过滤；
                             为 None 时从 MultimodalConfig 读取默认值

        Returns:
            RetrievedImage 列表，已按 score 降序排列，已去重
        """
        if score_threshold is None:
            try:
                from config import MultimodalConfig
                score_threshold = MultimodalConfig().image_score_threshold
            except Exception:
                score_threshold = 0.25

        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        # 多取一些候选，给阈值过滤留余量
        candidate_limit = max(top_k * 3, top_k + 10)
        results = self._collection.search(
            data=[query_vector],
            anns_field="clip_vector",
            param=search_params,
            limit=candidate_limit,
            output_fields=["image_id", "caption", "source", "page_idx", "image_base64"],
        )

        seen_image_ids: set[str] = set()
        retrieved = []
        for hit in results[0]:
            score = float(hit.score)
            # 阈值过滤：低于阈值的图片不返回
            if score < score_threshold:
                continue
            image_id = hit.entity.get("image_id", "")
            # image_id 去重：同一图片只保留最高分的那条
            if image_id in seen_image_ids:
                continue
            seen_image_ids.add(image_id)
            retrieved.append(
                RetrievedImage(
                    image_id=image_id,
                    caption=hit.entity.get("caption", ""),
                    source=hit.entity.get("source", ""),
                    page_idx=int(hit.entity.get("page_idx", 0)),
                    score=score,
                    image_base64=hit.entity.get("image_base64", ""),
                )
            )
            if len(retrieved) >= top_k:
                break

        logger.debug(
            f"[MilvusImageClient] 向量检索完成: top_k={top_k}, threshold={score_threshold:.2f}, found={len(retrieved)}"
        )
        return retrieved
