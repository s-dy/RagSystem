import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

from src.core.exceptions import (
    HybridRagError,
    StorageError,
    MilvusConnectionError,
)
from src.graph import Graph
from src.observability.logger import get_logger
from src.services.storage.milvus_client import ensure_milvus_database_exists
from config import MilvusConfig

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """管理应用生命周期：启动时预热 Graph，关闭时释放 PostgreSQL 连接"""
    # 启动时显式确保 Milvus 数据库存在（若配置了 Milvus）
    try:
        ensure_milvus_database_exists(MilvusConfig())
        logger.info("[Server] Milvus 数据库检查完成")
    except Exception as e:
        logger.warning(f"[Server] 启动时 Milvus 数据库检查失败: {e}")

    yield
    global rag_graph
    if rag_graph is not None:
        await rag_graph.close()
        logger.info("[Server] Graph 连接已关闭")


app = FastAPI(title="HybridRAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HybridRagError)
async def hybrid_rag_error_handler(request: Request, exc: HybridRagError):
    """统一处理 HybridRAG 业务异常，返回结构化错误响应"""
    status_code = 503 if isinstance(exc, (MilvusConnectionError, StorageError)) else 500
    return JSONResponse(
        {"error": type(exc).__name__, "message": str(exc)},
        status_code=status_code,
    )


# 全局 Graph 实例
rag_graph: Optional[Graph] = None

# 会话存储（简易内存版，生产环境应持久化）
conversations: dict[str, dict] = {}


async def get_graph() -> Graph:
    global rag_graph
    if rag_graph is None:
        rag_graph = Graph()
        await rag_graph._compile_graph()
    return rag_graph


async def resolve_internal_collection_name(collection_name: str) -> str:
    """支持 display_name 或 internal_name，同时返回 internal_name。"""
    if not collection_name:
        return ""
    from src.services.storage import PostgreSQLConnector
    from src.services.storage.milvus_client import sanitize_collection_name

    # 如果已经是正在处理的 internal_name，则直接返回
    if collection_name in ingest_status_store:
        return collection_name

    # 尝试通过 display_name 查询 internal_name
    try:
        internal_name = PostgreSQLConnector().get_internal_name_by_display_name(
            collection_name
        )
        if internal_name:
            return internal_name
    except Exception:
        pass

    # 如果输入值已是 internal_name，则直接返回，避免重复 sanitize 变更值
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*_[0-9a-f]{8}$", collection_name):
        return collection_name

    # 否则按用户输入生成 internal_name
    return sanitize_collection_name(collection_name)


# ─────────────────────────── 对话 API ───────────────────────────


@app.post("/api/chat")
async def chat(request: Request):
    """非流式对话接口"""
    body = await request.json()
    message = body.get("message", "")
    thread_id = body.get("thread_id", str(uuid.uuid4()))
    user_id = body.get("user_id", "default")
    enable_web_search = body.get("enable_web_search", False)

    if not message.strip():
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    logger.info(
        f"[Server] 非流式对话请求: thread_id={thread_id}, user_id={user_id}, message={message[:50]}..."
    )

    graph = await get_graph()
    input_data = {"messages": [HumanMessage(content=message)]}
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "enable_web_search": enable_web_search,
        }
    }

    result = await graph.start(input_data, config)
    answer = result.get("answer", "")

    # 记录会话
    if thread_id not in conversations:
        conversations[thread_id] = {
            "id": thread_id,
            "title": message[:30],
            "messages": [],
        }
    conversations[thread_id]["messages"].append({"role": "user", "content": message})
    conversations[thread_id]["messages"].append(
        {"role": "assistant", "content": answer}
    )

    return JSONResponse(
        {
            "answer": answer,
            "thread_id": thread_id,
        }
    )


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """SSE 流式对话接口"""
    body = await request.json()
    message = body.get("message", "")
    thread_id = body.get("thread_id", str(uuid.uuid4()))
    user_id = body.get("user_id", "default")
    enable_web_search = body.get("enable_web_search", False)

    if not message.strip():
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    # 记录用户消息
    if thread_id not in conversations:
        conversations[thread_id] = {
            "id": thread_id,
            "title": message[:30],
            "messages": [],
        }
    conversations[thread_id]["messages"].append({"role": "user", "content": message})

    async def event_generator():
        graph = await get_graph()
        input_data = {"messages": [HumanMessage(content=message)]}
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "enable_web_search": enable_web_search,
            }
        }

        full_answer = ""

        # 心跳任务：每 15 秒发送一次 keepalive，防止连接超时断开
        heartbeat_event = asyncio.Event()

        # 使用 asyncio.Queue 合并心跳和主事件流
        queue = asyncio.Queue()

        async def stream_events():
            try:
                async for event in graph.start_stream(input_data, config):
                    await queue.put(event)
            except Exception as exc:
                logger.error(
                    f"[Server] 流式对话异常: thread_id={thread_id}, error={exc}"
                )
                await queue.put({"type": "error", "content": str(exc)})
            finally:
                await queue.put(None)  # 结束标记

        async def send_heartbeats():
            while not heartbeat_event.is_set():
                await asyncio.sleep(15)
                if not heartbeat_event.is_set():
                    await queue.put({"type": "heartbeat"})

        # 启动事件流和心跳
        stream_task = asyncio.create_task(stream_events())
        heartbeat_task = asyncio.create_task(send_heartbeats())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break

                event_type = event.get("type", "")

                if event_type == "heartbeat":
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if event_type == "token":
                    full_answer += event.get("content", "")
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif event_type == "decomposition":
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif event_type == "sub_answer":
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif event_type == "retrieval_progress":
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif event_type == "final_answer":
                    full_answer = event.get("answer", full_answer)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif event_type == "done":
                    # 记录助手回复
                    conversations[thread_id]["messages"].append(
                        {
                            "role": "assistant",
                            "content": full_answer,
                        }
                    )
                    logger.info(
                        f"[Server] 流式对话完成: thread_id={thread_id}, answer_length={len(full_answer)}"
                    )
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif event_type == "error":
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        finally:
            heartbeat_event.set()
            heartbeat_task.cancel()
            stream_task.cancel()
            for task in (heartbeat_task, stream_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────── 会话管理 API ───────────────────────────


@app.get("/api/conversations")
async def list_conversations():
    """获取所有会话列表（按创建时间倒序）"""
    result = []
    for thread_id, conv in conversations.items():
        result.append(
            {
                "id": conv["id"],
                "title": conv.get("title", "新对话"),
                "message_count": len(conv["messages"]),
            }
        )
    # 最新的会话排在前面
    result.reverse()
    return JSONResponse(result)


@app.get("/api/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    """获取单个会话的消息历史"""
    conv = conversations.get(thread_id)
    if not conv:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    return JSONResponse(conv)


@app.delete("/api/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """删除会话"""
    if thread_id in conversations:
        del conversations[thread_id]
    return JSONResponse({"ok": True})


# ─────────────────────────── 知识库管理 API ───────────────────────────


@app.get("/api/knowledge/collections")
async def list_collections():
    """列出所有知识库集合"""
    try:
        from pymilvus import connections, utility
        from config import MilvusConfig
        from src.services.storage import PostgreSQLConnector

        milvus_config = MilvusConfig()
        connections.connect(
            alias="default",
            host=milvus_config.host,
            port=milvus_config.port,
            db_name=milvus_config.db_name,
            token=milvus_config.token,
        )

        # 首先从 PostgreSQL 读取已保存的知识库元数据
        knowledge_bases = PostgreSQLConnector().get_all_collections()
        
        # 获取 Milvus 中实际存在的所有集合名称
        actual_collections = set(utility.list_collections())
        logger.info(f"[Server] Milvus 中实际存在的集合: {actual_collections}")
        
        collections_info = []
        if knowledge_bases:
            for kb in knowledge_bases:
                index_name = kb.get("index")
                display_name = kb.get("display_name") or index_name
                num_entities = 0
                
                # 检查该集合是否在 Milvus 中实际存在
                if index_name not in actual_collections:
                    logger.warning(f"[Server] 集合 '{index_name}' 在 Milvus 中不存在")
                    # 仍然返回信息，但 num_entities 为 0
                    collections_info.append(
                        {
                            "name": index_name,
                            "internal_name": index_name,
                            "display_name": display_name,
                            "description": kb.get("description", ""),
                            "domain": kb.get("domain", ""),
                            "keywords": kb.get("keywords", []),
                            "num_entities": 0,
                        }
                    )
                    continue
                
                try:
                    from pymilvus import Collection

                    coll = Collection(index_name)
                    coll.flush()
                    num_entities = coll.num_entities
                    logger.info(f"[Server] 获取集合实体数: collection={index_name}, num_entities={num_entities}")
                except Exception as e:
                    logger.warning(f"[Server] 获取集合实体数失败: collection={index_name}, error={e}")
                    num_entities = 0

                collections_info.append(
                    {
                        "name": index_name,
                        "internal_name": index_name,
                        "display_name": display_name,
                        "description": kb.get("description", ""),
                        "domain": kb.get("domain", ""),
                        "keywords": kb.get("keywords", []),
                        "num_entities": num_entities,
                    }
                )
        else:
            collection_names = utility.list_collections()
            for name in collection_names:
                from pymilvus import Collection

                coll = Collection(name)
                coll.flush()
                collections_info.append(
                    {
                        "name": name,
                        "internal_name": name,
                        "display_name": name,
                        "num_entities": coll.num_entities,
                    }
                )

        connections.disconnect("default")
        return JSONResponse(collections_info)
    except ConnectionError as conn_err:
        raise MilvusConnectionError("无法连接 Milvus 服务", cause=conn_err)
    except HybridRagError:
        raise
    except Exception as exc:
        raise StorageError("列出知识库集合失败", cause=exc)


@app.delete("/api/knowledge/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """删除知识库集合"""
    internal_name = await resolve_internal_collection_name(collection_name)
    try:
        from pymilvus import connections, utility
        from config import MilvusConfig

        milvus_config = MilvusConfig()
        connections.connect(
            alias="default",
            host=milvus_config.host,
            port=milvus_config.port,
            db_name=milvus_config.db_name,
            token=milvus_config.token,
        )
        utility.drop_collection(internal_name)
        connections.disconnect("default")

        # 同步删除 PostgreSQL 中的知识库配置
        try:
            from src.services.storage import PostgreSQLConnector

            PostgreSQLConnector().delete_knowledge_collection(internal_name)
        except Exception as pg_err:
            logger.warning(f"[Server] 删除 PostgreSQL 知识库配置失败: {pg_err}")

        return JSONResponse({"ok": True})
    except ConnectionError as conn_err:
        raise MilvusConnectionError("无法连接 Milvus 服务", cause=conn_err)
    except HybridRagError:
        raise
    except Exception as exc:
        raise StorageError(f"删除知识库「{collection_name}」失败", cause=exc)


@app.post("/api/knowledge/upload")
async def upload_document(
    files: list[UploadFile] = File(...),
    request: Request = None,
):
    """上传文档到知识库，并自动触发分块→向量化→入库流程。

    支持通过 query 参数：
    - collection_name: 目标知识库名称（默认 'default'）
    - chunk_size: 分块大小（默认读取全局配置）
    - chunk_overlap: 分块重叠（默认读取全局配置）
    - use_parent_child: 是否使用父子文档策略（默认 false）
    """
    import os
    from urllib.parse import unquote
    from src.services.data_load import IngestConfig

    collection_name = (
        request.query_params.get("collection_name", "default") if request else "default"
    )
    collection_name = unquote(collection_name).strip()
    if not collection_name:
        collection_name = "default"

    # 保留原始用户输入的 display_name，用于前端展示
    raw_collection_name = collection_name

    # 支持 collection_name 使用 display_name 或 internal_name
    if request and request.query_params.get("collection_name"):
        collection_name = await resolve_internal_collection_name(collection_name)

    # 生成后端使用的 internal_name（Sanitize）
    from src.services.storage.milvus_client import sanitize_collection_name
    internal_name = (
        collection_name
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*_[0-9a-f]{8}$", collection_name)
        else sanitize_collection_name(collection_name)
    )

    # 读取 chunk 配置：优先使用请求参数，其次使用全局配置
    global_chunk_config = getattr(
        app.state, "chunk_config", {"chunk_size": 500, "chunk_overlap": 50}
    )
    chunk_size = (
        int(request.query_params.get("chunk_size", global_chunk_config["chunk_size"]))
        if request
        else global_chunk_config["chunk_size"]
    )
    chunk_overlap = (
        int(
            request.query_params.get(
                "chunk_overlap", global_chunk_config["chunk_overlap"]
            )
        )
        if request
        else global_chunk_config["chunk_overlap"]
    )
    use_parent_child = (
        request.query_params.get("use_parent_child", "true").lower() == "true"
        if request
        else True
    )

    # 知识库元数据（仅新建知识库时需要）
    is_new_collection = (
        request.query_params.get("is_new_collection", "false").lower() == "true"
        if request
        else False
    )
    collection_description = (
        unquote(request.query_params.get("description", "")).strip() if request else ""
    )
    collection_domain = (
        unquote(request.query_params.get("domain", "")).strip() if request else ""
    )
    collection_keywords_raw = (
        unquote(request.query_params.get("keywords", "")).strip() if request else ""
    )

    upload_dir = os.path.join(os.path.dirname(__file__), "uploads", internal_name)
    os.makedirs(upload_dir, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved_files.append(
            {
                "filename": file.filename,
                "size": len(content),
                "path": file_path,
            }
        )

    # 异步触发入库流程
    ingest_config = IngestConfig(
        collection_name=internal_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_parent_child=use_parent_child,
    )
    collection_meta = (
        {
            "is_new": is_new_collection,
            "description": collection_description,
            "domain": collection_domain,
            "keywords_raw": collection_keywords_raw,
            "display_name": raw_collection_name,
            "internal_name": internal_name,
        }
        if is_new_collection
        else None
    )

    # 先行写入元数据，避免刚创建的知识库展示为 internal_name
    if collection_meta and collection_meta.get("is_new"):
        try:
            from src.services.storage import PostgreSQLConnector

            keywords_raw = collection_meta.get("keywords_raw", "")
            keywords_list = (
                [k.strip() for k in keywords_raw.split(",") if k.strip()]
                if keywords_raw
                else []
            )
            PostgreSQLConnector().insert_knowledge_collection(
                {
                    "internal_name": internal_name,
                    "display_name": raw_collection_name,
                    "description": collection_description
                    or f"知识库：{raw_collection_name}",
                    "domain": collection_domain or "general",
                    "keywords": keywords_list,
                }
            )
        except Exception as pg_err:
            logger.warning(f"[Server] 先行写入知识库配置失败: {pg_err}")

    logger.info(
        f"[Server] 文档上传请求: collection={collection_name}, files_count={len(files)}"
    )
    asyncio.create_task(_run_ingest(upload_dir, ingest_config, collection_meta))

    logger.info(
        f"[Server] 文档上传完成: display_name={collection_name}, internal_name={internal_name}, saved_files={len(saved_files)}"
    )

    return JSONResponse(
        {
            "message": f"成功上传 {len(saved_files)} 个文件到知识库「{raw_collection_name}」，入库处理已启动",
            "collection_name": internal_name,
            "display_name": raw_collection_name,
            "files": saved_files,
            "ingest_status": "processing",
            "chunk_config": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "use_parent_child": use_parent_child,
            },
        }
    )


# 入库任务状态跟踪（内存版，按 collection_name 记录最近一次入库状态）
ingest_status_store: dict[str, dict] = {}


async def _run_ingest(data_path: str, config, collection_meta: dict = None):
    """后台执行入库流程，更新状态到 ingest_status_store"""
    from src.services.data_load import DataDBStorage

    logger.info(
        f"[Server] 开始入库任务: collection={config.collection_name}, path={data_path}"
    )
    ingest_status_store[config.collection_name] = {
        "status": "processing",
        "message": "正在处理文档...",
        "result": None,
    }
    try:
        storage = DataDBStorage(collection_name=config.collection_name)
        result = await storage.ingest(config, data_path)

        # 仅新建知识库时，将配置写入 PostgreSQL（用于查询路由）
        if collection_meta and collection_meta.get("is_new"):
            try:
                from src.services.storage import PostgreSQLConnector

                keywords_raw = collection_meta.get("keywords_raw", "")
                keywords_list = (
                    [k.strip() for k in keywords_raw.split(",") if k.strip()]
                    if keywords_raw
                    else []
                )
                PostgreSQLConnector().insert_knowledge_collection(
                    {
                        "index": config.collection_name,
                        "internal_name": config.collection_name,
                        "display_name": collection_meta.get("display_name") or collection_meta.get("internal_name") or config.collection_name,
                        "description": collection_meta.get("description")
                        or f"知识库：{config.collection_name}",
                        "domain": collection_meta.get("domain") or "general",
                        "keywords": keywords_list,
                    }
                )
            except Exception as pg_err:
                logger.warning(f"[Server] 写入知识库配置失败: {pg_err}")

        ingest_status_store[config.collection_name] = {
            "status": "completed",
            "message": f"入库完成，共生成 {result.get('total_chunks', 0)} 个分块",
            "result": result,
        }
        logger.info(
            f"[Server] 入库任务完成: collection={config.collection_name}, total_chunks={result.get('total_chunks', 0)}"
        )
    except StorageError as storage_err:
        logger.error(
            f"[Server] 入库失败(存储层): collection={config.collection_name}, error={storage_err}"
        )
        ingest_status_store[config.collection_name] = {
            "status": "failed",
            "message": f"入库失败（存储层）: {storage_err}",
            "result": None,
        }
    except Exception as exc:
        logger.error(
            f"[Server] 入库失败: collection={config.collection_name}, error={exc}"
        )
        ingest_status_store[config.collection_name] = {
            "status": "failed",
            "message": f"入库失败: {exc}",
            "result": None,
        }


@app.get("/api/knowledge/ingest-status/{collection_name}")
async def get_ingest_status(collection_name: str):
    """查询入库处理状态"""
    internal_name = await resolve_internal_collection_name(collection_name)
    status = ingest_status_store.get(internal_name)
    if not status:
        return JSONResponse(
            {"status": "unknown", "message": "未找到该知识库的入库记录"}
        )
    return JSONResponse(status)


# ─────────────────────────── 系统 API ───────────────────────────


@app.get("/api/system/models")
async def get_model_status():
    """获取当前模型配置状态"""
    import os

    llm_model = os.getenv("QWEN_MODEL_NAME") or os.getenv("DASHSCOPE_MODEL_NAME") or "未配置"
    llm_base_url = os.getenv("QWEN_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", "")
    llm_provider = (
        "DashScope (通义千问)" if "dashscope" in llm_base_url else llm_base_url or "未知"
    )

    embedding_model = "qwen3-embedding:0.6B"
    embedding_provider = "Ollama (本地)"

    reranker_model = "BAAI/bge-reranker-v2-m3"
    reranker_provider = "HuggingFace (本地)"

    return JSONResponse(
        {
            "llm": {
                "name": llm_model,
                "provider": llm_provider,
                "status": "online",
            },
            "embedding": {
                "name": embedding_model,
                "provider": embedding_provider,
                "status": "online",
            },
            "reranker": {
                "name": reranker_model,
                "provider": reranker_provider,
                "status": "online",
            },
        }
    )


@app.get("/api/knowledge/documents")
async def list_documents(collection_name: str = ""):
    """列出已上传的文档文件"""
    import os
    from datetime import datetime

    if collection_name:
        collection_name = await resolve_internal_collection_name(collection_name)

    uploads_root = os.path.join(os.path.dirname(__file__), "uploads")
    if not os.path.exists(uploads_root):
        return JSONResponse([])

    collection_display_map = {}
    try:
        from src.services.storage import PostgreSQLConnector

        collection_display_map = {
            item["index"]: item["display_name"]
            for item in PostgreSQLConnector().get_all_collections()
        }
    except Exception:
        collection_display_map = {}

    documents = []

    if collection_name:
        target_dirs = [collection_name]
    else:
        target_dirs = [
            entry
            for entry in os.listdir(uploads_root)
            if os.path.isdir(os.path.join(uploads_root, entry))
        ]

    for dir_name in target_dirs:
        dir_path = os.path.join(uploads_root, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if not os.path.isfile(file_path):
                continue
            stat = os.stat(file_path)
            documents.append(
                {
                    "filename": filename,
                    "size": stat.st_size,
                    "collection_name": dir_name,
                    "collection_display_name": collection_display_map.get(dir_name, dir_name),
                    "upload_time": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "status": "uploaded",
                }
            )

    documents.sort(key=lambda doc: doc["upload_time"], reverse=True)
    return JSONResponse(documents)


@app.delete("/api/knowledge/documents")
async def delete_document(request: Request):
    """删除指定文档"""
    import os

    body = await request.json()
    collection_name = body.get("collection_name", "")
    filename = body.get("filename", "")

    if not collection_name or not filename:
        return JSONResponse(
            {"error": "缺少 collection_name 或 filename"}, status_code=400
        )

    collection_name = await resolve_internal_collection_name(collection_name)
    file_path = os.path.join(
        os.path.dirname(__file__), "uploads", collection_name, filename
    )

    if not os.path.exists(file_path):
        return JSONResponse({"error": "文件不存在"}, status_code=404)

    os.remove(file_path)
    return JSONResponse({"ok": True, "message": f"已删除文件 {filename}"})


@app.post("/api/knowledge/chunk-config")
async def save_chunk_config(request: Request):
    """保存分块配置（存储在内存中，供后续上传使用）"""
    body = await request.json()
    chunk_size = body.get("chunk_size", 500)
    chunk_overlap = body.get("chunk_overlap", 50)

    chunk_size = max(100, min(2000, int(chunk_size)))
    chunk_overlap = max(0, min(500, int(chunk_overlap)))

    app.state.chunk_config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    return JSONResponse(
        {
            "ok": True,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
    )


@app.get("/api/knowledge/chunk-config")
async def get_chunk_config():
    """获取当前分块配置"""
    config = getattr(
        app.state,
        "chunk_config",
        {
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
    )
    return JSONResponse(config)


# ─────────────────────────── 静态文件 & 启动 ───────────────────────────

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
