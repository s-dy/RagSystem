## 文档上传与入库链路架构

### 整体流程

```
用户上传文件 → 前端 uploadFiles()
    ↓
POST /api/knowledge/upload?collection_name=xxx
    ↓
server.py: 保存文件到 uploads/{collection_name}/
    ↓
asyncio.create_task(_run_ingest(...))  ← 异步后台任务
    ↓
DataDBStorage.ingest(config, data_path)
    ↓
┌──────────────────────────────────────────────┐
│  1. load_document(data_path)                 │  ← file_tool.py
│     支持 PDF / DOCX / Markdown               │
│                                              │
│  2. ChunkHandler 分块                        │  ← chunk.py
│     ├─ Markdown → markdown_chunk()           │
│     └─ 其他格式 → recursive_chunk()           │
│     (可选) parent_child_chunk()               │
│                                              │
│  3. MilvusExecutor.aadd_documents()          │  ← milvus_client.py
│     向量化 + 写入 Milvus                      │
│                                              │
│  (可选) PostgreSQLConnector                   │
│     .batch_insert_parent_documents()         │  ← 父子文档模式
└──────────────────────────────────────────────┘
    ↓
前端 pollIngestStatus() 轮询状态
    ↓
GET /api/knowledge/ingest-status/{collection_name}
    ↓
展示入库结果（成功/失败/超时）
```

### 关键组件

| 组件 | 文件 | 职责 |
|------|------|------|
| **DataDBStorage** | `src/services/data_load/data_storage.py` | 入库编排：加载→分块→向量化→写入 |
| **IngestConfig** | `src/services/data_load/data_storage.py` | 入库配置数据类 |
| **ChunkHandler** | `src/services/data_load/chunk.py` | 文档分块策略 |
| **load_document** | `src/services/data_load/file_tool.py` | 文件解析（PDF/DOCX/MD） |
| **MilvusExecutor** | `src/services/storage/milvus_client.py` | Milvus 向量库客户端（单例池） |
| **PostgreSQLConnector** | `src/services/storage/postgres_connector.py` | 父文档存储 |
| **server.py** | `server.py` | API 层：上传、入库、状态查询 |
| **knowledge.js** | `frontend/js/knowledge.js` | 前端上传与状态轮询 |

### 数据流向

1. **文件存储**：上传的文件保存在 `uploads/{collection_name}/` 目录
2. **向量存储**：分块后的文档向量写入 Milvus 的对应 collection
3. **父文档存储**（可选）：父子文档模式下，父文档写入 PostgreSQL 的 `parent_documents` 表
4. **状态跟踪**：入库状态通过内存字典 `ingest_status_store` 跟踪

### 分块策略

| 策略 | 适用场景 | 说明 |
|------|----------|------|
| `recursive_chunk` | PDF、DOCX 等非结构化文档 | 使用中文优化分隔符递归切分 |
| `markdown_chunk` | Markdown 文档 | 先按标题层级切分，超长章节再递归切分 |
| `parent_child_chunk` | 需要上下文回溯的场景 | 大 chunk 作为父文档，小 chunk 用于检索 |
| `markdown_parent_child_chunk` | Markdown + 父子文档 | 标题层级切分出父文档，再切分子文档 |
