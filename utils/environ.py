import os
from pathlib import Path

def set_huggingface_hf_env():
    """设置huggingface环境"""
    # 设置镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ["HF_HUB_ENABLE_HF_MIRROR"] = "true"
    # 设置模型缓存环境
    cache_dir = os.environ.get('HF_HOME', "") or (Path(os.environ.get("HOME","")) / '.cache' / 'huggingface' / 'hub').as_posix()
    if cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
    
    # 检查关键模型是否已在本地缓存，如果是则设置离线模式
    # 避免 transformers 尝试联网下载
    reranker_cache = Path(cache_dir) / "models--BAAI--bge-reranker-v2-m3"
    if reranker_cache.exists():
        # 检查是否有完整的 snapshot
        snapshots_dir = reranker_cache / "snapshots"
        if snapshots_dir.exists() and list(snapshots_dir.glob("*")):
            print(f"[environ] 检测到本地 reranker 模型缓存，设置离线模式")
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

