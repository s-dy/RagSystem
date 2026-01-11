import os
from pathlib import Path

def set_huggingface_hf_env():
    """设置huggingface环境"""
    # 设置镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ["HF_HUB_ENABLE_HF_MIRROR"] = "true"
    # 设置模型缓存环境
    cache_dir = os.environ.get('HF_HOME') or (Path(os.environ['HOME']) / '.cache' / 'huggingface' / 'hub').as_posix()
    if cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

