"""
测试切分策略脚本

测试以下切分策略：
1. 中文优化递归切分（recursive_chunk）
2. 父子文档切分（parent_child_chunk）

输出每种策略的切分数量、chunk 长度分布、示例内容等统计信息。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.data_load import ChunkHandler, load_document

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data", "知识图谱构建技术综述_刘峤.pdf")


def print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_chunk_stats(chunks, label: str):
    """打印 chunk 列表的统计信息。"""
    if not chunks:
        print(f"  [{label}] 无 chunk 产出")
        return

    lengths = [len(chunk.page_content) for chunk in chunks]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    print(f"\n  [{label}] 统计信息：")
    print(f"    总数量:     {len(chunks)}")
    print(f"    平均长度:   {avg_length:.0f} 字符")
    print(f"    最短:       {min_length} 字符")
    print(f"    最长:       {max_length} 字符")

    # 长度分布
    brackets = [(0, 100), (100, 300), (300, 500), (500, 800), (800, 1024), (1024, 1500), (1500, float("inf"))]
    print(f"    长度分布:")
    for low, high in brackets:
        count = sum(1 for length in lengths if low <= length < high)
        if count > 0:
            high_label = f"{high}" if high != float("inf") else "∞"
            print(f"      [{low:>5}, {high_label:>5}): {count} 个")

    # 打印前 3 个 chunk 的摘要
    print(f"\n    前 3 个 chunk 摘要:")
    for i, chunk in enumerate(chunks[:3]):
        content_preview = chunk.page_content[:80].replace("\n", "\\n")
        metadata_keys = list(chunk.metadata.keys())
        print(f"      [{i}] ({len(chunk.page_content)} 字符) metadata={metadata_keys}")
        print(f"          \"{content_preview}...\"")


def test_recursive_chunk(documents):
    """测试中文优化递归切分。"""
    print_separator("策略 1: 中文优化递归切分 (recursive_chunk)")

    handler = ChunkHandler()
    chunks = handler.recursive_chunk(documents, chunk_size=1024, chunk_overlap=128)
    print_chunk_stats(chunks, "recursive_chunk(1024/128)")

    # 对比不同参数
    chunks_small = handler.recursive_chunk(documents, chunk_size=512, chunk_overlap=64)
    print_chunk_stats(chunks_small, "recursive_chunk(512/64)")

    chunks_large = handler.recursive_chunk(documents, chunk_size=1500, chunk_overlap=200)
    print_chunk_stats(chunks_large, "recursive_chunk(1500/200)")

    return chunks


def test_parent_child_chunk(documents):
    """测试父子文档切分。"""
    print_separator("策略 2: 父子文档切分 (parent_child_chunk)")

    handler = ChunkHandler()
    parent_store, child_docs = handler.parent_child_chunk(
        documents, parent_size=1500, parent_overlap=200, child_size=400, child_overlap=64
    )

    print(f"\n  父文档数量: {len(parent_store)}")
    print(f"  子文档数量: {len(child_docs)}")

    # 父文档统计
    parent_docs_list = list(parent_store.values())
    print_chunk_stats(parent_docs_list, "父文档")

    # 子文档统计
    print_chunk_stats(child_docs, "子文档")

    # 验证父子关系
    parent_ids_from_children = set()
    for child in child_docs:
        parent_id = child.metadata.get("parent_id")
        if parent_id:
            parent_ids_from_children.add(parent_id)

    print(f"\n  父子关系验证:")
    print(f"    子文档引用的 parent_id 去重数: {len(parent_ids_from_children)}")
    print(f"    父文档实际数量:               {len(parent_store)}")
    print(f"    所有 parent_id 都能匹配:      {parent_ids_from_children == set(parent_store.keys())}")

    # 展示一组父子文档示例
    if parent_store:
        first_parent_id = list(parent_store.keys())[0]
        first_parent = parent_store[first_parent_id]
        matching_children = [c for c in child_docs if c.metadata.get("parent_id") == first_parent_id]
        print(f"\n  示例父子文档 (parent_id={first_parent_id}):")
        print(f"    父文档长度: {len(first_parent.page_content)} 字符")
        print(f"    子文档数量: {len(matching_children)}")
        for i, child in enumerate(matching_children):
            preview = child.page_content[:60].replace("\n", "\\n")
            print(f"      子文档[{i}]: ({len(child.page_content)} 字符) \"{preview}...\"")

    return parent_store, child_docs


def test_comparison(documents):
    """对比所有策略的切分效果。"""
    print_separator("策略对比总结")

    handler = ChunkHandler()

    recursive_chunks = handler.recursive_chunk(documents, chunk_size=1024, chunk_overlap=128)
    parent_store, child_docs = handler.parent_child_chunk(documents)

    recursive_lengths = [len(c.page_content) for c in recursive_chunks]
    child_lengths = [len(c.page_content) for c in child_docs]
    parent_lengths = [len(p.page_content) for p in parent_store.values()]

    original_total_chars = sum(len(doc.page_content) for doc in documents)

    print(f"\n  原始文档总字符数: {original_total_chars}")
    print(f"\n  {'策略':<30} {'chunk数':>8} {'平均长度':>8} {'最短':>6} {'最长':>6}")
    print(f"  {'-' * 68}")
    print(f"  {'递归切分(1024/128)':<30} {len(recursive_chunks):>8} {sum(recursive_lengths)/len(recursive_lengths):>8.0f} {min(recursive_lengths):>6} {max(recursive_lengths):>6}")
    print(f"  {'父子-父文档(1500/200)':<30} {len(parent_store):>8} {sum(parent_lengths)/len(parent_lengths):>8.0f} {min(parent_lengths):>6} {max(parent_lengths):>6}")
    print(f"  {'父子-子文档(400/64)':<30} {len(child_docs):>8} {sum(child_lengths)/len(child_lengths):>8.0f} {min(child_lengths):>6} {max(child_lengths):>6}")


def main():
    print(f"数据源: {DATA_PATH}")
    print(f"文件存在: {os.path.exists(DATA_PATH)}")

    documents = load_document(DATA_PATH)
    if not documents:
        print("文档加载失败，请检查文件路径")
        return

    print(f"加载文档数: {len(documents)}")
    for doc in documents:
        print(f"  文件: {doc.metadata.get('file_name')}")
        print(f"  类型: {doc.metadata.get('file_type')}")
        print(f"  总字符数: {doc.metadata.get('total_chars')}")

    test_recursive_chunk(documents)
    test_parent_child_chunk(documents)
    test_comparison(documents)

    print(f"\n{'=' * 60}")
    print("  所有测试完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
