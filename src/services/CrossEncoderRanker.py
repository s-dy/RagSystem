from utils.environ import set_huggingface_hf_env
set_huggingface_hf_env()
from sentence_transformers import CrossEncoder
import numpy as np
# 加载预训练的交叉编码器模型
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
# 查询和检索到的文档
query = "什么是深度学习？"
retrieved_docs = [
   "深度学习是机器学习的一个分支，主要使用神经网络进行特征学习",
   "机器学习包含监督学习、无监督学习和强化学习",
   "神经网络由多个层次组成，包括输入层、隐藏层和输出层",
   "深度学习在计算机视觉和自然语言处理中有广泛应用"
]
# 生成查询-文档对
pairs = [(query, doc) for doc in retrieved_docs]
# 预测相关性分数
scores = model.predict(pairs)
# 按分数降序排序
reranked_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
# 输出重排序结果
print("重排序后结果:")
for i, (doc, score) in enumerate(reranked_docs):
   print(f"{i+1}. [Score: {score:.4f}] {doc}")