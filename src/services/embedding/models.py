from utils.environ import set_huggingface_hf_env
set_huggingface_hf_env()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

sentences = [
    "贵州男子王铭"
]
embeddings = model.encode(sentences)
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# [3, 3]