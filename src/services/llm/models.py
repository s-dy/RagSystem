from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from src.monitoring.logger import monitor_task_status

load_dotenv()


def get_qwen_model(**kwargs):
    return init_chat_model(
        os.getenv('QWEN_MODEL_NAME'),
        api_key=os.getenv('QWEN_API_KEY'),
        base_url=os.getenv('QWEN_BASE_URL'),
        model_provider='openai',
        **kwargs
    )

def get_openai_alibaba_model(**kwargs):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv('LAB_API_KEY'),
        base_url=os.getenv('LAB_BASE_URL'),
    )
    return client

def get_embedding_model(model,b=0.6,**kwargs):
    from langchain.embeddings import init_embeddings
    if model == 'nomic':
        model_name = 'ollama:nomic-embed-text'
    elif model == 'qwen':
        model_name = f'ollama:qwen3-embedding:{b}B'
    else:
        raise ValueError('model not supported')

    embeddings = init_embeddings(
        model=model_name,
    )
    return embeddings

def get_ollama_deepseek_model(b=8,**kwargs):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=f'deepseek-r1:{b}b')

if __name__ == '__main__':
    # model = get_ollama_deepseek_model()
    # print(model.invoke('hello'))
    # get_embedding_model()
    embeddings = get_embedding_model('qwen')
    monitor_task_status(embeddings.embed_query('hello world'))