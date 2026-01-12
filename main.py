import os

from src.graph import Graph
from utils.async_task import async_run

os.environ['IS_LANGSMITH'] = 'True'

graph = async_run(Graph().graph)
