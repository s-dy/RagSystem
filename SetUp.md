- Postgres
```shell
docker pull docker.1ms.run/library/postgres:14

docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 docker.1ms.run/library/postgres:14
```

- Ollama
```shell
docker pull docker.1ms.run/ollama/ollama:latest

docker run -d --name ollama -p 11434:11434 docker.1ms.run/ollama/ollama:latest

# 安装embedding模型
docker exec -it ollama bash -c "ollama pull qwen3-embedding:0.6b"
```

- Redis
```shell
docker pull docker.1ms.run/bitnami/redis:latest

docker run -d --name redis -p 6379:6379 docker.1ms.run/bitnami/redis:latest
```
