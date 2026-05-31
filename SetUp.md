```shell
docker pull docker.1ms.run/library/postgres:14

docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 docker.1ms.run/library/postgres:14
```