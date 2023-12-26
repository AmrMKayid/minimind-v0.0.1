# minimind

<img align="right" src="assets/minimind.png">

Reinforcement Learning and Brain Research in JAX 🧠

## Environment setup

```bash
conda create -n minimind python=3.10.11 -y
conda activate minimind
pip install poetry

poetry env use $(which python)

poetry install
```

## Entry point

```shell
poetry run minimind --config-path configs/default.yaml
```


## Docker

```shell

# For tracking history
touch ~/miniminddockerinputrc
touch ~/miniminddockerhistory

docker compose run minimind bash
```
