# minimind

<img align="right" src="assets/minimind.png">

ML Research 🪼 ❤️ 🌊

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
docker compose run minimind bash
```
