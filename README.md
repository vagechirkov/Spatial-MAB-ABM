


## Installation instructions

```bash
# source ~/.bashrc
pyenv install 3.12
pyenv local 3.12
pyenv version
pyenv which python

# need to have poetry installed
poetry init --python "^3.12" -q  # skip this if poetry.lock already exists
poetry env use $(pyenv which python)
which python
poetry run which python
```

Add poetry dependencies

```bash
poetry add "mesa[all]"
poetry add "wandb[media,sweeps]"
poetry add --group dev pytest ruff
```

If import issue occurs

```bash
poetry run pip uninstall sbi pymc pytensor numpy -y
poetry update
```

Install dependencies (if the project already exists)

```bash
poetry install
```
