


## Installation instructions

```bash

pyenv install 3.12
pyenv local 3.12
pyenv version
pyenv which python

# need to have poetry installed
poetry init --python "^3.12" -q
poetry env use $(pyenv which python)
which python
```

Add poetry dependencies

```bash
poetry add "mesa[all]"
poetry add "wandb[media,sweeps]"
poetry add --group dev pytest ruff
```