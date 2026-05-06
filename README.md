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
poetry config virtualenvs.in-project true  # <- install venvirtural environment to .venv folder
poetry install
source .venv/bin/activate
```


## Code execution

```bash
# 50000 envs takes 2-3 hours on 64 cores
poetry run python abm/run_rho_update.py --n_envs 50000 --output_dir results_single_round
# 50000 envs takes 2-3 hours on 64 cores
poetry run python abm/run_rho_update_multiround.py --n_envs 50000 --output_dir results_multiround

# figures are produced in figures/ folder
poerty run python abm/figures_results.py

# 50000 envs takes 2-3 hours on 64 cores
poetry run python abm/run_sensitivity_heapmap.py --output_dir figures
```
