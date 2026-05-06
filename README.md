## Installation instructions

```bash
# NOTE: you need to have Python 3.12 and poetry installed
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
