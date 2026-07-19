# Trust Issues: Social Learning Under Misaligned Goals

This repository contains the agent-based simulation code and figure-generation scripts for the CogSci 2026 paper "Trust Issues: Social Learning Under Misaligned Goals".

## Installation

The project uses Python 3.12 and Poetry. From the repository root:

```bash
poetry config virtualenvs.in-project true
poetry install
```

Run all commands below from the repository root with `poetry run`.

## Reproducing the Simulation Results

The full simulation runs are computationally expensive. The camera-ready results were produced with 50,000 environments; each major simulation step can take several hours on a machine with 64 CPU cores.

Single-round simulations:

```bash
poetry run python abm/run_rho_update.py \
  --n_envs 50000 \
  --n_steps 30 \
  --output_dir results_single_round
```

Multiround simulations:

```bash
poetry run python abm/run_rho_update_multiround.py \
  --n_envs 50000 \
  --n_rounds 7 \
  --n_steps 15 \
  --output_dir results_multiround
```

Sensitivity analysis:

```bash
poetry run python abm/run_sensitivity_heatmap.py \
  --n_seeds 5000 \
  --n_steps 30 \
  --output_dir results_sensitivity
```

For quick smoke tests, reduce `--n_envs` or `--n_seeds`, for example:

```bash
poetry run python abm/run_rho_update.py --n_envs 50 --n_steps 30 --output_dir results_test_single
poetry run python abm/run_rho_update_multiround.py --n_envs 50 --n_rounds 2 --n_steps 15 --output_dir results_test_multiround
poetry run python abm/run_sensitivity_heatmap.py --n_seeds 10 --n_steps 30 --output_dir results_test_sensitivity
```

## Generating Figures

After simulation outputs are available, update the input paths in `abm/figures_results.py` if needed and run:

```bash
poetry run python abm/figures_results.py
```

The script writes final figure files to `final_figures_revision_04_2026/`.

The model-illustration panel extracted from the old notebook can be regenerated with:

```bash
poetry run python abm/figure_1d_model_illustration.py \
  --output_dir figures/model_illustration
```

This writes `scale_model_illustration.png`, `scale_model_illustration.svg`, and `scale_model_illustration.pdf`.

The sensitivity-analysis script writes a heatmap figure named `sensitivity_heatmap.png` in its output directory:

```bash
poetry run python abm/run_sensitivity_heatmap.py \
  --n_seeds 5000 \
  --n_steps 30 \
  --output_dir figures/sensitivity_analysis
```

If the final camera-ready sensitivity figure is generated on the cluster, commit or upload that final `sensitivity_heatmap.png`/`.svg` alongside the code or include it as a separate Zenodo file. Do not use quick-test outputs for archival figures.

## Data Files

The Zenodo version 2 record should include the code archive and the large simulation result archives:

- `results_one_round_20260129_212526.zip`
- `results_multiround_20260123_225704.zip`

Each archive contains a `results.csv` file and a `rho_histories.npy` file. Download and unzip these archives before running the figure-generation script.

## Citation

Please cite both the paper and the archived code/data release.

Paper citation:

```text
Lee, L., Chirkov, V., Tian, S., Wu, C. M., & Witt, A. (2026).
Trust Issues: Social Learning Under Misaligned Goals.
In Proceedings of the 48th Annual Conference of the Cognitive Science Society.
```

Code/data citation:

```text
Lee, L., Chirkov, V., Tian, S., Wu, C. M., & Witt, A. (2026).
Trust Issues: Social Learning Under Misaligned Goals (Version 2) [Code and data].
Zenodo. https://doi.org/10.5281/zenodo.20053031
```

The DOI above is the Zenodo concept DOI, which resolves to the latest version of the archived record. For exact reproducibility, replace it with the version-specific DOI for the final Zenodo v2 record after that version is published.

## Authors

- Liang Lee, Hessian AI, Darmstadt, Germany, ORCID: https://orcid.org/0009-0003-4421-1811
- Valerii Chirkov, Humboldt-Universitat zu Berlin, ORCID: https://orcid.org/0000-0003-3950-898X
- Shen Tian, Karolinska Institutet, ORCID: https://orcid.org/0009-0003-3473-3093
- Charley Wu, Technical University Darmstadt, Darmstadt, Germany, ORCID: https://orcid.org/0000-0002-2215-572X
- Alexandra Witt, RIKEN Center for Brain Science, RIKEN, Wako, Japan, ORCID: https://orcid.org/0009-0000-3537-5249

## Funding

We thank the Computational Summer School on Modeling Social and Collective Behaviour (COSMOS), supported by the RIKEN CBS-Toyota Collaboration Center (RIKEN BTCC), where the initial ideas for this project were developed.

LL and CMW are supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (C4: 101164709), the Hessian research funding programme LOEWE/4b//519/05/01.002(0022)/119, the Deutsche Forschungsgemeinschaft (German Research Foundation, DFG) under Germany's Excellence Strategy (EXC 3066/1 "The Adaptive Mind", Project No. 533717223), and the Excellence Cluster "Reasonable AI" by the Deutsche Forschungsgemeinschaft (German Research Foundation, DFG) under Germany's Excellence Strategy - EXC-3057.

VC was supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2002/1 "Science of Intelligence".

## License

This repository is released under the MIT License. See `LICENSE`.
