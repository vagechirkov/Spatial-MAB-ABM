"""
Sensitivity heatmap: SCALE learning rate × observation noise.

Computes mean(SCALE cumulative reward - AS cumulative reward) across seeds
for each (rho_lr, observation_noise_private) grid cell.

Usage:
    poetry run python abm/run_sensitivity_heatmap.py --n_seeds 5000
    poetry run python abm/run_sensitivity_heatmap.py --n_seeds 100          # quick test
    poetry run python abm/run_sensitivity_heatmap.py --n_seeds 5000 --n_steps 30 --output_dir results/heatmap_5k

Note on numerical warnings:
    Grid cells with observation_noise_private = 0.01 (log10 = -2, bottom row) will
    trigger LinAlgError / UserWarning from sklearn. This is expected: at that noise
    level the ICM kernel matrix becomes ill-conditioned after ~80 accumulated
    observations (Cholesky fails). Values in that row should be treated as
    unreliable. This is mechanistically why the paper fixes sigma^2 = 0.1 for
    stability. Cells with noise >= 0.032 are unaffected.
"""

import argparse
import os

# Disable multi-threading in BLAS/MKL to prevent oversubscription in worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from model import SocialGPModel
from rewards import build_corr_matrix_option1, generate_env


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORR_MATRIX = build_corr_matrix_option1()
N_CHILDREN = CORR_MATRIX.shape[0] - 1
GRID_SIZE = 11
RHO_ZERO = np.array([1.0, 0.0, 0.0, 0.0])

RHO_LR_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]
PRIV_NOISE_GRID = list(np.logspace(-2, 0, 5))   # [0.01, 0.032, 0.1, 0.316, 1.0]

# Index of the default parameter values in each grid (for red outline on plot)
DEFAULT_LR_IDX = RHO_LR_GRID.index(0.05)                                  # col 2
DEFAULT_NOISE_IDX = min(range(len(PRIV_NOISE_GRID)),
                        key=lambda i: abs(PRIV_NOISE_GRID[i] - 0.1))      # row 2


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _run_one_seed(env_seed, rho_lr_val, priv_noise_val, n_steps):
    _parent, _cmaps = generate_env(
        env_seed,
        np.array([1, 0.6, 0.0, -0.6]),
        CORR_MATRIX,
        grid_size=GRID_SIZE,
    )
    common = dict(
        n=N_CHILDREN + 1,
        child_maps=[_parent] + _cmaps,
        network_type="directed_one_to_four",
        tau_sampling=True,
        beta_sampling=True,
        length_scale_sampling=True,
        observation_noise_private=priv_noise_val,
        rho_update_sigma_zeta=0.1,
        seed=env_seed,
    )

    m_scale = SocialGPModel(
        **common,
        model_type="SG-ICM",
        rho_update_rule="landmarks_corr",
        rho_update_kwargs={"rho_lr": rho_lr_val},
        rho=RHO_ZERO.copy(),
        rho_update_init=0.0,
        rho_update_basis="full",
    )
    for _ in range(n_steps):
        m_scale.step()
    df = m_scale.datacollector.get_agent_vars_dataframe().reset_index()
    focal = df[df["AgentID"] == df["AgentID"].min()]
    scale_rew = focal.loc[focal["Step"] == focal["Step"].max(), "cumulative_reward"].values[0]

    m_as = SocialGPModel(
        **common,
        model_type="SG-ICM",
        rho_update_rule=None,
        rho=RHO_ZERO.copy(),
    )
    for _ in range(n_steps):
        m_as.step()
    df = m_as.datacollector.get_agent_vars_dataframe().reset_index()
    focal = df[df["AgentID"] == df["AgentID"].min()]
    as_rew = focal.loc[focal["Step"] == focal["Step"].max(), "cumulative_reward"].values[0]

    return scale_rew - as_rew


def run_heatmap(n_envs, n_steps):
    """
    Run the sensitivity heatmap simulations in parallel.
    Uses process-based parallelization to avoid GIL and BLAS contention issues.
    """
    # Flatten the grid and seeds into a single list of tasks
    tasks = []
    for i, pn in enumerate(PRIV_NOISE_GRID):
        for j, lr in enumerate(RHO_LR_GRID):
            for s in range(n_envs):
                tasks.append((s, lr, pn, i, j))

    print(f"Starting {len(tasks)} simulations using {os.cpu_count()} processes...")

    # Run in parallel using processes
    results_flat = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_run_one_seed)(s, lr, pn, n_steps)
        for s, lr, pn, i, j in tqdm(tasks, desc="Simulations")
    )

    # Reconstruct the heatmap data
    heatmap_data = np.zeros((len(PRIV_NOISE_GRID), len(RHO_LR_GRID)))
    # results_flat[idx] corresponds to tasks[idx]
    # We need to average over n_envs for each (i, j)
    
    # Simple reshaping if we want to be more efficient
    results_reshaped = np.array(results_flat).reshape(
        len(PRIV_NOISE_GRID), len(RHO_LR_GRID), n_envs
    )
    heatmap_data = np.mean(results_reshaped, axis=2)
    
    return heatmap_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(heatmap_data, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        heatmap_data,
        xticklabels=[f"{v:.2f}" for v in RHO_LR_GRID],
        yticklabels=[
            f"{np.log10(v):.1g}  ({v:.3f})" for v in PRIV_NOISE_GRID
        ],
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        cbar_kws={"label": "SCALE − AS (final cumulative reward)"},
    )
    ax.add_patch(plt.Rectangle(
        (DEFAULT_LR_IDX, DEFAULT_NOISE_IDX), 1, 1,
        fill=False, edgecolor="red", lw=2.5, zorder=5,
    ))
    ax.set_xlabel("Learning rate (α)")
    ax.set_ylabel(r"Observation noise, $\log_{10}(\sigma^2_\varepsilon)$")
    ax.set_title("SCALE vs AS baseline (final cumulative reward)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sensitivity heatmap: SCALE vs AS")
    parser.add_argument("--n_seeds", type=int, default=100,
                        help="Number of environment seeds per grid cell (default: 100)")
    parser.add_argument("--n_steps", type=int, default=30,
                        help="Steps per simulation (default: 30)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results_heatmap_TIMESTAMP)")
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_heatmap_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Grid: {len(PRIV_NOISE_GRID)}×{len(RHO_LR_GRID)}, "
          f"n_seeds={args.n_seeds}, n_steps={args.n_steps}")
    print(f"rho_lr_grid:     {RHO_LR_GRID}")
    print(f"priv_noise_grid: {[round(v, 4) for v in PRIV_NOISE_GRID]}")

    heatmap_data = run_heatmap(args.n_seeds, args.n_steps)

    np.save(os.path.join(args.output_dir, "heatmap_data.npy"), heatmap_data)
    print(f"Saved: {os.path.join(args.output_dir, 'heatmap_data.npy')}")

    plot_heatmap(
        heatmap_data,
        output_path=os.path.join(args.output_dir, "sensitivity_heatmap.png"),
    )


if __name__ == "__main__":
    main()
