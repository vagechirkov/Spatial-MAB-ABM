"""
Run rho update experiments with parallel processing.

Usage:
    poetry run python abm/run_rho_update.py --n_envs 10000 --n_steps 30 --output_dir results/
    # test
    poetry run python abm/run_rho_update.py --n_envs 50 --n_steps 30 
"""

import argparse
import numpy as np
import pandas as pd
import os
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm

from model import SocialGPModel
from rewards import (
    build_corr_matrix_option1,
    generate_env,
)


def setup_conditions():
    """Define experimental conditions."""
    corr_matrix = build_corr_matrix_option1()
    rho_true = corr_matrix[0]
    rho_sign_flipped = np.array([1.0, -0.6, 0.0, 0.6])
    rho_zero = np.array([1.0, 0.0, 0.0, 0.0])

    conditions = {
        "perfectly_calibrated": {
            "label": "SG-ICM\ncalibrated",
            "params": {
                "rho_update_rule": None,
                "rho": rho_true,
                "model_type": "SG-ICM",
            }
        },
        "sign_flipped": {
            "label": "SG-ICM\nflipped",
            "params": {
                "rho_update_rule": None,
                "rho": rho_sign_flipped,
                "model_type": "SG-ICM",
            }
        },
        "asocial": {
            "label": "AS",
            "params": {
                "rho_update_rule": None,
                "rho": rho_zero,
                "model_type": "SG-ICM",
            }
        },
        "landmarks_corr": {
            "label": "SG-ICM\nlearning rho",
            "params": {
                "rho_update_rule": "landmarks_corr_2",
                "rho_update_kwargs": {"rho_lr": 0.1},
                "rho": rho_zero,
                "model_type": "SG-ICM",
            }
        },
        "sg_calibrated": {
            "label": "SG calibrated\n(0.01, 20, 20)",
            "params": {
                "model_type": "SG",
                "rho_update_rule": None,
                "observation_noise_social": [0.01, 20, 20],
                "rho": rho_zero,
            }
        },
        "sg_fixed": {
            "label": "SG fixed\n(3.0, 3.0, 3.0)",
            "params": {
                "model_type": "SG",
                "rho_update_rule": None,
                "observation_noise_social": [3.0, 3.0, 3.0],
                "rho": rho_zero,
            }
        },
        "sg_flipped": {
            "label": "SG flipped\n(20 0.01 0.01)",
            "params": {
                "model_type": "SG",
                "rho_update_rule": None,
                "observation_noise_social": [20, 0.01, 0.01],
                "rho": rho_zero,
            }
        }
    }

    return conditions, corr_matrix


def get_agent0_rho(model):
    """Extract rho values from agent 0."""
    agent = model.grid.agents[0]
    return np.array(agent.rho, dtype=float)


def run_condition(condition_name, condition_cfg, parent, child_maps, n_children, n_steps=15, seed=0):
    """Run a single condition simulation."""
    m = SocialGPModel(
        n=n_children + 1,
        child_maps=[parent] + child_maps,
        network_type="directed_one_to_four",
        tau_sampling=True,
        beta_sampling=True,
        length_scale_sampling=True,
        observation_noise_private=0.1,
        seed=seed,
        **condition_cfg["params"]
    )

    rho_history = []
    for _ in range(n_steps):
        m.step()
        rho_history.append(get_agent0_rho(m))

    df = m.datacollector.get_agent_vars_dataframe().reset_index()
    df["condition"] = condition_name
    df["condition_label"] = condition_cfg["label"]
    df["seed"] = seed
    return df, np.array(rho_history)


def process_one_seed(env_seed, conditions, corr_matrix, n_children, grid_size, n_steps):
    """Process simulations for one environment seed."""
    local_results = []
    local_rhos = []

    parent, child_maps = generate_env(env_seed, np.array([1, 0.6, 0.0, -0.6]), corr_matrix, grid_size=grid_size)

    for condition_name, condition_cfg in conditions.items():
        df, rho_hist = run_condition(
            condition_name,
            condition_cfg,
            parent,
            child_maps,
            n_children,
            n_steps=n_steps,
            seed=env_seed,
        )
        local_results.append(df)
        local_rhos.append(
            {
                "condition": condition_name,
                "condition_label": condition_cfg["label"],
                "seed": env_seed,
                "rho_history": rho_hist,
            }
        )
    return local_results, local_rhos


def run_simulations(n_envs, n_steps, conditions, corr_matrix, grid_size=11, n_jobs=-1):
    """Run parallel simulations."""
    n_children = corr_matrix.shape[0] - 1

    parallel_outputs = Parallel(n_jobs=n_jobs)(
        delayed(process_one_seed)(seed, conditions, corr_matrix, n_children, grid_size, n_steps)
        for seed in tqdm(range(n_envs))
    )

    all_results = []
    rho_histories = []

    for seed_dfs, seed_rhos in parallel_outputs:
        all_results.extend(seed_dfs)
        rho_histories.extend(seed_rhos)

    results = pd.concat(all_results, ignore_index=True)

    return results, rho_histories


def save_results(results, rho_histories, output_dir):
    """Save results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save main results dataframe
    results_path = os.path.join(output_dir, 'results.csv')
    results.to_csv(results_path, index=False)

    # Save rho histories
    rho_path = os.path.join(output_dir, 'rho_histories.npy')
    np.save(rho_path, rho_histories, allow_pickle=True)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run rho update experiments')
    parser.add_argument('--n_envs', type=int, default=10000,
                       help='Number of environments to simulate')
    parser.add_argument('--n_steps', type=int, default=30,
                       help='Number of steps per simulation')
    parser.add_argument('--grid_size', type=int, default=11,
                       help='Grid size for environments')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: results_TIMESTAMP)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results_{timestamp}'

    # Setup conditions
    conditions, corr_matrix = setup_conditions()

    # Run simulations
    results, rho_histories = run_simulations(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        conditions=conditions,
        corr_matrix=corr_matrix,
        grid_size=args.grid_size,
        n_jobs=args.n_jobs
    )

    # Save results
    save_results(results, rho_histories, args.output_dir)


if __name__ == "__main__":
    main()
