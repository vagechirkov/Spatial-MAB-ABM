"""
Run multi-round rho update experiments with parallel processing.

Usage:
    poetry run python abm/run_rho_update_multiround.py --n_envs 1000 --n_rounds 5 --n_steps 15 --output_dir simulation_output/
"""

import argparse
import numpy as np
import pandas as pd
import os
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm

# Assumes model and rewards are available in the python path
from model import SocialGPModel
from rewards import (
    build_corr_matrix_option1,
    generate_env,
)

def setup_conditions():
    """
    Define experimental conditions.
    Distinguishes between conditions that learn continuously (transfer_learning=True)
    and those that reset every round.
    """
    corr_matrix = build_corr_matrix_option1()
    rho_true = corr_matrix[0]
    rho_zero = np.array([1.0, 0.0, 0.0, 0.0])

    conditions = {
        "perfectly_calibrated": {
            "label": "SG-ICM\ncalibrated",
            "params": {
                "rho_update_rule": None,
                "rho": rho_true,
                "model_type": "SG-ICM",
            },
            "transfer_learning": False
        },
        "asocial": {
            "label": "AS",
            "params": {
                "rho_update_rule": None,
                "rho": rho_zero,
                "model_type": "SG-ICM",
            },
            "transfer_learning": False
        },
        "landmarks_corr": {
            "label": "SG-ICM\nlearning rho",
            "params": {
                "rho_update_rule": "landmarks_corr_2",
                "rho": rho_zero,
                "model_type": "SG-ICM",
                "rho_update_kwargs": {"rho_lr": 0.05},
            },
            "transfer_learning": True # Pass learned rho to next round
        },
    }

    return conditions, corr_matrix

def get_agent0_rho(model):
    """Extracts the current rho estimate from the first agent."""
    agent = model.grid.agents[0]
    # Ensure we get a numpy array copy, not a reference
    return np.array(agent.rho, dtype=float)

def run_condition(condition_name, condition_cfg, parent, child_maps, n_children, n_steps=15, seed=0, override_rho=None):
    """
    Runs a single simulation.
    If override_rho is provided, it replaces the 'rho' in params (used for transfer learning).
    """
    params = condition_cfg["params"].copy()

    # If we are transferring knowledge, overwrite the starting rho
    if override_rho is not None:
        params["rho"] = override_rho

    m = SocialGPModel(
        n=n_children + 1,
        child_maps=[parent] + child_maps,
        network_type="directed_one_to_four",
        tau_sampling=True,
        beta_sampling=True,
        length_scale_sampling=True,
        observation_noise_private=0.1,
        seed=seed,
        **params
    )

    rho_history = []

    # Initial State
    rho_history.append(get_agent0_rho(m))

    for _ in range(n_steps):
        m.step()
        rho_history.append(get_agent0_rho(m))

    df = m.datacollector.get_agent_vars_dataframe().reset_index()
    df["condition"] = condition_name
    df["condition_label"] = condition_cfg["label"]
    df["seed"] = seed

    final_rho = rho_history[-1]

    return df, np.array(rho_history), final_rho

def process_multiround_seed(base_seed, n_rounds, n_steps, conditions, corr_matrix, grid_size):
    """
    Runs a chain of experiments for a single base seed.
    1. Generates n_rounds environments.
    2. Runs all conditions.
    3. For conditions with 'transfer_learning': True, passes the final rho
       from Round R to Round R+1.
    """
    local_results = []
    local_rhos = []
    n_children = corr_matrix.shape[0] - 1

    # Store the rho to be used for the *next* round for each condition
    # Initialize with None (so Round 0 uses config defaults)
    next_round_priors = {k: None for k in conditions.keys()}

    for r in range(n_rounds):
        # Generate a distinct environment for this round
        # Deterministic hash on seed+round ensures reproducibility
        current_env_seed = base_seed * 1000 + r

        parent, child_maps = generate_env(
            current_env_seed,
            np.array([1, 0.6, 0.0, -0.6]),
            corr_matrix,
            grid_size=grid_size
        )

        for condition_name, condition_cfg in conditions.items():
            # Determine input rho: Use carried-over value if available and allowed
            rho_input = None
            if condition_cfg.get("transfer_learning", False):
                rho_input = next_round_priors[condition_name]

            # Run Simulation
            df, rho_hist, final_rho = run_condition(
                condition_name,
                condition_cfg,
                parent,
                child_maps,
                n_children=n_children,
                n_steps=n_steps,
                seed=current_env_seed,
                override_rho=rho_input
            )

            # Add Round metadata
            df["Round"] = r

            # Store data
            local_results.append(df)
            local_rhos.append({
                "condition": condition_name,
                "condition_label": condition_cfg["label"],
                "seed": base_seed,  # Track by the base chain seed
                "round": r,
                "env_seed": current_env_seed,
                "rho_history": rho_hist
            })

            # Update prior for next round
            next_round_priors[condition_name] = final_rho

    return local_results, local_rhos

def run_simulations(n_envs, n_rounds, n_steps, conditions, corr_matrix, grid_size=11, n_jobs=-1):
    """Run parallel simulations across multiple seeds and rounds."""

    parallel_outputs = Parallel(n_jobs=n_jobs)(
        delayed(process_multiround_seed)(
            seed, n_rounds, n_steps, conditions, corr_matrix, grid_size
        ) for seed in tqdm(range(n_envs))
    )

    # Unpack results
    all_dfs = []
    all_rho_histories = []

    for seed_dfs, seed_rhos in parallel_outputs:
        all_dfs.extend(seed_dfs)
        all_rho_histories.extend(seed_rhos)

    results = pd.concat(all_dfs, ignore_index=True)

    return results, all_rho_histories

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run Transfer Learning Rho Experiments')
    parser.add_argument('--n_envs', type=int, default=1000,
                        help='Number of independent seeds (chains)')
    parser.add_argument('--n_rounds', type=int, default=7,
                        help='Number of sequential rounds per seed')
    parser.add_argument('--n_steps', type=int, default=15,
                        help='Number of steps per round')
    parser.add_argument('--grid_size', type=int, default=11,
                        help='Grid size for environments')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Global random seed')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results_multiround_{timestamp}'

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Running experiments... Output will be saved to: {args.output_dir}")

    # Setup conditions
    conditions, corr_matrix = setup_conditions()

    # Run simulations
    results, rho_histories = run_simulations(
        n_envs=args.n_envs,
        n_rounds=args.n_rounds,
        n_steps=args.n_steps,
        conditions=conditions,
        corr_matrix=corr_matrix,
        grid_size=args.grid_size,
        n_jobs=args.n_jobs
    )

    # Save raw data
    print("Saving raw data...")
    results.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)
    np.save(os.path.join(args.output_dir, 'rho_histories.npy'), rho_histories, allow_pickle=True)

    print("Done.")

if __name__ == "__main__":
    main()
