"""
Run multi-round rho update experiments with parallel processing.

Usage:
    poetry run python abm/run_rho_update_multiround.py --n_envs 1000 --n_rounds 5 --n_steps 15 --output_dir simulation_output/
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pylab as plt
import seaborn as sns
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

# --- Visual Styling ---
# Define Tol colorblind-friendly palette
TOL_BRIGHT = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#228833',  # green
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#BBBBBB',  # grey
]

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

    return conditions, corr_matrix, rho_true

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

# --- Plotting Helpers ---

def _select_sg_agent(df):
    """Helper to select only the relevant agent for plotting."""
    plot_df = df.copy()
    if "model_type" in plot_df.columns:
        sg = plot_df[(plot_df["model_type"] == "SG-ICM") | (plot_df["model_type"] == "SG")].copy()
        if not sg.empty:
            return sg
    if "AgentID" in plot_df.columns:
        min_id = plot_df["AgentID"].min()
        return plot_df[plot_df["AgentID"] == min_id].copy()
    return plot_df

def plot_learning_across_rounds(df, output_path=None):
    """Plot final cumulative reward per round to show improvement."""
    plot_df = _select_sg_agent(df)

    # Get the final step of each round/seed combo
    idx_max_step = plot_df.groupby(["seed", "Round", "condition_label"])["Step"].idxmax()
    summary = plot_df.loc[idx_max_step]

    plt.figure(figsize=(8, 5))
    sns.catplot(
        data=summary,
        x="Round",
        y="cumulative_reward",
        hue="condition_label",
        kind="point",
        dodge=True,
        legend=True,
        palette=TOL_BRIGHT
    )
    plt.title("Performance Improvement Across Rounds")
    plt.ylabel("Final Cumulative Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rho_longitudinal(all_rho_histories, rho_true, n_rounds, target_condition="landmarks_corr", output_path=None):
    """
    Visualizes how the rho estimate evolves continuously across rounds (stitched).
    """
    # Filter histories for the specific condition
    relevant_histories = [
        h for h in all_rho_histories
        if h["condition"] == target_condition
    ]

    if not relevant_histories:
        print(f"No history found for {target_condition}")
        return

    # Group by seed to stitch
    max_seed = max(h['seed'] for h in relevant_histories)
    stitched_trajectories = []

    for s in range(max_seed + 1):
        # Sort by round
        seed_hists = [h for h in relevant_histories if h["seed"] == s]
        seed_hists.sort(key=lambda x: x["round"])

        arrays = [h["rho_history"] for h in seed_hists]
        if not arrays: continue

        stitched = np.concatenate(arrays, axis=0)
        stitched_trajectories.append(stitched)

    if not stitched_trajectories:
        return

    # Average across seeds
    min_len = min(len(t) for t in stitched_trajectories)
    trimmed_trajectories = [t[:min_len] for t in stitched_trajectories]
    avg_trajectory = np.mean(np.stack(trimmed_trajectories), axis=0)

    true_rhos = rho_true[1:] # peers only

    plt.figure(figsize=(10, 5))
    steps = np.arange(len(avg_trajectory))

    colors = ['r', 'g', 'b']
    for i in range(3): # 3 peers
        plt.plot(steps, avg_trajectory[:, i+1], label=f'Est Rho {i+1}', color=colors[i], linewidth=2)
        plt.axhline(true_rhos[i], linestyle='--', color=colors[i], alpha=0.3, label=f'True Rho {i+1}')

    # Draw vertical lines for round boundaries
    steps_per_round = len(avg_trajectory) / n_rounds
    for r in range(1, n_rounds):
        plt.axvline(r * steps_per_round, color='k', linestyle=':', alpha=0.5)

    plt.title(f"Continuous Rho Estimation across {n_rounds} Rounds ({target_condition})")
    plt.xlabel("Total Steps")
    plt.ylabel("Rho Value")

    # De-duplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_round_start_mae(all_rho_histories, rho_true, target_condition="landmarks_corr", output_path=None):
    """Plots the Mean Absolute Error of the rho estimate at the very first step of each round."""
    relevant_histories = [
        h for h in all_rho_histories
        if h["condition"] == target_condition
    ]

    data = []
    for h in relevant_histories:
        start_rho = h["rho_history"][0, 1:] # Peers only
        true_r = rho_true[1:]
        mae = np.mean(np.abs(start_rho - true_r))
        data.append({
            "Round": h["round"],
            "Start_MAE": mae
        })

    df_mae = pd.DataFrame(data)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_mae, x="Round", y="Start_MAE", errorbar="ci", palette="Blues_d")
    plt.title(f"Initial Error at Start of Each Round ({target_condition})")
    plt.ylabel("MAE of Rho Estimate (Step 0)")
    plt.xlabel("Round")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_longitudinal_metric(df, y_col="reward", hue="condition_label", output_path=None):
    """
    Plots a metric across all rounds stitched together (Global Step).
    """
    plot_df = _select_sg_agent(df).copy()

    steps_per_round = plot_df["Step"].max() + 1
    plot_df["GlobalStep"] = plot_df["Round"] * steps_per_round + plot_df["Step"]

    plt.figure(figsize=(12, 4))
    sns.lineplot(
        data=plot_df,
        x="GlobalStep",
        y=y_col,
        hue=hue,
        errorbar="ci",
        palette=TOL_BRIGHT
    )

    # Add vertical lines for round boundaries
    max_round = plot_df["Round"].max()
    for r in range(1, max_round + 1):
        plt.axvline(r * steps_per_round, color='k', linestyle=':', alpha=0.3)

    plt.title(f"{y_col.replace('_', ' ').title()} over all rounds (Continuous)")
    plt.xlabel("Global Step (across all rounds)")
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots(results, rho_histories, rho_true, n_rounds, output_dir, extension='png'):
    """Generate and save all plots."""

    # 1. Performance Improvement across rounds
    plot_learning_across_rounds(
        results,
        output_path=os.path.join(output_dir, f'learning_across_rounds.{extension}')
    )

    # 2. Continuous Metric Plots
    plot_longitudinal_metric(
        results, y_col="reward",
        output_path=os.path.join(output_dir, f'longitudinal_reward.{extension}')
    )
    plot_longitudinal_metric(
        results, y_col="cumulative_reward",
        output_path=os.path.join(output_dir, f'longitudinal_cumulative_reward.{extension}')
    )

    # 3. Rho specific plots (for the learning condition)
    # We assume 'landmarks_corr' is the learning condition key
    plot_rho_longitudinal(
        rho_histories, rho_true, n_rounds, target_condition="landmarks_corr",
        output_path=os.path.join(output_dir, f'rho_longitudinal_landmarks_corr.{extension}')
    )

    plot_round_start_mae(
        rho_histories, rho_true, target_condition="landmarks_corr",
        output_path=os.path.join(output_dir, f'rho_start_mae_landmarks_corr.{extension}')
    )

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
    conditions, corr_matrix, rho_true = setup_conditions()

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

    # Generate plots
    print("Generating plots...")
    generate_all_plots(results, rho_histories, rho_true, args.n_rounds, args.output_dir, extension='png')
    generate_all_plots(results, rho_histories, rho_true, args.n_rounds, args.output_dir, extension='svg')

    print("Done.")

if __name__ == "__main__":
    main()