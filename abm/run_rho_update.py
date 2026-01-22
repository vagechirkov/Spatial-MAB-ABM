"""
Run rho update experiments with parallel processing.

Usage:
    poetry run python abm/run_rho_update.py --n_envs 10000 --n_steps 30 --output_dir results/
    # test
    poetry run python abm/run_rho_update.py --n_envs 50 --n_steps 30 
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

from model import SocialGPModel
from rewards import (
    make_parent_and_children_cholesky2,
    build_corr_matrix_option1,
    _min_max,
    generate_env,
)

# Define Tol colorblind-friendly palette
tol_bright = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#228833',  # green
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#BBBBBB',  # grey
]

tol_muted = [
    '#332288',  # indigo
    '#88CCEE',  # cyan
    '#44AA99',  # teal
    '#117733',  # green
    '#999933',  # olive
    '#DDCC77',  # sand
    '#CC6677',  # rose
    '#882255',  # wine
    '#AA4499',  # purple
]


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
                "rho_update_rule": "landmarks_corr",
                "rho_update_kwargs": {"rho_lr": 0.05},
                "rho": rho_zero,
                "model_type": "SG-ICM",
                "rho_update_init": 0.0,
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

    update_conditions = ["landmarks_corr"]

    return conditions, update_conditions, corr_matrix, rho_true


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
        rho_update_sigma_zeta=0.1,
        rho_update_basis="full",
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


def _select_sg_agent(df):
    """Select SG-ICM agent data."""
    plot_df = df.copy()
    if "model_type" in plot_df.columns:
        sg = plot_df[(plot_df["model_type"] == "SG-ICM") | (plot_df["model_type"] == "SG")].copy()
        if not sg.empty:
            return sg
    if "AgentID" in plot_df.columns:
        min_id = plot_df["AgentID"].min()
        return plot_df[plot_df["AgentID"] == min_id].copy()
    return plot_df


def build_condition_palette(df):
    """Build a colorblind-friendly palette for all conditions using Tol colors."""
    plot_df = _select_sg_agent(df)
    condition_labels = sorted(plot_df["condition_label"].unique())

    palette = {
        "AS": tol_bright[3],
        "SG-ICM\ncalibrated": tol_bright[0],
        "SG-ICM\nflipped": tol_bright[1],
        "SG-ICM\nlearning rho": tol_bright[2],
        "SG calibrated\n(0.01, 20, 20)": tol_bright[4],
        "SG flipped\n(20 0.01 0.01)": tol_bright[5],
    }

    assigned = set(palette.keys())
    remaining = [lbl for lbl in condition_labels if lbl not in assigned]
    for i, lbl in enumerate(remaining):
        palette[lbl] = tol_muted[i % len(tol_muted)]

    return palette


def plot_reward(df, y_col="reward", output_path=None, figsize=(9, 5)):
    """Plot reward over time."""
    plot_df = _select_sg_agent(df)
    plot_df = plot_df.dropna(subset=["Step", y_col, "condition_label"])

    hue_order = sorted(plot_df["condition_label"].unique())
    palette = build_condition_palette(plot_df)

    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=plot_df,
        x="Step",
        y=y_col,
        hue="condition_label",
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title(f"{y_col.replace('_', ' ').title()} for SG-ICM agent")
    ax.set_xlabel("Step")
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.legend(
        title="Condition",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_final_cumreward(df, output_path=None):
    """Plot final cumulative reward."""
    plot_df = _select_sg_agent(df)

    x = 'condition_label'
    group_cols = ["seed", x]
    final_step = plot_df.groupby(group_cols)["Step"].transform('max')
    last = plot_df[plot_df['Step'] == final_step].copy()
    last = last.dropna(subset=[x, 'cumulative_reward'])

    order = sorted(last[x].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(
        data=last,
        x=x,
        y='cumulative_reward',
        order=order,
        errorbar=('ci', 95),
        dodge=False,
        color='C0',
        linestyles='',
        markers='o',
        ax=ax,
    )

    as_mean = last.loc[last[x] == "AS", "cumulative_reward"].mean()
    flipped_mean = last.loc[last[x] == "SG-ICM\nflipped", "cumulative_reward"].mean()

    if pd.notna(as_mean):
        ax.axhline(y=as_mean, color='gray', linestyle='--', label='AS Mean', alpha=0.7)

    if pd.notna(flipped_mean):
        ax.axhline(y=flipped_mean, color='gray', linestyle='--', label='SG-ICM flipped Mean', alpha=0.7)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Final cumulative reward')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rho_history(rho_histories, conditions, update_conditions, rho_true, output_path=None):
    """Plot estimated rho over time."""
    def mean_rho_history(condition):
        histories = [
            entry["rho_history"]
            for entry in rho_histories
            if entry["condition"] == condition
        ]
        if not histories:
            return None
        return np.mean(np.stack(histories, axis=0), axis=0)

    true_rho = rho_true[1:]
    peer_colors = ['#228833', '#CCBB44', '#EE6677']  # green, yellow, red

    fig, axes = plt.subplots(1, len(update_conditions), figsize=(6 * len(update_conditions), 4))
    if len(update_conditions) == 1:
        axes = [axes]

    for ax, condition in zip(axes, update_conditions):
        mean_hist = mean_rho_history(condition)
        if mean_hist is None:
            continue

        for idx, (rho_t, color) in enumerate(zip(true_rho, peer_colors)):
            ax.plot(mean_hist[:, idx + 1], label=f"peer {idx+1} (true {rho_t:+.2f})",
                    color=color, linewidth=2)
            ax.axhline(rho_t, linestyle='--', color=color, alpha=0.3, linewidth=1.5)

        ax.set_title(f"Estimated rho over time ({conditions[condition]['label']})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Estimated rho")
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rho_mae(rho_histories, conditions, update_conditions, rho_true, output_path=None):
    """Plot mean absolute error of rho estimates."""
    def mean_rho_history(condition):
        histories = [
            entry["rho_history"]
            for entry in rho_histories
            if entry["condition"] == condition
        ]
        if not histories:
            return None
        return np.mean(np.stack(histories, axis=0), axis=0)

    def rho_mae(history):
        est = history[:, 1:]
        return np.mean(np.abs(est - rho_true[1:]), axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, condition in enumerate(update_conditions):
        mean_hist = mean_rho_history(condition)
        if mean_hist is None:
            continue
        color = tol_bright[i % len(tol_bright)]
        ax.plot(rho_mae(mean_hist), label=conditions[condition]["label"],
                color=color, linewidth=2)

    ax.set_title("Mean absolute error of rho estimates")
    ax.set_xlabel("Step")
    ax.set_ylabel("MAE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_exploration_distance(df, output_path=None):
    """Plot exploration distance over time."""
    plot_df = df.copy()
    plot_df = plot_df[plot_df["Step"] >= 2].copy()

    hue_order = sorted(plot_df["condition_label"].unique())
    palette = build_condition_palette(plot_df)

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=plot_df,
        x="Step",
        y="last_choice_distance_private",
        hue="condition_label",
        hue_order=hue_order,
        palette=palette,
        errorbar="ci",
        linewidth=2,
    )
    plt.title("Exploration distance over time (private step-to-step)")
    plt.xlabel("Step")
    plt.ylabel("Last choice distance")
    plt.legend(title="Condition", bbox_to_anchor=(1.02, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results, rho_histories, output_dir):
    """Save results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save main results dataframe
    results_path = os.path.join(output_dir, 'results.csv')
    results.to_csv(results_path, index=False)

    # Save rho histories
    rho_path = os.path.join(output_dir, 'rho_histories.npy')
    np.save(rho_path, rho_histories, allow_pickle=True)

def generate_all_plots(results, rho_histories, conditions, update_conditions, rho_true, output_dir):
    """Generate and save all plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Reward plots
    plot_reward(results, y_col="reward",
                output_path=os.path.join(output_dir, 'reward_over_time.png'))
    plot_reward(results, y_col="cumulative_reward",
                output_path=os.path.join(output_dir, 'cumulative_reward_over_time.png'))

    # Final cumulative reward
    plot_final_cumreward(results,
                        output_path=os.path.join(output_dir, 'final_cumulative_reward.png'))

    # Rho plots
    plot_rho_history(rho_histories, conditions, update_conditions, rho_true,
                    output_path=os.path.join(output_dir, 'rho_estimates_over_time.png'))
    plot_rho_mae(rho_histories, conditions, update_conditions, rho_true,
                output_path=os.path.join(output_dir, 'rho_mae_over_time.png'))

    # Exploration distance
    plot_exploration_distance(results,
                             output_path=os.path.join(output_dir, 'exploration_distance.png'))


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
    conditions, update_conditions, corr_matrix, rho_true = setup_conditions()

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

    # Generate plots
    generate_all_plots(results, rho_histories, conditions, update_conditions,
                      rho_true, args.output_dir)

if __name__ == "__main__":
    main()
