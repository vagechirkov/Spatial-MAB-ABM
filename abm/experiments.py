import itertools
import os
import uuid

import mesa
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from model import SocialGPModel

def analyze_and_log_heatmaps(
    df, mutant_values, resident_values, max_steps, run, param_key="tau"
):
    total_avg_reward_matrix = []
    residents_reward_matrix = []
    diff_matrix = []

    df["mutant_param"] = df[param_key].apply(lambda p: p[0])
    df["resident_param"] = df[param_key].apply(lambda p: p[-1])
    grouped = df.groupby(["mutant_param", "resident_param"])

    # For heatmaps, order by mutant and resident value grid
    for m in mutant_values:
        param_matrix_row = []
        reward_row = []
        diff_row = []
        for r in resident_values:
            group = grouped.get_group((m, r))
            total_mean = group["cumulative_reward"].mean()
            # we need f"individual_{param_key}_value" to identify whether individual is mutant or resident
            mutant_mean = group.loc[group[f"individual_{param_key}_value"] == m, "cumulative_reward"].mean()
            resident_mean = group.loc[group[f"individual_{param_key}_value"] == r, "cumulative_reward"].mean()
            norm = 1 * max_steps
            param_matrix_row.append(total_mean / norm)
            reward_row.append(resident_mean / norm)
            diff_row.append((resident_mean - mutant_mean) / norm)
        total_avg_reward_matrix.append(param_matrix_row)
        residents_reward_matrix.append(reward_row)
        diff_matrix.append(diff_row)

    total_avg_reward_matrix = np.array(total_avg_reward_matrix)
    residents_reward_matrix = np.array(residents_reward_matrix)
    diff_matrix = np.array(diff_matrix)

    # Find the homogeneous rewards (diagonal)
    diag_values = np.diag(total_avg_reward_matrix)
    # Build a matrix to subtract from every cell: diag_matrix[i, j] = diag_values[j]
    diag_matrix = np.zeros_like(total_avg_reward_matrix)
    for i in range(len(mutant_values)):
        for j in range(len(resident_values)):
            # Use resident axis for diagonal subtraction
            if j < len(diag_values):
                diag_matrix[i, j] = diag_values[j]
    reward_matrix_minus_diag = total_avg_reward_matrix - diag_matrix

    # Plot & log all heatmaps
    def wandb_log_heatmap(matrix, title, xvals, yvals, key):
        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, aspect="auto", origin="lower")
        plt.colorbar(label=title)
        plt.xticks(range(len(xvals)), np.round(xvals, 2))
        plt.yticks(range(len(yvals)), np.round(yvals, 2))
        plt.xlabel("Resident Value")
        plt.ylabel("Mutant Value")
        plt.title(title)
        plt.tight_layout()
        run.log({key: wandb.Image(plt)})
        plt.close()

    wandb_log_heatmap(
        total_avg_reward_matrix,
        "Avg Cumulative Reward (norm)",
        resident_values,
        mutant_values,
        "total_reward_heatmap",
    )

    wandb_log_heatmap(
        residents_reward_matrix,
        "Residents Cumulative Reward (norm)",
        resident_values,
        mutant_values,
        "residents_reward_heatmap",
    )

    wandb_log_heatmap(
        diff_matrix,
        "Resident-Mutant Reward Diff (norm)",
        resident_values,
        mutant_values,
        "diff_heatmap",
    )

    wandb_log_heatmap(
        reward_matrix_minus_diag,
        "Reward minus Homogeneous (norm)",
        resident_values,
        mutant_values,
        "reward_minus_diag_heatmap",
    )

    # run.log(
    #     {
    #         "full_dataframe": wandb.Table(dataframe=df),
    #     }
    # )


def effect_of_heterogeneity(
    experiment_name,
    parameter="tau",
    mutant_value=(0.05,),
    resident_value=(0.03,),
    n_mutants=1,
    n_seeds=200,
    n_iterations=1,
    **kwargs,
):
    parameter_array = []

    for m, r in itertools.product(mutant_value, resident_value):
        parameter_array.append([m] * n_mutants + [r] * (kwargs["n"] - n_mutants))

    param_grid = {
        "model_type": "SG",
        "length_scale_private": 1.11,
        "length_scale_social": 1.11,
        "observation_noise_private": 0.0001,
        "observation_noise_social": 3,
        "beta_private": 0.33,
        "beta_social": 0.33,
        "tau": 0.03,
        "alpha": 0.5,
        "seed": list(range(n_seeds)),
    }
    # add kwargs to param_grid
    param_grid.update(kwargs)
    param_grid[parameter] = parameter_array

    # wandb setup
    wandb.login()
    config_info = dict(n_seeds=n_seeds, n_runs=n_seeds*n_iterations, **param_grid)
    config_info["mutant_value"] = mutant_value
    config_info["resident_value"] = resident_value
    config_info["n_mutants"] = n_mutants
    config_info["mutant_param"] = parameter
    config_info["n_iterations"] = n_iterations

    run = wandb.init(
        project="effect_of_heterogeneity",
        name=experiment_name + "-" + str(uuid.uuid4())[:8],
        config=config_info,
        dir=os.getcwd(),
    )

    max_steps = 15
    batch_results = mesa.batch_run(
        SocialGPModel,
        parameters=param_grid,
        iterations=n_iterations,
        max_steps=max_steps,
        number_processes=None,
        data_collection_period=-1,
        display_progress=True,
    )
    batch_results = pd.DataFrame(batch_results)

    analyze_and_log_heatmaps(
        batch_results, mutant_value, resident_value, max_steps, run, param_key=parameter
    )

    run.finish()


def exp_heterogeneity_tau():
    # np.linspace(0.01, 0.06, 5)
    tau_values = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    for n_agents in (4, 8):
        for n_mutants in (1, 2):
            effect_of_heterogeneity(
                experiment_name=f"{n_mutants}-tau-mutant_{n_agents}-agents",
                mutant_value=tau_values,
                resident_value=tau_values,
                n_mutants=n_mutants,
                n_seeds=20,
                n_iterations=10,
                n=n_agents
            )


def exp_heterogeneity_beta():
    beta_values = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    n_mutants = 1
    n_agents = 4
    param_key = "beta_private"
    effect_of_heterogeneity(
        experiment_name=f"{n_mutants}-{param_key}-mutant_{n_agents}-agents",
        parameter=param_key,
        mutant_value=beta_values,
        resident_value=beta_values,
        n_mutants=n_mutants,
        n_seeds=20,
        n_iterations=10,
        n=n_agents
    )


def exp_heterogeneity_length_scale_private():
    beta_values = (0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3)
    n_mutants = 1
    n_agents = 4
    param_key = "length_scale_private"
    effect_of_heterogeneity(
        experiment_name=f"{n_mutants}-{param_key}-mutant_{n_agents}-agents",
        parameter=param_key,
        mutant_value=beta_values,
        resident_value=beta_values,
        n_mutants=n_mutants,
        n_seeds=40,
        n_iterations=10,
        n=n_agents
    )



def plot_heatmap_cumulative_score(
    df,
    run,
    score_column="cumulative_reward",
    x_col="observation_noise_social_factor",
    y_col="rho_child_child",
    vmin=-2.0,
    vmax=2.0,
):
    # df['observation_noise_social_factor'] = df["observation_noise_social"] / observation_noise_private
    agg = (
        df.groupby([y_col, x_col])[score_column]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        agg.values,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap="seismic"
    )

    # Make colorbar same height as axes
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(score_column)

    ax.set_xticks(range(len(agg.columns)))
    ax.set_xticklabels([f"{v:.2g}" for v in agg.columns])
    ax.set_yticks(range(len(agg.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in agg.index])

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()

    run.log({f"{score_column}_{x_col}_{y_col}_heatmap": wandb.Image(plt)})
    plt.close()


def exp_obsnoise_vs_rho(
    n_agents=4,
    n_seeds=200,
    n_iterations=10,
    max_steps=15,
    observation_noise_private=0.0001,
    observation_noise_social_rel_factors=(1.0, 2.0, 5.0, 10.0, 100, 1000, 10_000, 20_000, 30_000, 100_000),
    rho_child_child_values=(0.4, 0.6, 0.8),
    **kwargs,
):
    # Build value grids
    obs_noise_social_values = [
        observation_noise_private * f for f in observation_noise_social_rel_factors
    ]

    params = dict(
        model_type="SG",
        n=n_agents,
        length_scale_private=1.11,
        observation_noise_private=observation_noise_private,
        observation_noise_social=obs_noise_social_values,
        rho_child_child=rho_child_child_values,
        beta_private=0.33,
        tau=0.03,
        seed=list(range(n_seeds)),
    )
    params.update(kwargs)

    # wandb setup
    wandb.login()
    config_info = dict(
        n_seeds=n_seeds,
        n_runs=n_seeds * n_iterations,
        **params
    )
    run = wandb.init(
        project="effect_of_obsnoise_vs_rho",
        name=f"obsnoise_vs_rho_{uuid.uuid4().hex[:6]}",
        config=config_info,
        dir=os.getcwd(),
    )

    batch_results = mesa.batch_run(
        SocialGPModel,
        parameters=params,
        iterations=n_iterations,
        max_steps=max_steps,
        number_processes=None,
        data_collection_period=-1,
        display_progress=True,
    )
    batch_results = pd.DataFrame(batch_results)
    # plot_heatmap_cumulative_score(batch_results, run, observation_noise_private=observation_noise_private)
    plt.figure(figsize=(12, 6))
    sns.catplot(
        data=batch_results,
        kind="point",
        hue="rho_child_child",
        y="cumulative_reward",
        x="observation_noise_social",
        dodge=0.3,
        linestyles="",
        aspect=1.3,
    )
    run.log({"cumulative_reward_catplot": wandb.Image(plt)})
    plt.close()

    run.finish()


def exp_env_size(n_seeds=200, n_iterations=1, max_steps=15, **kwargs):
    params = dict(
        model_type=["AS", "SG"],
        n=4,
        length_scale_private=1.11,
        observation_noise_private=0.0001,
        observation_noise_social=2.0,
        rho_child_child=0.6,
        beta_private=0.33,
        tau=0.03,
        grid_size=(11, 15, 19, 23, 27),
        seed=list(range(n_seeds)),
    )
    params.update(kwargs)

    # wandb setup
    wandb.login()
    config_info = dict(
        n_seeds=n_seeds,
        n_runs=n_seeds * n_iterations,
        max_steps=max_steps,
        heterogeneity=True,
        **params
    )
    run = wandb.init(
        project="soc-MAB-ABM",
        group="effect_of_env_size",
        name=f"env_size_{uuid.uuid4().hex[:6]}",
        config=config_info,
        dir=os.getcwd(),
    )

    batch_results = mesa.batch_run(
        SocialGPModel,
        parameters=params,
        iterations=n_iterations,
        max_steps=max_steps,
        number_processes=None,
        data_collection_period=-1,
        display_progress=True,
    )
    batch_results = pd.DataFrame(batch_results)

    plt.figure(figsize=(12, 6))
    sns.catplot(
        data=batch_results,
        kind="point",
        hue="model_type",
        y="cumulative_reward",
        x="grid_size",
        dodge=0.3,
        linestyles="",
        aspect=1.3,
    )
    run.log({"cumulative_reward_catplot": wandb.Image(plt)})
    plt.close()

    run.finish()


def exp_social_generalization_model(
    n_seeds=200, n_iterations=1, max_steps=15, comparison_model_type="VF-ICM", **kwargs
):
    l_s_priv = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
    params = dict(
        model_type=["AS", comparison_model_type],
        n=4,
        grid_size=11,
        rho_child_child=0.6,
        length_scale_private=l_s_priv,
        length_scale_social=2.0,
        length_scale_is_identical=True,
        observation_noise_private=0.001,
        observation_noise_social=0.001,  # (0.001, 0.001*10, 0.001*100, 0.001*1000, 0.001*10_000, 0.001*100_000),
        beta_private=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        beta_social=0.7,
        rho=0,  # (-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
        tau=0.03,
        seed=list(range(n_seeds)),
    )
    params.update(kwargs)

    # wandb setup
    wandb.login()
    config_info = dict(
        n_seeds=n_seeds,
        n_runs=n_seeds * n_iterations,
        max_steps=max_steps,
        heterogeneity=False,
        **params
    )
    run = wandb.init(
        project="soc-MAB-ABM",
        group="value_fusion_model",
        name=f"VF_{uuid.uuid4().hex[:6]}",
        config=config_info,
        dir=os.getcwd(),
    )

    batch_results = mesa.batch_run(
        SocialGPModel,
        parameters=params,
        iterations=n_iterations,
        max_steps=max_steps,
        number_processes=None,
        data_collection_period=-1,
        display_progress=True,
    )
    batch_results = pd.DataFrame(batch_results)

    # for l in [1.0, 2.0, 3.0]:
    #     plt.figure(figsize=(12, 6))
    #     sns.catplot(
    #         data=batch_results[batch_results['length_scale_private'] == l],
    #         kind="point",
    #         hue="model_type",
    #         y="cumulative_reward",
    #         x="rho",
    #         dodge=0.3,
    #         linestyles="",
    #         aspect=1.3,
    #     )
    #     run.log({f"cumulative_reward_catplot_lambda_{l}": wandb.Image(plt)})
    #     plt.close()

    x_values_column = "length_scale_private"  # "rho" "observation_noise_social"
    column = "beta_private" # "length_scale_private"
    for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        plt.figure(figsize=(12, 6))
        sns.catplot(
            data=batch_results[batch_results['beta_private'] == b],
            kind="point",
            hue="model_type",
            y="cumulative_reward",
            x=x_values_column,
            dodge=0.3,
            linestyles="",
            aspect=1.3,
        )
        run.log({f"cumulative_reward_catplot_beta_{b}": wandb.Image(plt)})
        plt.close()


    group_cols = [x_values_column, column, "model_type"]  # "rho"
    grouped = (
        batch_results
        .groupby(group_cols)["cumulative_reward"]
        .mean()
        .reset_index()
    )

    # Pivot to get AS and VF-ICM as columns for each (rho, length_scale_private)
    pivot = grouped.pivot_table(
        index=[x_values_column, column],
        columns="model_type",
        values="cumulative_reward"
    ).reset_index()

    # Calculate the difference: VF-ICM minus AS
    pivot["cumulative_reward_diff"] = pivot[comparison_model_type] - pivot["AS"]

    # Prepare for heatmap plotting
    batch_results_diff = pivot

    plot_heatmap_cumulative_score(
        batch_results_diff,
        run,
        score_column="cumulative_reward_diff",
        x_col=x_values_column,
        y_col=column,
        vmin=-1.5,
        vmax=1.5,
    )

    run.finish()


if __name__ == "__main__":
    n_seeds = 200
    n_iterations = 5
    max_steps = 15

    for rho_env in [0.6, 0.4]:  # , 0.4
        for tau in [0.01]:  # , 0.01
            # for o_n in [0.001, 0.001*10, 0.001*100, 0.001*1000, 0.001*10_000, 0.001*20_000]:
            for rho in [0.6, -0.2, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]:  # , -0.2, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8,
                exp_social_generalization_model(
                    n_seeds=n_seeds,
                    n_iterations=n_iterations,
                    max_steps=max_steps,
                    rho=rho,
                    # observation_noise_social=o_n,
                    comparison_model_type="VF−ICM",
                    # length_scale_private=length_scale,
                    # beta_private=beta,
                    # beta_social=beta,
                    rho_child_child=rho_env,
                    tau=tau,
                )
