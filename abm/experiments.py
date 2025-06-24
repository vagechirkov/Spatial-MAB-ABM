import itertools
import os
import uuid

import mesa
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

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
        batch_results, mutant_value, resident_value, max_steps, run, param_key="tau"
    )

    run.finish()


if __name__ == "__main__":
    # np.linspace(0.01, 0.06, 5)
    tau_values = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    # for n_agents in (4, 8):
    #     for n_mutants in (1, 2):
    n_agents = 4
    n_mutants = 1
    effect_of_heterogeneity(
        experiment_name=f"{n_mutants}-tau-mutant_{n_agents}-agents",
        mutant_value=tau_values,
        resident_value=tau_values,
        n_mutants=n_mutants,
        n_seeds=20,
        n_iterations=10,
        n=n_agents
    )
