"""
Generate the model-illustration panel extracted from models_illustrations.ipynb.

Input data path:
    None. The script generates deterministic synthetic one-dimensional reward
    environments from the provided seeds.

Output data path:
    By default, writes figures to figures/model_illustration/:
    - scale_model_illustration.png
    - scale_model_illustration.svg
    - scale_model_illustration.pdf

Flags:
    --output_dir: Directory for generated figure files.
    --env_seed: Seed for the synthetic reward environments.
    --np_seed: Seed for model prediction reproducibility.
    --formats: One or more output formats: png, svg, pdf.

Commands to test and run:
    poetry run python abm/figure_1d_model_illustration.py --output_dir /tmp/figure_1d_test
    poetry run python abm/figure_1d_model_illustration.py --output_dir figures/model_illustration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.gaussian_process.kernels import RBF

from agent import social_generalization, social_generalization_icm


def configure_matplotlib() -> None:
    """Apply the plotting style used by the original notebook."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["ytick.major.size"] = 0
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    """Evaluate an RBF covariance matrix for one-dimensional inputs."""
    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]
    return RBF(length_scale=length_scale)(x1, x2)


def normalize_centered(values: np.ndarray) -> np.ndarray:
    """Min-max normalize values to the range [-0.5, 0.5]."""
    min_val = values.min()
    max_val = values.max()
    return (values - min_val) / (max_val - min_val) - 0.5


def generate_environments(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    """Generate focal and peer reward curves with target correlations."""
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(0, 10, 100).reshape(-1, 1)

    kernel = rbf_kernel(x_grid, x_grid, length_scale=2.0)
    cholesky = np.linalg.cholesky(kernel + 1e-6 * np.eye(len(x_grid)))

    focal_std = cholesky @ rng.standard_normal(len(x_grid))
    independent_1 = cholesky @ rng.standard_normal(len(x_grid))
    independent_2 = cholesky @ rng.standard_normal(len(x_grid))
    independent_3 = cholesky @ rng.standard_normal(len(x_grid))

    focal = normalize_centered(focal_std)
    correlations = [-0.6, 0.0, 0.6]
    peers = [
        normalize_centered(-0.6 * focal_std + np.sqrt(1 - 0.6**2) * independent_1),
        normalize_centered(independent_2),
        normalize_centered(0.6 * focal_std + np.sqrt(1 - 0.6**2) * independent_3),
    ]

    return x_grid, focal, peers, correlations


def create_figure_1d(env_seed: int = 15, np_seed: int = 422) -> plt.Figure:
    """Create the model-illustration figure from deterministic synthetic inputs."""
    configure_matplotlib()

    x_grid, focal, peer_curves, correlations = generate_environments(seed=env_seed)
    rng = np.random.default_rng(np_seed)

    fig = plt.figure(figsize=(11, 6))
    grid = gridspec.GridSpec(
        3,
        4,
        wspace=0.15,
        hspace=0.15,
        width_ratios=[1, 1, 1, 0.8],
    )

    cmap = plt.get_cmap("viridis")
    peer_colors = [cmap(0.1), cmap(0.5), cmap(0.9)]
    hand_picked_points = [[5, 15], [50, 90], [40, 65]]

    peer_obs_data = []
    x_obs_social = []
    y_obs_social = []

    for row_idx, (rho, peer, color, indices) in enumerate(
        zip(correlations, peer_curves, peer_colors, hand_picked_points)
    ):
        ax = plt.subplot(grid[row_idx, 3])
        ax.plot(x_grid, peer, color=color, lw=2.5)

        this_peer_x = []
        this_peer_y = []
        for idx in indices:
            x_val = x_grid[idx][0]
            y_val = peer[idx]
            ax.scatter(
                x_val,
                y_val,
                s=100,
                facecolors=color,
                edgecolors="k",
                zorder=10,
                marker="o",
            )
            peer_obs_data.append((x_val, y_val, rho, color))
            this_peer_x.append([x_val])
            this_peer_y.append([y_val])

        x_obs_social.append(np.array(this_peer_x))
        y_obs_social.append(np.array(this_peer_y))

        ax.set_title(f"Peer {rho}", fontsize=12, fontweight="bold", pad=5)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([])
        ax.set_xticks([])
        if row_idx == 2:
            ax.set_xlabel("Choice", fontsize=12)

    private_indices = [70, 55]
    x_obs_private = np.array([[x_grid[idx][0]] for idx in private_indices])
    y_obs_private = np.array([[focal[idx]] for idx in private_indices])

    _, mean_sg, std_sg = social_generalization(
        X_obs_private=x_obs_private,
        y_obs_private=y_obs_private,
        X_obs_social=x_obs_social,
        y_obs_social=y_obs_social,
        X_predict=x_grid,
        length_scale=1.0,
        observation_noise_private=0.01,
        observation_noise_social=[20, 20, 0.01],
        beta=0.0,
        tau=1.0,
        random_state=rng.__getstate__(),
        return_predict=True,
    )

    _, mean_list, std_list = social_generalization_icm(
        X_obs_private=x_obs_private,
        y_obs_private=y_obs_private,
        X_obs_social=x_obs_social,
        y_obs_social=y_obs_social,
        X_predict=x_grid,
        length_scale_private=1.0,
        length_scale_social=1.0,
        observation_noise_private=0.01,
        observation_noise_social=0.01,
        rho=[1.0] + correlations,
        beta=0.0,
        tau=1.0,
        random_state=rng.__getstate__(),
        return_full_predictions=True,
    )
    mean_scale = np.array(mean_list[0])
    std_scale = np.array(std_list[0])

    ax_main = plt.subplot(grid[:, :3])
    ax_main.plot(
        x_grid,
        focal,
        color="black",
        ls="--",
        lw=3.5,
        label="True Environment",
        alpha=1,
    )

    scale_std = 2
    ax_main.plot(x_grid, mean_sg, color="#2E86AB", lw=2, ls="-")
    ax_main.plot(x_grid, mean_sg + std_sg / scale_std, color="#2E86AB", lw=1, ls="-")
    ax_main.fill_between(
        x_grid.flatten(),
        mean_sg.flatten() - std_sg / scale_std,
        mean_sg.flatten() + std_sg / scale_std,
        color="#2E86AB",
        alpha=0.1,
    )

    ax_main.plot(x_grid, mean_scale, color="#D64933", lw=2.5, ls="-")
    ax_main.plot(x_grid, mean_scale + std_scale / scale_std, color="#D64933", lw=1, ls="-")
    ax_main.fill_between(
        x_grid.flatten(),
        mean_scale.flatten() - std_scale / scale_std,
        mean_scale.flatten() + std_scale / scale_std,
        color="#D64933",
        alpha=0.1,
    )

    ax_main.scatter(
        x_obs_private,
        y_obs_private,
        s=200,
        facecolors="white",
        edgecolors="black",
        linewidth=2.5,
        zorder=15,
        marker="s",
    )
    for peer_x, peer_y, _, peer_color in peer_obs_data:
        ax_main.scatter(
            peer_x,
            peer_y,
            s=180,
            facecolors=peer_color,
            edgecolors="k",
            zorder=10,
            marker="o",
        )

    ax_main.set_title("Social Agent", fontsize=16, fontweight="bold", pad=15)
    ax_main.set_ylim(-0.9, 0.9)
    ax_main.set_yticks([])
    ax_main.set_xlabel("Choice", fontsize=12)

    legend_elements = [
        Line2D([0], [0], color="black", ls="--", lw=2, alpha=1, label="True Env"),
        Line2D([0], [0], color="#2E86AB", lw=2, ls="-", label="SG calibrated"),
        Line2D([0], [0], color="#D64933", lw=2, ls="-", label="SCALE calibrated"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markeredgecolor="black",
            markerfacecolor="white",
            markersize=10,
            label="Private Obs",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="black",
            markerfacecolor="white",
            markersize=10,
            label="Social Obs",
        ),
    ]
    ax_main.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=14)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Figure 1D model illustration.")
    parser.add_argument("--output_dir", type=Path, default=Path("figures/model_illustration"))
    parser.add_argument("--env_seed", type=int, default=15)
    parser.add_argument("--np_seed", type=int, default=422)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "svg", "pdf"],
        default=["png", "svg", "pdf"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = create_figure_1d(env_seed=args.env_seed, np_seed=args.np_seed)
    for extension in args.formats:
        output_path = args.output_dir / f"scale_model_illustration.{extension}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
