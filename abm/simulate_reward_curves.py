import argparse
import datetime
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from model import SocialGPModelSBI
from rewards import (
    sample_children_with_corr,
    make_parent_and_children_cholesky2,
    build_corr_matrix_option1,
    build_corr_matrix_option2,
    build_corr_matrix_option3,
    _min_max,
)
from sbi.utils import BoxUniform

_ = torch.manual_seed(42)

def parallel_simulate(theta, simulate_fnc, **simulate_kwargs):
    # Our simulator uses numpy, but prior samples are in PyTorch.
    theta_np = theta.numpy()

    # for debugging
    # simulate(theta_np[0, :])

    num_workers = -1  # Use all available CPUs
    simulation_outputs = Parallel(n_jobs=num_workers, verbose=0)(
        delayed(simulate_fnc)(batch, **simulate_kwargs)
        for batch in theta_np
    )
    return np.asarray(simulation_outputs)

def _build_corr_matrix(option: str, eps: float):
    if option == "option1":
        return build_corr_matrix_option1()
    if option == "option2":
        return build_corr_matrix_option2(eps=eps)
    if option == "option3":
        return build_corr_matrix_option3()
    return None

def simulate_sg_fnc(parameters, corr_matrix=None):
    rho_child_child = 0.6  # used only when corr_matrix is None
    n_children = (corr_matrix.shape[0] - 1) if corr_matrix is not None else 4
    columns = [
        "avg_reward",
        "last_choice_distance_private",
        "last_choice_distance_social",
        "nearest_choice_distance_private",
        "avg_choice_distance_private",
        "nearest_choice_distance_social",
        "avg_choice_distance_social"
        # 'private_landscape_reconstruction_mse',
        # 'social_landscape_reconstruction_mse'
    ]
    repetitions = []
    for _ in range(8):  # 8 rounds with the same parameters
        if corr_matrix is None:
            _, child_maps = sample_children_with_corr(
                rng=None,
                n_children=n_children,
                length_scale=2.0,
                rho_parent_child=rho_child_child,
                rho_child_child=rho_child_child,
                tol=0.1,
                max_tries=1000
            )
            child_maps = [_min_max(c) for c in child_maps]
        else:
            _, child_maps = make_parent_and_children_cholesky2(
                rng=None,
                grid_size=11,
                n_children=n_children,
                length_scale=2.0,
                corr_matrix=corr_matrix,
            )
            child_maps = [_min_max(c) for c in child_maps]

        # make the reward maps centered on 0
        child_maps = [c - 0.5 for c in child_maps]

        for _ in range(1):  # environment is repeated only once
            _model = SocialGPModelSBI(
                child_maps=child_maps,
                model_type="SG",
                rng=None,
                n=n_children,
                length_scale_private=parameters[0],
                length_scale_social=parameters[0],
                observation_noise_private=0.001,
                observation_noise_social=parameters[1],
                beta_private=parameters[2],
                beta_social=parameters[2],
                tau=parameters[3],
                reward_noise_sd=0.01
            )

            for _ in range(15):
                _model.step()
            results = _model.datacollector.get_model_vars_dataframe()
            results["avg_reward"] = results["avg_reward"].values + 0.5
            repetitions.append(results.loc[:, columns].to_numpy())

    return np.mean(repetitions, axis=0)

def simulate_sg_model(n_samples, corr_matrix):
    lb = [0.1, 0.0001, 0.01, 0.01]
    up = [5.0, 20, 2.0, 0.1]

    lower_bound = torch.as_tensor(lb)
    upper_bound = torch.as_tensor(up)
    prior = BoxUniform(low=lower_bound, high=upper_bound)

    theta = prior.sample((n_samples,))
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    np.save(f"simulation_outputs_sg_4_par_no_prior_mean_{today_str}_{n_samples}_theta.npy", theta.numpy())

    start_time = time.time()
    simulation_outputs = parallel_simulate(theta, simulate_sg_fnc, corr_matrix=corr_matrix)
    elapsed = time.time() - start_time
    print(f"Execution took {elapsed:.2f} seconds")

    fname = f"simulation_outputs_sg_4_par_no_prior_mean_{today_str}_{n_samples}.npy"
    np.save(fname, simulation_outputs)

def simulate_sg_icm_fnc(parameters, corr_matrix=None):
    rho_child_child = 0.6  # used only when corr_matrix is None
    n_children = (corr_matrix.shape[0] - 1) if corr_matrix is not None else 4
    columns = [
        "avg_reward",
        "last_choice_distance_private",
        "last_choice_distance_social",
        "nearest_choice_distance_private",
        "avg_choice_distance_private",
        "nearest_choice_distance_social",
        "avg_choice_distance_social"
        # 'private_landscape_reconstruction_mse',
        # 'social_landscape_reconstruction_mse'
    ]
    repetitions = []
    for _ in range(8):  # 8 rounds with the same parameters
        if corr_matrix is None:
            _, child_maps = sample_children_with_corr(
                rng=None,
                n_children=n_children,
                length_scale=2.0,
                rho_parent_child=rho_child_child,
                rho_child_child=rho_child_child,
                tol=0.1,
                max_tries=1000
            )
            child_maps = [_min_max(c) for c in child_maps]
        else:
            _, child_maps = make_parent_and_children_cholesky2(
                rng=None,
                grid_size=11,
                n_children=n_children,
                length_scale=2.0,
                corr_matrix=corr_matrix,
            )
            child_maps = [_min_max(c) for c in child_maps]

        # make the reward maps centered on 0
        child_maps = [c - 0.5 for c in child_maps]

        for _ in range(1):  # environment is repeated only once
            _model = SocialGPModelSBI(
                child_maps=child_maps,
                model_type="SG-ICM",
                rng=None,
                n=n_children,
                length_scale_private=parameters[0],
                length_scale_social=parameters[0],
                observation_noise_private=0.001,
                observation_noise_social=0.001,
                rho=parameters[1],
                beta_private=parameters[2],
                beta_social=parameters[2],
                tau=parameters[3],
                reward_noise_sd=0.01
            )

            for _ in range(15):
                _model.step()
            results = _model.datacollector.get_model_vars_dataframe()
            results["avg_reward"] = results["avg_reward"].values + 0.5
            repetitions.append(results.loc[:, columns].to_numpy())

    return np.mean(repetitions, axis=0)


def simulate_sg_icm_model(n_samples, corr_matrix):
    lb = [0.1, -0.25, 0.01, 0.01]
    up = [5.0, 0.99, 2.0, 0.1]

    lower_bound = torch.as_tensor(lb)
    upper_bound = torch.as_tensor(up)
    prior = BoxUniform(low=lower_bound, high=upper_bound)

    theta = prior.sample((n_samples,))
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    np.save(f"simulation_outputs_sg_icm_4_par_no_prior_mean_{today_str}_{n_samples}_theta.npy", theta.numpy())

    start_time = time.time()
    simulation_outputs = parallel_simulate(theta, simulate_sg_icm_fnc, corr_matrix=corr_matrix)
    elapsed = time.time() - start_time
    print(f"Execution took {elapsed:.2f} seconds")

    fname = f"simulation_outputs_sg_icm_4_par_no_prior_mean_{today_str}_{n_samples}.npy"
    np.save(fname, simulation_outputs)



if __name__ == "__main__":
    # poetry run python simulate_reward_curves.py --n_samples 10_000 --model SG
    # poetry run python simulate_reward_curves.py --n_samples 10_000 --model SG-ICM
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",
                        type=int, default=2_000, help="Number of samples to draw from the prior")
    parser.add_argument("--model", type=str, default="SG-ICM")
    parser.add_argument(
        "--corr_matrix_option",
        type=str,
        default="none",
        choices=["none", "option1", "option2", "option3"],
        help="Select a predefined correlation matrix; 'none' keeps scalar correlation sampling.",
    )
    parser.add_argument(
        "--corr_eps",
        type=float,
        default=0.2,
        help="Epsilon for option2 random AS-AS correlations.",
    )
    args = parser.parse_args()

    corr_matrix = _build_corr_matrix(args.corr_matrix_option, args.corr_eps)

    if args.model == "SG":
        simulate_sg_model(args.n_samples, corr_matrix)
    elif args.model == "SG-ICM":
        simulate_sg_icm_model(args.n_samples, corr_matrix)
    else:
        raise ValueError("Unknown model")
