import argparse
import datetime
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from model import SocialGPModelSBI
from rewards import sample_children_with_corr
from sbi.utils import BoxUniform

_ = torch.manual_seed(42)

def parallel_simulate(theta, simulate_fnc):
    # Our simulator uses numpy, but prior samples are in PyTorch.
    theta_np = theta.numpy()

    # for debugging
    # simulate(theta_np[0, :])

    num_workers = -1  # Use all available CPUs
    simulation_outputs = Parallel(n_jobs=num_workers, verbose=0)(
        delayed(simulate_fnc)(batch)
        for batch in theta_np
    )
    return np.asarray(simulation_outputs)

def simulate_sg_fnc(parameters):
    rho_child_child = 0.6
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
        _, child_maps = sample_children_with_corr(
            rng=None,
            n_children=4,
            length_scale=2.0,
            rho_parent_child=rho_child_child,
            rho_child_child=rho_child_child,
            tol=0.1,
            max_tries=1000
        )

        # make the reward maps centered on 0
        child_maps = [c - 0.5 for c in child_maps]

        for _ in range(1):  # environment is repeated only once
            _model = SocialGPModelSBI(
                child_maps,
                model_type="SG",
                rng=None,
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

def simulate_sg_model(n_samples):
    lb = [0.1, 0.0001, 0.01, 0.01]
    up = [5.0, 20, 2.0, 0.1]

    lower_bound = torch.as_tensor(lb)
    upper_bound = torch.as_tensor(up)
    prior = BoxUniform(low=lower_bound, high=upper_bound)

    theta = prior.sample((n_samples,))
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    np.save(f"simulation_outputs_sg_4_par_no_prior_mean_{today_str}_{n_samples}_theta.npy", theta.numpy())

    start_time = time.time()
    simulation_outputs = parallel_simulate(theta, simulate_sg_fnc)
    elapsed = time.time() - start_time
    print(f"Execution took {elapsed:.2f} seconds")

    fname = f"simulation_outputs_sg_4_par_no_prior_mean_{today_str}_{n_samples}.npy"
    np.save(fname, simulation_outputs)

def simulate_sg_icm_fnc(parameters):
    rho_child_child = 0.6
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
        _, child_maps = sample_children_with_corr(
            rng=None,
            n_children=4,
            length_scale=2.0,
            rho_parent_child=rho_child_child,
            rho_child_child=rho_child_child,
            tol=0.1,
            max_tries=1000
        )

        # make the reward maps centered on 0
        child_maps = [c - 0.5 for c in child_maps]

        for _ in range(1):  # environment is repeated only once
            _model = SocialGPModelSBI(
                child_maps,
                model_type="SG-ICM",
                rng=None,
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


def simulate_sg_icm_model(n_samples):
    lb = [0.1, -0.25, 0.01, 0.01]
    up = [5.0, 0.99, 2.0, 0.1]

    lower_bound = torch.as_tensor(lb)
    upper_bound = torch.as_tensor(up)
    prior = BoxUniform(low=lower_bound, high=upper_bound)

    theta = prior.sample((n_samples,))
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    np.save(f"simulation_outputs_sg_icm_4_par_no_prior_mean_{today_str}_{n_samples}_theta.npy", theta.numpy())

    start_time = time.time()
    simulation_outputs = parallel_simulate(theta, simulate_sg_icm_fnc)
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
    args = parser.parse_args()

    if args.model == "SG":
        simulate_sg_model(args.n_samples)
    elif args.model == "SG-ICM":
        simulate_sg_icm_model(args.n_samples)
    else:
        raise ValueError("Unknown model")
