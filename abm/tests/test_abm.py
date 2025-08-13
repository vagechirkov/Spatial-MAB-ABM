import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from abm.model import SocialGPModelSBI
from abm.original_sg_model import model_fit, model_sim_orig, param_gen
from abm.replication import _run_simulation
from abm.rewards import sample_children_with_corr


def _env_to_array(env_data):
    x_vals = sorted(set(item["x1"] for item in env_data))
    y_vals = sorted(set(item["x2"] for item in env_data))
    arr = np.zeros((len(y_vals), len(x_vals)))

    for item in env_data:
        x_idx = x_vals.index(item["x1"])
        y_idx = y_vals.index(item["x2"])
        arr[y_idx, x_idx] = item["payoff"]

    return arr

def _array_to_env(arr):
    env_data = []
    for y in range(arr.shape[1]):
        for x in range(arr.shape[0]):
            env_data.append({"x1": x + 1, "x2": y + 1, "payoff": arr[y, x]})
    return env_data


@pytest.mark.parametrize("eps_soc", [1.0, 10.0])
@pytest.mark.parametrize("beta", [0.3, 0.7])
@pytest.mark.parametrize("prior_mean", [0.0])  # , 0.5
def test_sg_simulation(eps_soc, beta, prior_mean):
    np.random.seed(42)
    _, child_maps = sample_children_with_corr(
        rng=None,
        n_children=4,
        length_scale=2.0,
        rho_parent_child=0.6,
        rho_child_child=0.6,
        tol=0.1,
        max_tries=1000
    )

    n_simulations = 200

    pars = param_gen(4, 1, hom=True, models=3)  # 3 = SG model
    for i in range(4):
        pars[0][i]['lambda'] = 1.0
        pars[0][i]['eps_soc'] = eps_soc
        pars[0][i]['beta'] = beta
        pars[0][i]['tau'] = 0.03

    envList = [[_array_to_env(c)] for c in child_maps]

    plt.imshow(child_maps[0])
    plt.show()
    plt.imshow(child_maps[1])
    plt.show()


    np.random.seed(42)
    simulation_results_orig = model_sim_orig(
        pars,
        envList,
        n_simulations,
        15,
        payoff=True,
        prior_mean=prior_mean,
        prior_scale=1,
        baseEpsilon=0.0001,
        add_noize_to_rewards=True
    )
    avg_reward_orig = simulation_results_orig.groupby(["trial"]).reward.mean().values

    np.random.seed(42)
    avg_reward_new = np.zeros_like(avg_reward_orig)
    child_maps = [m - prior_mean for m in child_maps]
    
    for r in range(n_simulations):
        _model = SocialGPModelSBI(
            child_maps=child_maps,
            model_type="SG",
            rng=None,
            length_scale_private=pars[0][0]['lambda'],
            length_scale_social=pars[0][0]['lambda'],
            observation_noise_private=0.0001,
            observation_noise_social=pars[0][0]['eps_soc'],
            beta_private=pars[0][0]['beta'],
            beta_social=pars[0][0]['beta'],
            tau=pars[0][0]['tau'],
            reward_noise_sd=0.01
        )

        for _ in range(15):
            _model.step()

        simulation_results = _model.datacollector.get_model_vars_dataframe()
        avg_reward_new += simulation_results.avg_reward.values
    avg_reward_new /= n_simulations
    avg_reward_diff = avg_reward_orig - avg_reward_new

    plt.plot(avg_reward_diff)
    plt.ylim([-0.05, 0.05])
    plt.show()

    plt.plot(avg_reward_orig, label="original")
    plt.plot(avg_reward_new, label="new")
    plt.legend()
    plt.show()

    assert np.allclose(avg_reward_new, avg_reward_orig, atol=0.05)


@pytest.mark.parametrize("prior_mean", [0.0, 0.5])  # , 0.5
def test_sg_replication(prior_mean):
    data = pd.read_csv("data/e1_data.csv")
    agent = 1
    group = 0
    round_test = 0
    shor = 15

    # "lambda","beta","tau","eps_soc"
    params = np.log(np.array([1.0, 0.3, 0.03, 10]))

    # new fitting procedure
    data_new = data[
        (data["group"] == group)
        & (data["agent"] == agent)
        & (data["round"] != round_test)
        ].reset_index(drop=True)

    np.random.seed(2023)
    new_nll, new_policy = _run_simulation(params, data_new, subtract_reward=prior_mean,
                                          full_nll=True, search_horizon=shor)

    # old fitting procedure
    subdata = data.loc[(data['group']==group)]
    subdata.loc[:,"reward"] = subdata["reward"] - prior_mean

    tardata = subdata.loc[(subdata['agent']==agent) &
                          (subdata['round']!=round_test),
    ['round','trial','choice','reward','isRandom']]
    tardata = tardata.to_numpy()
    socdata = subdata.loc[(subdata['agent']!=agent) &
                          (subdata['round']!=round_test),
    ['round','trial','choice','reward','isRandom','agent']]
    socdata = socdata.to_numpy()

    np.random.seed(2023)
    orig_nll, orig_policy = model_fit(params, 3, tardata, socdata, shor=shor, full_nll=True)

    diff_nll = np.array(orig_nll) - np.array(new_nll)
    orig_policy = np.array(orig_policy)
    new_policy = np.array(new_policy)

    for i in range(7):
        plt.plot(diff_nll[i * 14: (i + 1) * 14], label="round {}".format(i))
    plt.legend()
    plt.show()

    plt.imshow(orig_policy[0, :].reshape(11, 11))
    plt.show()
    plt.imshow(new_policy[0, :].reshape(11, 11).T)
    plt.show()

    plt.imshow(orig_policy[13, :].reshape(11, 11))
    plt.show()
    plt.imshow(new_policy[13, :].reshape(11, 11).T)
    plt.show()

    plt.imshow(orig_policy[14, :].reshape(11, 11))
    plt.show()
    plt.imshow(new_policy[14, :].reshape(11, 11).T)
    plt.show()

    assert np.allclose(new_nll, orig_nll, atol=0.05)


def test_prior_mean_subtraction():
    np.random.seed(42)
    _, child_maps_base = sample_children_with_corr(
        rng=None,
        n_children=4,
        length_scale=2.0,
        rho_parent_child=0.6,
        rho_child_child=0.6,
        tol=0.1,
        max_tries=1000
    )

    n_simulations = 200
    avg_rewards = []

    for prior_mean in [0.0, 0.5]:
        avg_reward = np.zeros(15)
        child_maps = [m - prior_mean for m in child_maps_base]

        for r in range(n_simulations):
            _model = SocialGPModelSBI(
                child_maps=child_maps,
                model_type="SG",
                rng=None,
                length_scale_private=1.0,
                length_scale_social=1.0,
                observation_noise_private=0.0001,
                observation_noise_social=10.0,
                beta_private=0.3,
                beta_social=0.3,
                tau=0.03,
                reward_noise_sd=0.01
            )

            for _ in range(15):
                _model.step()

            simulation_results = _model.datacollector.get_model_vars_dataframe()
            avg_reward += simulation_results.avg_reward.values
        avg_reward /= n_simulations
        avg_rewards.append(avg_reward)

        plt.plot(avg_reward + prior_mean, label=f"prior mean: {prior_mean}")

    plt.legend()
    plt.show()
