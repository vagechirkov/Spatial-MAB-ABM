import json

import numpy as np
import pandas as pd

from abm.model import SocialGPModelSBI  # , SocialGPModelReplication
from abm.original_sg_model import model_fit, model_sim_orig, param_gen
from abm.replication import _run_simulation


def _env_to_array(env_data):
    x_vals = sorted(set(item["x1"] for item in env_data))
    y_vals = sorted(set(item["x2"] for item in env_data))
    arr = np.zeros((len(y_vals), len(x_vals)))

    for item in env_data:
        x_idx = x_vals.index(item["x1"])
        y_idx = y_vals.index(item["x2"])
        arr[y_idx, x_idx] = item["payoff"]

    return arr


def test_sg_simulation():
    f = open("tests/test_A_canon.json")
    test_env = json.load(f)

    pars = param_gen(4, 1, hom=True, models=3)  # 3 = SG model
    for i in range(4):
        pars[0][i]['lambda'] = 1.0
        pars[0][i]['eps_soc'] = 10.0
        pars[0][i]['beta'] = 0.3
        pars[0][i]['tau'] = 0.03

    envList = []
    for _ in range(5):
        envList.append(test_env)

    np.random.seed(42)
    simulation_results_orig = model_sim_orig(
        pars,
        envList,
        1,
        15,
        payoff=True,
        prior_mean=0,
        prior_scale=1,
        baseEpsilon=0.0001,
        add_noize_to_rewards=False
    )
    avg_reward_orig = simulation_results_orig.groupby(["trial"]).reward.mean()

    child_maps = [_env_to_array(env) for env in test_env]

    np.random.seed(42)
    _model = SocialGPModelSBI(
        child_maps,
        model_type="SG",
        rng=None,
        length_scale_private=pars[0][0]['lambda'],
        length_scale_social=pars[0][0]['lambda'],
        observation_noise_private=0.0001,
        observation_noise_social=pars[0][0]['eps_soc'], #0.001,
        beta_private=pars[0][0]['beta'],
        beta_social=pars[0][0]['beta'],
        # rho=parameters[1],
        tau=pars[0][0]['tau']
    )

    for _ in range(15):
        _model.step()

    simulation_results = _model.datacollector.get_model_vars_dataframe()
    avg_reward_new = simulation_results.avg_reward

    avg_reward_diff = avg_reward_new.values - avg_reward_orig.values
    assert np.allclose(avg_reward_new.values, avg_reward_orig.values)


def test_sg_replication():
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
    new_nll, new_policy = _run_simulation(params, data_new, subtract_reward=0.5, full_nll=True, search_horizon=shor)

    # old fitting procedure
    subdata = data.loc[(data['group']==group)]
    subdata.loc[:,"reward"] = subdata["reward"]-0.5

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

    import matplotlib.pyplot as plt
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



