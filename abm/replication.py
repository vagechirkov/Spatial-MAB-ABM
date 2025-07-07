import numpy as np
import pandas as pd
import scipy.optimize as opt
from tqdm import tqdm

from model import SocialGPModelReplication

Xstar = np.array([(x, y) for x in range(11) for y in range(11)])

def _run_simulation(x, data):
    nll = 0.0
    for r in np.unique(data["round"]):
        agent_trials = data.loc[(data["round"] == r)].sort_values("trial")

        social_choices = [
            Xstar[np.int32(agent_trials[col].values[1:])]
            for col in ["soc_choice1", "soc_choice2", "soc_choice3"]
        ]
        social_rewards = [
            agent_trials[col].values[1:]
            for col in ["soc_rew1", "soc_rew2", "soc_rew3"]
        ]
        _model = SocialGPModelReplication(
            social_choices=social_choices,
            social_rewards=social_rewards,
            individual_choices=Xstar[np.int32(agent_trials["choice"].values)],
            individual_rewards=agent_trials["reward"].values,
            model_type="SG_fitting",
            length_scale=np.exp(x[0]),
            beta=np.exp(x[1]),
            tau=np.exp(x[2]),
            observation_noise_private=0.0001,
            observation_noise_social=np.exp(x[3]),
            reward_map=None
        )

        for _ in range(15):
            _model.step()

        results = _model.datacollector.get_model_vars_dataframe()
        nll += results['nll'].sum()
    return nll


def replicate_model_fitting():
    np.random.seed(2023)
    data = pd.read_csv("./data/e1_data.csv")

    results = []
    param_names = ['length_scale', 'beta', 'tau', 'observation_noise_social']

    for g in tqdm(np.unique(data["group"])):
        for a in np.unique(data.loc[(data["group"] == g), "agent"]):
            repeated_optimizations = []
            test_nll = 0
            for r in np.unique(data["round"]):
                pars = opt.differential_evolution(
                    _run_simulation,
                    [(-5, 3), (-5, 3), (np.log(0.01), 3), (-5, np.log(19))],  # -7.5
                    (
                        data[
                            (data["group"] == g)
                            & (data["agent"] == a)
                            & (data["round"] != r)
                        ].reset_index(drop=True),
                    ),
                    maxiter=100,
                    workers=-1,
                )["x"]
                repeated_optimizations.append(np.exp(pars))
                test_nll += _run_simulation(
                    pars,
                    data[
                        (data["group"] == g)
                        & (data["agent"] == a)
                        & (data["round"] == r)
                    ],
                )

            mean_pars = np.mean(repeated_optimizations, axis=0)

            result = {
                "group": g,
                "agent": a,
                "nll": test_nll
            }
            result.update({param: val for param, val in zip(param_names, mean_pars)})
            results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv("./data/replicate_fits.csv", index=False)


if __name__ == "__main__":
    replicate_model_fitting()