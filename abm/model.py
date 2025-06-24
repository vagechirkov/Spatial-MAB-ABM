import mesa
import networkx as nx
import numpy as np
import pandas as pd

from agent import SocialGPAgent
from mesa import DataCollector
from mesa.discrete_space import Network
from rewards import sample_children_with_corr
from scipy.spatial.distance import cosine


def _build_network(network_type, reward_maps, gamma_pa, rng):
    n = len(reward_maps)

    if network_type == "fully_connected":
        return nx.complete_graph(n)

    if network_type == "similarity_pa":
        G = nx.empty_graph(n)
        G.add_edge(0, 1)  # seed edge
        for new in range(2, n):
            sims = np.array(
                [
                    1.0 - cosine(reward_maps[new].ravel(), reward_maps[j].ravel())
                    for j in range(new)
                ]
            )
            probs = (sims + 1e-6) ** gamma_pa
            probs /= probs.sum()
            chosen = rng.choice(np.arange(new), p=probs)
            G.add_edge(new, int(chosen))

        # enforce min-degree ≥ 4 (attention budget)
        for node in G:
            while G.degree(node) < 4:
                cand = rng.choice(list(set(G.nodes) - {node} - set(G.neighbors(node))))
                G.add_edge(node, cand)
        return G

    raise ValueError(f"Unknown network_type '{network_type}'")


class SocialGPModel(mesa.Model):
    def __init__(
        self,
        *,
        n: int = 4,
        rho_parent_child: float = 0.60,
        rho_child_child: float = 0.60,
        grid_size: int = 11,
        model_type: str = "SG",
        length_scale_private: float | None = 2.0,
        length_scale_social: float | None = 2.0,
        observation_noise_private: float | None = 0.1,
        observation_noise_social: float | None = 0.1,
        alpha: float = 0.5,
        beta_private: float | None = None,
        beta_social: float | None = None,
        tau: float = 1.0,
        network_type: str = "fully_connected",
        attention_budget: int = 4,
        gamma_pa: float = 2.0,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)

        self.num_agents = n
        self.grid_size = grid_size
        self.model_type = model_type
        self.attention_budget = attention_budget
        self.network_type = network_type
        self.gamma_pa = gamma_pa

        rho_parent_child = rho_child_child

        # generate reward maps
        _, child_maps = sample_children_with_corr(
            rng=self.rng,
            n_children=n,
            length_scale=2.0,
            rho_parent_child=rho_parent_child,
            rho_child_child=rho_child_child,
            tol=0.1,
            max_tries=1000
        )

        # generate network
        G = _build_network(network_type, child_maps, gamma_pa, self.rng)
        self.grid = Network(G, random=self.random)

        SocialGPAgent.create_agents(
            self,
            self.num_agents,
            cell=self.rng.choice(
                self.grid.all_cells, replace=False, size=self.num_agents
            ),
            reward_environment=self.rng.choice(
                child_maps, replace=False, size=self.num_agents
            ),
            length_scale_private=length_scale_private,
            length_scale_social=length_scale_social,
            observation_noise_private=observation_noise_private,
            observation_noise_social=observation_noise_social,
            beta_private=beta_private,
            beta_social=beta_social,
            tau=tau,
            alpha=alpha
        )

        self.datacollector = DataCollector(
            model_reporters={
                "avg_cumulative_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]),
                "avg_reward": lambda m: np.mean([a.last_reward for a in m.grid.agents])
            },
            agent_reporters={
                "choice": lambda a: a.last_choice,
                "reward": lambda a: a.last_reward,
                "cumulative_reward": lambda a: a.total_reward,
                "individual_tau_value": lambda a: a.tau,
                "individual_beta_private_value": lambda a: a.beta_private,
                "individual_length_scale_private_value": lambda a: a.length_scale_private,
            },
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    m = SocialGPModel()

    param_grid = {
        "n": [4],
        "model_type": ["SG"], # , "VS"
        "length_scale_private": [1.11],
        "length_scale_social":  [4],
        "observation_noise_private": [0.0001],
        "observation_noise_social":  [3],  # , 20 , 0.0001 + 3  # 3, 50, 100, 200, 500
        "beta_private":  [0.33],
        "beta_social":   [0.33],
        "tau": [[0.03, 0.03, 0.03, 0.05]],
        "alpha": [0.5],
        "seed": list(range(10))
    }

    batch_results = mesa.batch_run(
        SocialGPModel,
        parameters=param_grid,
        iterations=1,
        max_steps=15,
        number_processes=None,
        data_collection_period=-1,
        display_progress=True,
    )

    batch_results = pd.DataFrame(batch_results)
    # batch_results.dropna(inplace=True)
    #mask = (((batch_results["model_type"] == "SG") & (batch_results["observation_noise_social"] > 0.0001)) |
    #        (batch_results["model_type"] == "VS") & (batch_results["observation_noise_social"]) <  12)
    sns.lineplot(batch_results, # [mask],
                 x="Step",
                 y="reward",
                 hue="observation_noise_social"  # "observation_noise_social" # "model_type"
                 )
    plt.show()

    batch_results.head()

