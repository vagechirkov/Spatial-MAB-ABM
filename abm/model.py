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
        length_scale_is_identical: bool = True,
        observation_noise_private: float | None = 0.1,
        observation_noise_social: float | None = 0.1,
        rho: float = 0.60,
        beta_private: float | None = 0.7,
        beta_social: float | None = 0.7,
        tau: float = 1.0,
        network_type: str = "fully_connected",
        attention_budget: int = 4,
        gamma_pa: float = 2.0,
        seed: int | None = None,
        reward_noise_sd : float = 0.01,
    ):
        super().__init__(seed=seed)

        self.num_agents = n
        self.grid_size = grid_size
        self.model_type = model_type
        self.attention_budget = attention_budget
        self.network_type = network_type
        self.gamma_pa = gamma_pa
        self.reward_noise_sd = reward_noise_sd

        rho_parent_child = rho_child_child

        # generate reward maps
        _, child_maps = sample_children_with_corr(
            rng=self.rng,
            n_children=n,
            length_scale=2.0,
            rho_parent_child=rho_parent_child,
            rho_child_child=rho_child_child,
            tol=0.1,
            max_tries=1000,
            grid_size=grid_size
        )

        # generate network
        G = _build_network(network_type, child_maps, gamma_pa, self.rng)
        self.grid = Network(G, random=self.random)

        if length_scale_is_identical:
            length_scale_social = length_scale_private

        child_maps = [c - 0.5 for c in child_maps]

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
            rho=rho,
        )

        self.datacollector = DataCollector(
            model_reporters={
                "avg_cumulative_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]) + 0.5 * m.steps,
                "avg_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]) / m.steps + 0.5,
                "group_composition": lambda m: "-".join(
                    sorted([str(a.observation_noise_social if m.model_type == "SG" else a.rho) for a in m.grid.agents])[::-1]),
            },
            agent_reporters={
                "choice": lambda a: a.last_choice,
                "reward": lambda a: a.last_reward + 0.5,
                "cumulative_reward": lambda a: a.total_reward + 0.5,
                "individual_tau_value": lambda a: a.tau,
                "individual_beta_private_value": lambda a: a.beta_private,
                "individual_length_scale_private_value": lambda a: a.length_scale_private,
                "social_coupling": lambda a: a.observation_noise_social if a.model.model_type == "SG" else a.rho,
            },
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


class SocialGPModelSBI(mesa.Model):
    def __init__(
            self,
            child_maps,
            rng = None,
            n: int = 4,
            grid_size: int = 11,
            model_type: str = "SG",
            length_scale_private: float | None = 2.0,
            length_scale_social: float | None = 2.0,
            length_scale_is_identical: bool = True,
            observation_noise_private: float | None = 0.1,
            observation_noise_social: float | None = 0.1,
            rho: float = 0.60,
            beta_private: float | None = 0.7,
            beta_social: float | None = 0.7,
            tau: float = 1.0,
            reward_noise_sd : float = 0,
    ):
        super().__init__(rng=rng)

        self.num_agents = n
        self.grid_size = grid_size
        self.model_type = model_type
        self.attention_budget = 4
        self.reward_noise_sd = reward_noise_sd

        if length_scale_is_identical:
            length_scale_social = length_scale_private

        # generate network
        G = nx.complete_graph(n)
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
            rho=rho
        )

        self.datacollector = DataCollector(
            model_reporters={
                "avg_cumulative_reward": lambda m: np.mean(
                    [a.total_reward for a in m.grid.agents]
                ),
                "avg_reward": lambda m: np.mean(
                    [a.last_reward for a in m.grid.agents]
                ),
                "last_choice_distance_private": lambda m: np.mean(
                    [a.last_choice_distance_private for a in m.grid.agents]
                ),
                "last_choice_distance_social": lambda m: np.mean(
                    [a.last_choice_distance_social for a in m.grid.agents]
                ),
                "nearest_choice_distance_private": lambda m: np.mean(
                    [a.nearest_choice_distance_private for a in m.grid.agents]
                ),
                "avg_choice_distance_private": lambda m: np.mean(
                    [a.avg_choice_distance_private for a in m.grid.agents]
                ),
                "nearest_choice_distance_social": lambda m: np.mean(
                    [a.nearest_choice_distance_social for a in m.grid.agents]
                ),
                "avg_choice_distance_social": lambda m: np.mean(
                    [a.avg_choice_distance_social for a in m.grid.agents]
                ),
            },
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


class SocialGPModelReplication(mesa.Model):
    def __init__(
            self,
            reward_map,
            social_choices,
            social_rewards,
            model_type = "SG_fitting",
            individual_choices = None | tuple[tuple[int, int]],
            individual_rewards = None | tuple[float],
            random_choices = None | tuple[bool],
            rng = None,
            length_scale: float = 1.11,
            observation_noise_private: float = 0.0001,
            observation_noise_social: float = 0.1,
            beta: float = 0.33,
            tau: float = 0.03
    ):
        super().__init__(rng=rng)
        self.social_choices = social_choices
        self.social_rewards = social_rewards
        self.individual_choices = individual_choices
        self.individual_rewards = individual_rewards
        self.random_choices = random_choices
        self.model_type = model_type

        self.num_agents = 1
        G = nx.complete_graph(self.num_agents)
        self.grid = Network(G, random=self.random)

        if reward_map is None:
            reward_map = np.zeros((11, 11))

        SocialGPAgent.create_agents(
            self,
            self.num_agents,
            cell=self.rng.choice(
                self.grid.all_cells, replace=False, size=self.num_agents
            ),
            reward_environment=reward_map,
            length_scale_private=length_scale,
            length_scale_social=length_scale,
            observation_noise_private=observation_noise_private,
            observation_noise_social=observation_noise_social,
            beta_private=beta,
            beta_social=beta,
            tau=tau,
            rho=None
        )

        self.datacollector = DataCollector(
            model_reporters={
                "nll": lambda m: np.mean(
                    [a.neg_log_likelihood for a in m.grid.agents]
                ),
            },
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    m = SocialGPModel(model_type="SG-ICM")
    for _ in range(15):
        m.step()

    param_grid = {
        "n": [4],
        "model_type": ["AS", "SG", "SG-ICM", "VS-F", "VS-CK"], # , "VS"
        "length_scale_private": [1.11],  #
        "length_scale_social": [1.11],
        # "length_scale_private": [[2, 1.11, 1.11, 1.11], [1.11, 1.11, 1.11, 1.11]],
        # "length_scale_private": [[2.5, 3, 3, 3], [3, 3, 3, 3]],
        # "length_scale_private": [[0.5, 0.5, 0.5, 0.5], [2.5, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        "observation_noise_private": [0.0001],
        "observation_noise_social":  [0.0001],  # , 20 , 0.0001 + 3  # 3, 50, 100, 200, 500
        # "observation_noise_social":  [[0.01, 12, 12, 12], [3, 12, 12, 12], [12, 12, 12, 12]],
        # "observation_noise_social":  [0.0001, 0.0005, 0.001, 0.1, 3, 12],
        "beta_private": [0.33],
        # "beta_private": [[0.6, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]],
        # "beta_private": [[0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
        "beta_social":   [0.33],
        # "tau": [[0.01, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        #         [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]],
        # "tau": [[0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        #         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]],
        # "tau": [[0.5, 0.01, 0.01, 0.01],
        #         [0.01, 0.01, 0.01, 0.01]],
        "tau": [0.03],
        "rho": [0.5],
        "seed": list(range(20))
    }

    batch_results = mesa.batch_run(
        SocialGPModel,
        parameters=param_grid,
        iterations=1,
        max_steps=15,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,
    )

    batch_results = pd.DataFrame(batch_results)
    batch_results.dropna(inplace=True)
    # mask = (((batch_results["model_type"] == "SG") & (batch_results["observation_noise_social"] > 0.0001)) |
    #         (batch_results["model_type"] == "VS") & (batch_results["observation_noise_social"]) < 12)
    # mask = batch_results["AgentID"] != 1.0

    batch_results['tau_str'] = [str(l) for l in batch_results['tau'].to_list()]

    sns.lineplot(batch_results, # [mask],
                 x="Step",
                 y="reward",
                 hue="model_type"  # "observation_noise_social" # "model_type"
                 )
    plt.show()

    batch_results.head()

