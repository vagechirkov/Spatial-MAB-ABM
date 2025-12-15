import mesa
import networkx as nx
import numpy as np
import pandas as pd

from agent import SocialGPAgent
from mesa import DataCollector
from mesa.discrete_space import Network
from rewards import (
    sample_children_with_corr,
    make_parent_and_children_cholesky2,
    build_corr_matrix_option1,
    build_corr_matrix_option2,
    build_corr_matrix_option3,
    _min_max,
)
from scipy.spatial.distance import cosine


def _build_network(network_type, reward_maps, gamma_pa, rng):
    n = len(reward_maps)

    if network_type == "fully_connected":
        return nx.complete_graph(n)

    if network_type == "directed_one_to_four":
        if n < 2:
            raise ValueError(
                f"Network type '{network_type}' requires at least 2 nodes (1 source + 1 targets), but found {n}.")

        # Create a directed graph
        G = nx.empty_graph(n, create_using=nx.DiGraph)

        # Always connect Node 0 to Node 1, 2, 3, 4, ... (Fixed order)
        source_node = 0
        for target in range(1, n):
            G.add_edge(source_node, target)

        return G

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
        rho: float | np.ndarray = 0.60,
        beta_private: float | None = 0.7,
        beta_social: float | None = 0.7,
        tau: float = 1.0,
        tau_sampling: bool = False,
        beta_sampling: bool = False,
        length_scale_sampling: bool = False,
        network_type: str = "fully_connected",
        attention_budget: int = 4,
        gamma_pa: float = 2.0,
        seed: int | None = None,
        reward_noise_sd : float = 0.01,
        corr_matrix=None, # for new environment matrix
        child_maps=None,  # for new environment matrix
    ):
        super().__init__(seed=seed)

        self.num_agents = n
        self.grid_size = grid_size
        self.attention_budget = attention_budget
        self.network_type = network_type
        self.gamma_pa = gamma_pa
        self.reward_noise_sd = reward_noise_sd

        # check the model types
        assert model_type in ["SG", "SG-ICM", "AS"]

        # sample parameters from priors (matching Witt et al. repo)
        if tau_sampling:
            tau = float(self.rng.lognormal(mean=-4.5, sigma=0.9))

        def _sample_positive_from_prior(mean, sigma, lower=1e-9):
            """Sample log-normal and guard against zeros."""
            draw = float(self.rng.lognormal(mean=mean, sigma=sigma))
            return max(draw, lower)

        if length_scale_sampling:
            # λ prior: LogNormal(-0.75, 0.5)
            length_scale_private = _sample_positive_from_prior(
                mean=-0.75, sigma=0.5, lower=1e-6
            )
            if length_scale_is_identical:
                length_scale_social = length_scale_private
            else:
                length_scale_social = _sample_positive_from_prior(
                    mean=-0.75, sigma=0.5, lower=1e-6
                )

        if beta_sampling:
            # β prior: LogNormal(-0.75, 0.5)
            beta_private = _sample_positive_from_prior(
                mean=-0.75, sigma=0.5, lower=1e-6
            )
            beta_social = _sample_positive_from_prior(
                mean=-0.75, sigma=0.5, lower=1e-6
            )
        
        # 1. If explicit maps are passed in (for testing the code)
        if child_maps is not None:
            # maybe sanity-check length == n
            child_maps = list(child_maps)
            assert np.array(rho).size == len(child_maps) == self.num_agents
            rho = [rho]

        # 2. if a full correlation matrix is provided, use the new generator
        elif corr_matrix is not None:
            parent, child_maps = make_parent_and_children_cholesky2(
                rng=self.rng,
                grid_size=grid_size,
                n_children=n,
                length_scale=2.0,
                corr_matrix=corr_matrix,
            )
            # keep the reward scale consistent with the scalar-corr branch
            child_maps = [_min_max(c) for c in [parent] + child_maps]

            assert np.array(rho).size == len(child_maps) == self.num_agents
            rho = [rho]

        # 3. original scalar-correlation behavior
        else:
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

        # only one agent is social for this network structure
        if network_type == "directed_one_to_four":
            agent_model_type = [model_type] + ['AS'] * (self.num_agents - 1)
        else:
            agent_model_type = model_type

        SocialGPAgent.create_agents(
            self,
            self.num_agents,
            cell=self.grid.all_cells.cells,
            reward_environment=child_maps,
            # reward_environment=self.rng.choice(
            #     child_maps, replace=False, size=self.num_agents
            # ),
            model_type=agent_model_type,
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
            },
            agent_reporters={
                "choice": lambda a: a.last_choice,
                "reward": lambda a: a.last_reward + 0.5,
                "cumulative_reward": lambda a: a.total_reward + 0.5,
                "model_type": lambda a: a.model_type,
                "tau": lambda a: a.tau,
            },
        )

    def step(self):
        self.agents.do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. If explicit maps are passed in (for testing the code)
    n_agents = 5
    grid_size = 11
    # correlation matrix must include the parent + all children (n_agents + 1)
    R = np.array([
        [1.0, 0.0, -0.6, 0.6, 0.0],
        [0.0, 1.0, 0.2, -0.3, 0.0],
        [-0.6, 0.2, 1.0, 0.1, 0.0],
        [0.6, -0.3, 0.1, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    parent, children = make_parent_and_children_cholesky2(
        rng=None,
        grid_size=grid_size,
        n_children=n_agents,
        length_scale=2.0,
        corr_matrix=R,
    )
    m = SocialGPModel(
        n=n_agents,
        model_type="SG-ICM",
        network_type="directed_one_to_four",
        child_maps=[parent] + children,  # triggers the child_maps branch
        rho=np.array([1.0, 0.0, -0.6, 0.6, 0.0])
    )

    # 2. if a full correlation matrix is provided, use the new generator
    n_agents = 4
    grid_size = 11
    R = build_corr_matrix_option1()
    # R = build_corr_matrix_option2(eps=0.2)
    # R = build_corr_matrix_option3()
    m = SocialGPModel(
        n=n_agents,
        model_type="SG-ICM",
        network_type="directed_one_to_four",
        corr_matrix=R,  # triggers the corr_matrix branch
        tau_sampling=True,
        rho=R[0, :]
    )

    # 3. original scalar-correlation behavior
    # m = SocialGPModel(n=5, model_type="SG-ICM", network_type='directed_one_to_four')

    for _ in range(15):
        m.step()

    param_grid = {
        "n": [5],
        "model_type": ["SG-ICM"],  # "AS", "SG", "SG-ICM", "VS-F", "VS-CK"
        "length_scale_private": [1.11],
        "length_scale_social": [1.11],
        "observation_noise_private": [0.0001],
        "observation_noise_social":  [0.0001],
        "beta_private": [0.33],
        "beta_social":   [0.33],
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

    batch_results['tau_str'] = [str(l) for l in batch_results['tau'].to_list()]

    sns.lineplot(batch_results,
                 x="Step",
                 y="reward",
                 hue="model_type"  # "observation_noise_social" # "model_type"
                 )
    plt.show()

    batch_results.head()
