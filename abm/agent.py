
import numpy as np
from mesa.discrete_space import CellAgent
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def gp_base_generalization(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_predict: np.ndarray,
    length_scale: float,
    observation_noise: np.ndarray | float,
    rng,
):
    """Fit a zero-mean GP and return μ, σ on the prediction grid."""
    gpr = GaussianProcessRegressor(
        kernel=RBF(length_scale=length_scale),
        alpha=observation_noise,
        random_state=rng,
        optimizer=None,
        normalize_y=False,
    )
    gpr.fit(X_obs, y_obs)
    return gpr.predict(X_predict, return_std=True)


def social_generalization(
    X_obs_private: np.ndarray,
    y_obs_private: np.ndarray,
    X_obs_social: list[np.ndarray],
    y_obs_social: list[np.ndarray],
    X_predict: np.ndarray,
    length_scale: float,
    observation_noise_private: float,
    observation_noise_social: float,
    beta: float,
    tau: float,
    random_state,
) -> np.ndarray:
    """Original SG model from Witt et al., 2024."""
    assert len(X_obs_private) > 0
    assert len(X_obs_social) > 0

    observation_noise = np.hstack(
        [np.ones(len(y_obs_private)) * observation_noise_private]
        + [np.ones(len(y_soc)) * observation_noise_social for y_soc in y_obs_social]
    )
    X_obs = np.vstack([X_obs_private] + X_obs_social)
    y_obs = np.vstack([y_obs_private] + y_obs_social)

    gp_mean, gp_std = gp_base_generalization(
        X_obs, y_obs, X_predict, length_scale, observation_noise, random_state
    )
    value_ucb = gp_mean + beta * gp_std
    return np.exp(value_ucb / tau)  # soft-max logits (unnormalised)


def value_shaping(
    X_obs_private: np.ndarray,
    y_obs_private: np.ndarray,
    X_obs_social: list[np.ndarray],
    y_obs_social: list[np.ndarray],
    X_predict: np.ndarray,
    length_scale_private: float,
    length_scale_social: float,
    observation_noise_private: float,
    observation_noise_social: float,
    beta_private: float,
    beta_social: float,
    alpha: float,  # social weight (0 … private-only, 1 … social-only)
    tau: float,
    random_state,
) -> np.ndarray:
    """
    Value-shaping (different from the model presented in Witt et al., 2024).

    1. Fit **one GP per information source**:
         • private history with (length_scale_private, observation_noise_private)
         • every social peer with (length_scale_social, observation_noise_social)
    2. Compute a UCB value for each:
            v_i = μ_i + β_i σ_i
    3. Combine by simple convex combination:
            v_final = (1-α) * v_private + α * mean_i(v_social_i)
    4. Soft-max with temperature τ.
    """
    assert len(X_obs_private) > 0
    assert len(X_obs_social) > 0

    # Private GP
    gp_mean_p, gp_std_p = gp_base_generalization(
        X_obs_private,
        y_obs_private,
        X_predict,
        length_scale_private,
        np.ones(len(X_obs_private)) * observation_noise_private,
        random_state,
    )
    value_ucb_private = gp_mean_p + beta_private * gp_std_p

    # Social GPs (one per neighbor)
    if len(X_obs_social) == 0:  # socially isolated
        value_ucb_social = value_ucb_private.copy()
    else:
        social_ucbs = []
        for xs, ys in zip(X_obs_social, y_obs_social):
            gp_mean_s, gp_std_s = gp_base_generalization(
                xs,
                ys,
                X_predict,
                length_scale_social,
                np.ones(len(xs)) * observation_noise_social,
                random_state,
            )
            social_ucbs.append(gp_mean_s + beta_social * gp_std_s)
        value_ucb_social = np.mean(np.vstack(social_ucbs), axis=0)

    # Combine & soft-max
    value_final = (1.0 - alpha) * value_ucb_private + alpha * value_ucb_social
    return np.exp(value_final / tau)  # unnormalised soft-max


class SocialGPAgent(CellAgent):
    """GP-based explorer living on a Network grid."""

    def __init__(
        self,
        model,
        cell,
        reward_environment: np.ndarray,
        length_scale_private: float,
        length_scale_social: float,
        observation_noise_private: float,
        observation_noise_social: float,
        beta_private: float,
        beta_social: float,
        tau: float,
        alpha: float,
    ):
        super().__init__(model)

        # graph node the agent occupies
        self.cell = cell

        # reward landscape specific to this agent
        self.reward_environment = reward_environment

        # hyperparameters
        self.length_scale_private = length_scale_private
        self.length_scale_social = length_scale_social

        self.observation_noise_private = observation_noise_private
        self.observation_noise_social = observation_noise_social

        self.beta_private = beta_private
        self.beta_social = beta_social

        self.tau = tau
        self.alpha = alpha

        # memory buffers
        self.X_observations: list[tuple[int, int]] = []
        self.y_observations: list[float] = []

        # prediction grid
        self.meshgrid = np.meshgrid(
            range(reward_environment.shape[0]), range(reward_environment.shape[1])
        )
        self.meshgrid_flatten = np.array(self.meshgrid, dtype=np.int32).reshape(2, -1).T
        self.uniform_probs = np.ones(len(self.meshgrid_flatten)) / len(self.meshgrid_flatten)

    @property
    def last_choice(self) -> tuple[int, int]:
        return self.X_observations[-1]

    @property
    def last_reward(self) -> float:
        return self.y_observations[-1]

    @property
    def total_reward(self) -> float:
        return np.sum(self.y_observations)

    def _gather_social_info(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        neighbours = list(self.model.grid[self.cell.coordinate].neighborhood)
        #TODO: more this to the network generation script
        neighbours = neighbours[: self.model.attention_budget]  # w = 4

        X_soc, y_soc = [], []
        # observe only the choices before the last step
        history_horizon = self.model.steps - 1

        for neighbour in neighbours:
            neighbor_agent = neighbour.agents[0]
            X_soc.append(np.array(neighbor_agent.X_observations[:history_horizon]))
            y_soc.append(np.array(neighbor_agent.y_observations[:history_horizon]).reshape(-1, 1))
        return X_soc, y_soc

    def _random_choice(self) -> None:
        idx = self.model.rng.choice(len(self.uniform_probs), p=self.uniform_probs)
        coord = tuple(self.meshgrid_flatten[idx])
        reward = float(self.reward_environment[coord])
        self.X_observations.append(coord)
        self.y_observations.append(reward)

    def _make_choice(self):
        X_priv = np.array(self.X_observations)
        y_priv = np.array(self.y_observations).reshape(-1, 1)

        X_soc, y_soc = self._gather_social_info()

        if self.model.model_type == "SG":
            logits = social_generalization(
                X_priv,
                y_priv,
                X_soc,
                y_soc,
                self.meshgrid_flatten,
                length_scale=self.length_scale_private,
                observation_noise_private=self.observation_noise_private,
                observation_noise_social=self.observation_noise_social,
                beta=self.beta_private,
                tau=self.tau,
                random_state=self.model.rng.__getstate__()
            )
        elif self.model.model_type == "VS":
            logits = value_shaping(
                X_priv,
                y_priv,
                X_soc,
                y_soc,
                self.meshgrid_flatten,
                length_scale_private=self.length_scale_private,
                length_scale_social=self.length_scale_social,
                observation_noise_private=self.observation_noise_private,
                observation_noise_social=self.observation_noise_social,
                beta_private=self.beta_private,
                beta_social=self.beta_social,
                alpha=self.alpha,
                tau=self.tau,
                random_state=self.model.rng.__getstate__()
            )
        else:
            raise ValueError(f"Unknown model_type '{self.model.model_type}'")

        # sample next arm
        probs = logits.ravel()
        probs /= probs.sum() + 1e-12
        idx = self.model.rng.choice(len(probs), p=probs)
        coord = tuple(self.meshgrid_flatten[idx])

        reward = float(self.reward_environment[coord])
        self.X_observations.append(coord)
        self.y_observations.append(reward)

    def step(self):
        # first choice is random
        if len(self.X_observations) == 0:
            self._random_choice()
            return

        self._make_choice()
