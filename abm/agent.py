import numpy as np
from mesa.discrete_space import CellAgent
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from utils import ICMKernel, LMCKernel


def step_distance(x):
    if len(x) < 2:
        return 0
    choice = np.asarray(x[-1], dtype=float)
    previous_choice = np.asarray(x[-2], dtype=float)
    return float(np.linalg.norm(choice - previous_choice))


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

def asocial_generalization(
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        X_predict: np.ndarray,
        length_scale: float,
        observation_noise: float,
        beta: float,
        tau: float,
        random_state,
):
    gp_mean, gp_std = gp_base_generalization(
        X_obs, y_obs, X_predict, length_scale, np.ones(len(y_obs)) * observation_noise, random_state
    )
    value_ucb = gp_mean + beta * gp_std
    return np.exp(value_ucb / tau)

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
    subtract_max_value: bool = False,
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
    if subtract_max_value:
        value_ucb -= np.max(value_ucb)
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
    alpha: float,
    tau: float,
    random_state,
    value_shaping_type: str = "linear_weight"
) -> np.ndarray:
    assert len(X_obs_private) > 0
    assert len(X_obs_social) > 0
    assert value_shaping_type in ["naive", "linear_weight", "correlated_kalman"]

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

    value_final = value_ucb_private.copy()

    # Social GPs (one per neighbor)
    if value_shaping_type == "naive":
        value_ucb_social = ...
    elif value_shaping_type == "linear_weight":
        ucb_s_list = []
        for xs, ys in zip(X_obs_social, y_obs_social):
            gp_mean_s, gp_std_s = gp_base_generalization(
                xs,
                ys,
                X_predict,
                length_scale_social,
                np.ones(len(xs)) * observation_noise_social,
                random_state,
                )
            ucb_s_list.append(gp_mean_s + beta_social * gp_std_s)

        value_ucb_social = np.mean(np.vstack(ucb_s_list), axis=0)
        value_final = (1.0 - alpha) * value_ucb_private + alpha * value_ucb_social
    elif value_shaping_type == "correlated_kalman":
        ucb_s_list, w_priv_raw_list, w_soc_raw_list = [], [], []
        for xs, ys in zip(X_obs_social, y_obs_social):
            gp_mean_s, gp_std_s = gp_base_generalization(
                xs,
                ys,
                X_predict,
                length_scale_social,
                np.ones(len(xs)) * observation_noise_social,
                random_state,
                )
            ucb_s_list.append(gp_mean_s + beta_social * gp_std_s)

            rho_k = alpha
            # private-weight for this peer (variance-ratio + ρ_k)
            w_p_k = (gp_std_s**2 - rho_k * gp_std_p * gp_std_s) / (
                gp_std_p**2 + gp_std_s**2 - 2 * rho_k * gp_std_p * gp_std_s + 1e-12
            )
            w_p_k = np.clip(w_p_k, 0.0, 1.0)

            w_priv_raw_list.append(w_p_k)
            w_soc_raw_list.append(1.0 - w_p_k)

        w_priv_raw = np.mean(np.vstack(w_priv_raw_list), axis=0, keepdims=True)
        w_soc_raw = np.vstack(w_soc_raw_list)
        total = w_priv_raw + np.sum(w_soc_raw, axis=0, keepdims=True)
        w_priv = w_priv_raw / total
        w_soc = w_soc_raw / total

        value_final = w_priv * value_ucb_private + np.sum(
            w_soc * np.vstack(ucb_s_list), axis=0, keepdims=True
        )
        value_final = value_final.flatten()
    else:
        raise NotImplementedError(f"{value_shaping_type} is not implemented.")

    return np.exp(value_final / tau)  # unnormalised soft-max


def _stack_tasks(X_private, X_social_list):
    X_p = np.hstack([X_private, np.zeros((len(X_private), 1))])
    X_s = [
        np.hstack([Xs, np.full((len(Xs), 1), k + 1)])
        for k, Xs in enumerate(X_social_list)
    ]
    return np.vstack([X_p] + X_s)


def _stack_targets(y_private, y_social_list):
    return np.vstack([y_private] + y_social_list)


def value_fusion_icm(
    X_obs_private,
    y_obs_private,
    X_obs_social,
    y_obs_social,
    X_predict,
    length_scale_private: float,
    length_scale_social: float,
    length_scale_is_identical: bool,
    observation_noise_private,
    observation_noise_social,
    beta,
    rho,
    tau,
    random_state,
):
    X_all = _stack_tasks(X_obs_private, X_obs_social)
    Y_all = _stack_targets(y_obs_private, y_obs_social)

    if length_scale_is_identical:
        kernel = ICMKernel(length_scale=length_scale_private, rho=rho)
    else:
        kernel = LMCKernel(
            length_scale_private=length_scale_private,
            length_scale_social=length_scale_social,
            rho=rho,
        )

    observation_noise = np.hstack(
        [np.ones(len(y_obs_private)) * observation_noise_private]
        + [np.ones(len(y_soc)) * observation_noise_social for y_soc in y_obs_social]
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=observation_noise,
        optimizer=None,
        normalize_y=False,
        random_state=random_state,
    )
    gpr.fit(X_all, Y_all.ravel())

    X_star_priv = np.hstack([X_predict, np.zeros((len(X_predict), 1))])
    mu, std = gpr.predict(X_star_priv, return_std=True)

    ucb = mu.reshape(-1, 1) + beta * std.reshape(-1, 1)
    logits = np.exp(ucb / tau)
    return logits


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
        rho: float,
        length_scale_is_identical: bool = False,
    ):
        super().__init__(model)

        # graph node the agent occupies
        self.cell = cell

        # reward landscape specific to this agent
        self.reward_environment = reward_environment

        # hyperparameters
        self.length_scale_private = length_scale_private
        self.length_scale_social = length_scale_social
        self.length_scale_is_identical = length_scale_is_identical
        if length_scale_is_identical:
            assert np.abs(length_scale_private - length_scale_social) < 0.01, "Length scales are not identical"

        self.observation_noise_private = observation_noise_private
        self.observation_noise_social = observation_noise_social

        self.beta_private = beta_private
        self.beta_social = beta_social

        self.tau = tau
        self.rho = rho

        # memory buffers
        self.X_observations: list[tuple[int, int]] = []
        self.y_observations: list[float] = []

        # prediction grid
        self.meshgrid = np.meshgrid(
            range(reward_environment.shape[0]), range(reward_environment.shape[1])
        )
        self.meshgrid_flatten = np.array(self.meshgrid, dtype=np.int32).reshape(2, -1).T
        self.meshgrid_dict = {tuple(coord): i for i, coord in enumerate(self.meshgrid_flatten)}
        self.uniform_probs = np.ones(len(self.meshgrid_flatten)) / len(self.meshgrid_flatten)
        self.policy = self.uniform_probs.copy()

    @property
    def last_choice(self) -> tuple[int, int]:
        return self.X_observations[-1]

    @property
    def last_reward(self) -> float:
        return self.y_observations[-1]

    @property
    def total_reward(self) -> float:
        return np.sum(self.y_observations)

    @property
    def last_choice_distance_private(self) -> float:
        x = np.asarray(self.X_observations)
        if len(x) < 2:
            return 0.0
        return float(np.linalg.norm(x[-1] - x[-2], axis=-1))

    @property
    def last_choice_distance_social(self) -> float:
        X_soc, _ = self._gather_social_info()
        if len(X_soc) < 1 or len(X_soc[0]) < 1:
            return 0.0
        social_last_choices = np.array([_x_soc[-1] for _x_soc in X_soc])
        cur = np.asarray(self.X_observations[-1])
        return float(np.mean(np.linalg.norm(cur - social_last_choices, axis=-1)))

    @property
    def nearest_choice_distance_private(self) -> float:
        x = np.asarray(self.X_observations)
        if len(x) < 2:
            return 0.0
        return float(np.min(np.linalg.norm(x[-1] - x[:-1], axis=-1)))

    @property
    def avg_choice_distance_private(self) -> float:
        x = np.asarray(self.X_observations)
        if len(x) < 2:
            return 0.0
        return float(np.mean(np.linalg.norm(x[-1] - x[:-1], axis=-1)))

    @property
    def nearest_choice_distance_social(self) -> float:
        X_soc, _ = self._gather_social_info()
        if len(X_soc) < 1 or len(X_soc[0]) < 1:
            return 0.0
        social_choices = np.vstack(X_soc)
        cur = np.asarray(self.X_observations[-1])
        return float(np.min(np.linalg.norm(cur - social_choices, axis=-1)))

    @property
    def avg_choice_distance_social(self) -> float:
        X_soc, _ = self._gather_social_info()
        if len(X_soc) < 1 or len(X_soc[0]) < 1:
            return 0.0
        social_choices = np.vstack(X_soc)
        cur = np.asarray(self.X_observations[-1])
        return float(np.mean(np.linalg.norm(cur - social_choices, axis=-1)))

    @property
    def private_landscape_reconstruction_mse(self) -> float:
        if len(self.X_observations) < 1:
            return 0
        X_priv = np.array(self.X_observations)
        y_priv = np.array(self.y_observations).reshape(-1, 1)
        gp_mean_p, _ = gp_base_generalization(
            X_priv,
            y_priv,
            self.meshgrid_flatten,
            length_scale=2,
            observation_noise=np.ones(len(X_priv)) * 1e-10,
            rng=self.model.rng.__getstate__()
            )
        return float(np.square(self.reward_environment.T.ravel() - gp_mean_p).sum())

    @property
    def social_landscape_reconstruction_mse(self) -> float:
        neighbours = list(self.model.grid[self.cell.coordinate].neighborhood)
        mses = [self.private_landscape_reconstruction_mse] + [
            n.agents[0].private_landscape_reconstruction_mse for n in neighbours
        ]
        return float(np.mean(mses))

    @property
    def neg_log_likelihood(self) -> float:
        last_choice_is_random = self.model.random_choices[self.model.steps - 1]
        if (self.model.steps == 1) or last_choice_is_random:
            return 0.0
        return -np.log(self.policy[self.meshgrid_dict[self.last_choice]])

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

        if self.model.model_type == "SG":
            X_soc, y_soc = self._gather_social_info()
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
        elif self.model.model_type == "SG_fitting":
            X_soc = [s_c[:self.model.steps] for s_c in self.model.social_choices]
            y_soc = [s_r[:self.model.steps].reshape(-1, 1) for s_r in self.model.social_rewards]
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
                random_state=self.model.rng.__getstate__(),
                subtract_max_value=True
            )
        elif self.model.model_type == "VF-ICM":
            X_soc, y_soc = self._gather_social_info()
            logits = value_fusion_icm(
                X_priv,
                y_priv,
                X_soc,
                y_soc,
                self.meshgrid_flatten,
                length_scale_private=self.length_scale_private,
                length_scale_social=self.length_scale_social,
                length_scale_is_identical=self.length_scale_is_identical,
                observation_noise_private=self.observation_noise_private,
                observation_noise_social=self.observation_noise_social,
                beta=self.beta_private,
                # beta_private=self.beta_private,
                # beta_social=self.beta_social,
                rho=self.rho,  # rho values per social info
                tau=self.tau,
                random_state=self.model.rng.__getstate__()
            )
        elif self.model.model_type == "AS":
            logits = asocial_generalization(
                X_priv,
                y_priv,
                self.meshgrid_flatten,
                length_scale=self.length_scale_private,
                observation_noise=self.observation_noise_private,
                beta=self.beta_private,
                tau=self.tau,
                random_state=self.model.rng.__getstate__()
            )
        else:
            raise ValueError(f"Unknown model_type '{self.model.model_type}'")

        # sample next arm
        probs = logits.ravel()

        if "fitting" in self.model.model_type:
            # quick fix, consider adjusting gamma priors instead
            probs[probs / np.sum(probs) == 0] = 0.001 * np.sum(probs)
            probs[probs < 0] = 0.001 * np.sum(probs)
            probs = probs / np.sum(probs)

            self.policy = probs
            inx = self.model.steps - 1
            if inx == 1:
                # overwrite random choice
                self.X_observations[0] = self.model.individual_choices[0]
                self.y_observations[0] = self.model.individual_rewards[0]
            coord = tuple(self.model.individual_choices[inx])
            self.X_observations.append(coord)
            self.y_observations.append(self.model.individual_rewards[inx])
        else:
            probs /= probs.sum() + 1e-12
            self.policy = probs
            idx = self.model.rng.choice(len(self.policy), p=self.policy)
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
