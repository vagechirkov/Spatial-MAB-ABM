import numpy as np
from mesa.discrete_space import CellAgent
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel

from utils import ICMKernel, LMCKernel, _stack_targets, _stack_tasks


def gp_base_generalization(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_predict: np.ndarray,
    kernel: Kernel,
    observation_noise: np.ndarray | float,
    rng,
):
    """Fit a zero-mean GP and return μ, σ on the prediction grid."""
    gpr = GaussianProcessRegressor(
        kernel=kernel,
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
        X_obs,
        y_obs,
        X_predict,
        RBF(length_scale=length_scale),
        np.ones(len(y_obs)) * observation_noise,
        random_state,
    )
    value_ucb = gp_mean + beta * gp_std
    logits = value_ucb / tau
    logits = np.clip(logits, -40, 40)  # avoid overflow in exp
    return np.exp(logits)

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
        + [
            np.ones(len(y_soc)) * observation_noise_social + observation_noise_private
            for y_soc in y_obs_social
        ]
    )
    X_obs = np.vstack([X_obs_private] + X_obs_social)
    y_obs = np.vstack([y_obs_private] + y_obs_social)

    gp_mean, gp_std = gp_base_generalization(
        X_obs,
        y_obs,
        X_predict,
        RBF(length_scale=length_scale),
        observation_noise,
        random_state,
    )
    value_ucb = gp_mean + beta * gp_std
    if subtract_max_value:
        value_ucb -= np.max(value_ucb)
    logits = value_ucb / tau
    logits = np.clip(logits, -40, 40)  # avoid overflow in exp
    return np.exp(logits)  # soft-max logits (unnormalised)


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
    value_shaping_type: str = "F"
) -> np.ndarray:
    assert len(X_obs_private) > 0
    assert len(X_obs_social) > 0
    # N = Naive, F = Full linear weight, CK = Correlated Kalman
    assert value_shaping_type in ["N", "F", "CK"]

    # Private GP
    gp_mean_p, gp_std_p = gp_base_generalization(
        X_obs_private,
        y_obs_private,
        X_predict,
        RBF(length_scale=length_scale_private),
        np.ones(len(X_obs_private)) * observation_noise_private,
        random_state,
    )
    value_ucb_private = gp_mean_p + beta_private * gp_std_p

    value_final = value_ucb_private.copy()

    # Social GPs (one per neighbor)
    if value_shaping_type == "N":
        value_ucb_social = ...
    elif value_shaping_type == "F":
        ucb_s_list = []
        for xs, ys in zip(X_obs_social, y_obs_social):
            gp_mean_s, gp_std_s = gp_base_generalization(
                xs,
                ys,
                X_predict,
                RBF(length_scale=length_scale_social),
                np.ones(len(xs)) * observation_noise_social,
                random_state,
                )
            ucb_s_list.append(gp_mean_s + beta_social * gp_std_s)

        value_ucb_social = np.mean(np.vstack(ucb_s_list), axis=0)
        value_final = (1.0 - alpha) * value_ucb_private + alpha * value_ucb_social
    elif value_shaping_type == "CK":
        ucb_s_list, w_priv_raw_list, w_soc_raw_list = [], [], []
        for xs, ys in zip(X_obs_social, y_obs_social):
            gp_mean_s, gp_std_s = gp_base_generalization(
                xs,
                ys,
                X_predict,
                RBF(length_scale=length_scale_social),
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

    logits = value_final / tau
    logits = np.clip(logits, -40, 40)  # avoid overflow in exp
    return np.exp(logits)  # unnormalised soft-max


def social_generalization_icm(
    X_obs_private,
    y_obs_private,
    X_obs_social,
    y_obs_social,
    X_predict,
    length_scale_private: float,
    length_scale_social: float,
    observation_noise_private,
    observation_noise_social,
    beta,
    rho,
    tau,
    random_state,
    model="ICM",
    subtract_max_value=False,
):
    X_all = _stack_tasks(X_obs_private, X_obs_social)
    Y_all = _stack_targets(y_obs_private, y_obs_social)

    if model == "ICM":
        kernel = ICMKernel(length_scale=length_scale_private, rho=rho)
    elif model == "LCM":
        kernel = LMCKernel(
            length_scale_private=length_scale_private,
            length_scale_social=length_scale_social,
            rho=rho,
        )
    else:
        raise NotImplementedError(f"{model} is not implemented.")

    observation_noise = np.hstack(
        [np.ones(len(y_obs_private)) * observation_noise_private]
        + [np.ones(len(y_soc)) * observation_noise_social for y_soc in y_obs_social]
    )

    # make a prediction for private output channel only
    X_star_priv = np.hstack([X_predict, np.zeros((len(X_predict), 1))])
    gp_mean_p, gp_std_p = gp_base_generalization(
        X_all,
        Y_all.ravel(),
        X_star_priv,
        kernel,
        observation_noise,
        random_state,
    )

    ucb = gp_mean_p.reshape(-1, 1) + beta * gp_std_p.reshape(-1, 1)
    if subtract_max_value:
        ucb -= np.max(ucb)
    logits = ucb / tau
    logits = np.clip(logits, -40, 40)  # avoid overflow in exp
    return np.exp(logits)


class SocialGPAgent(CellAgent):
    """GP-based explorer living on a Network grid."""

    def __init__(
        self,
        model,
        cell,
        reward_environment: np.ndarray,
        model_type: str,
        length_scale_private: float,
        length_scale_social: float,
        observation_noise_private: float,
        observation_noise_social: float,
        beta_private: float,
        beta_social: float,
        tau: float,
        rho: float,
    ):
        super().__init__(model)

        # graph node the agent occupies
        self.cell = cell

        # model type
        self.model_type = model_type

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
        self.rho = np.array(rho).flatten()

        # memory buffers
        self.X_observations: list[tuple[int, int]] = []
        self.y_observations: list[float] = []

        # prediction grid
        self.meshgrid = np.meshgrid(
            range(reward_environment.shape[0]),
            range(reward_environment.shape[1])
        )
        self.meshgrid_flatten = np.array(self.meshgrid, dtype=np.int32).reshape(2, -1).T
        self.meshgrid_dict = {
            tuple(coord): i for i, coord in enumerate(self.meshgrid_flatten)
        }
        self.uniform_probs = np.ones(len(self.meshgrid_flatten)) / len(
            self.meshgrid_flatten
        )
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
    def neg_log_likelihood(self) -> float:
        # First choice is random by design, so we do not score it.
        if len(self.X_observations) < 2:
            return 0.0

        # Some experiments track whether the last choice was random at the model level.
        # Fall back to "not random" if the attribute is missing.
        random_choices = getattr(self.model, "random_choices", None)
        if random_choices is not None:
            try:
                if self.model.steps - 1 < len(random_choices) and random_choices[self.model.steps - 1]:
                    return 0.0
            except Exception:
                pass

        return -np.log(self.policy[self.meshgrid_dict[self.X_observations[-1]]])

    def _gather_social_info(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        neighbours = list(self.model.grid[self.cell.coordinate].neighborhood)
        #TODO: more this to the network generation script
        # neighbours = neighbours[: self.model.attention_budget]  # w = 4

        X_soc, y_soc = [], []
        # observe only the choices before the last step
        history_horizon = self.model.steps - 1

        for neighbour in neighbours:
            neighbor_agent = neighbour.agents[0]
            X_soc.append(np.array(neighbor_agent.X_observations[:history_horizon]))
            y_soc.append(np.array(neighbor_agent.y_observations[:history_horizon]).reshape(-1, 1))
        return X_soc, y_soc

    def _add_noise_to_reward(self, reward: float):
        added_noise = 0
        if hasattr(self.model, "reward_noise_sd") and (self.model.reward_noise_sd > 0):
            added_noise = self.rng.normal(0, self.model.reward_noise_sd)
        return reward + added_noise

    def _random_choice(self) -> None:
        idx = self.model.random.randint(0, len(self.meshgrid_flatten) - 1)
        coord = tuple(self.meshgrid_flatten[idx])
        reward = float(self.reward_environment[coord])
        reward = self._add_noise_to_reward(reward)
        self.X_observations.append(coord)
        self.y_observations.append(reward)

    def _make_choice(self):
        X_priv = np.array(self.X_observations)
        y_priv = np.array(self.y_observations).reshape(-1, 1)

        if self.model_type in ["SG-ICM", "SG-LCM"]:
            X_soc, y_soc = self._gather_social_info()
            logits = social_generalization_icm(
                X_priv,
                y_priv,
                X_soc,
                y_soc,
                self.meshgrid_flatten,
                length_scale_private=self.length_scale_private,
                length_scale_social=self.length_scale_social,
                observation_noise_private=self.observation_noise_private,
                observation_noise_social=self.observation_noise_social,
                beta=self.beta_private,
                rho=self.rho,
                tau=self.tau,
                random_state=self.model.rng.__getstate__(),
                model=self.model_type.split("-")[1],
                subtract_max_value=True
            )
        elif "SG" in self.model_type:

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
                random_state=self.model.rng.__getstate__(),
                subtract_max_value=True
            )
        elif self.model_type in ["VS-N", "VS-F", "VS-CK"]:
            X_soc, y_soc = self._gather_social_info()
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
                alpha=self.rho,
                tau=self.tau,
                random_state=self.model.rng.__getstate__(),
                value_shaping_type=self.model_type.split("-")[1]
            )
        elif self.model_type == "AS":
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
            raise ValueError(f"Unknown model_type '{self.model_type}'")

        probs = logits.ravel()
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs += 1e-12
        total = probs.sum()
        if (not np.isfinite(total)) or (total <= 0.0):
            probs = self.uniform_probs.copy()
        else:
            probs /= total
            if not np.isfinite(probs).all():
                probs = self.uniform_probs.copy()
        self.policy = probs

        idx = self.model.rng.choice(len(self.policy), p=self.policy)
        coord = tuple(self.meshgrid_flatten[idx])
        reward = float(self.reward_environment[coord])
        reward = self._add_noise_to_reward(reward)

        self.X_observations.append(coord)
        self.y_observations.append(reward)

    def step(self):
        # first choice is random
        if len(self.X_observations) == 0:
            self._random_choice()
            return

        self._make_choice()
