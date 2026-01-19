import math

import numpy as np


class RunningStats:
    r"""
    Online mean/variance via Welford's algorithm.

    Used to compute running estimates of mean and variance for standardization:
    - \bar{m}^i_{1:t} and v(m_{1:t}^j) for predicted means
    - \bar{r}^j_{1:t} and v(r_{1:t}^j) for observed rewards

    As mentioned in the manuscript:
    "which can be computed using Welford's algorithm v(r)_t = \frac{1}{t-1}\sum_k^t (r_k - \bar{r})^2"
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0  # Running mean: \bar{m} or \bar{r}
        self._m2 = 0.0   # Running sum of squared differences

    @property
    def var(self):
        r"""Returns sample variance v(·) = \frac{1}{t-1}\sum_k^t (x_k - \bar{x})^2"""
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    def update(self, value: float) -> None:
        """Update running statistics with new observation"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2


class BaseRhoUpdater:
    r"""
    Base class for learning trust (rho) via trial-by-trial updates.

    Implements the core trust learning mechanism from the manuscript:
    \hat{\rho}_{t+1}^j = \hat{\rho}_t^j + \alpha_t^j (z_t^j - \hat{\rho}_t^j)  [Eq. rho_update]

    where z_t^j is the correlation evidence computed from standardized predictions
    and observations.
    """
    def __init__(self, *, rho_init: float, rho_max: float, sigma_zeta: float):
        self.rho_init = float(rho_init)  # Initial trust value
        self.rho_max = float(rho_max)    # Maximum correlation magnitude
        self.sigma_zeta = float(sigma_zeta)  # Noise term \sigma^2_\zeta for inference errors
        self.rho_by_id: dict[int, float] = {}  # Current \hat{\rho}^j for each neighbor j
        self.stats_m: dict[int, RunningStats] = {}  # Running stats for predicted means m(x)
        self.stats_r: dict[int, RunningStats] = {}  # Running stats for observed rewards r

    def ensure_neighbors(self, neighbor_ids: list[int]) -> None:
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.rho_by_id:
                continue
            self.rho_by_id[neighbor_id] = self.rho_init
            self.stats_m[neighbor_id] = RunningStats()
            self.stats_r[neighbor_id] = RunningStats()

    def _project_rho_vector(self, rho_vec: np.ndarray, max_norm: float = 0.99) -> np.ndarray:
        if len(rho_vec) <= 1:
            return rho_vec
        social = rho_vec[1:]
        norm = np.linalg.norm(social)
        if norm >= max_norm and norm > 0:
            rho_vec = rho_vec.copy()
            rho_vec[1:] = social * (max_norm / norm)
        return rho_vec

    def get_rho_vector(self, neighbor_ids: list[int]) -> np.ndarray:
        self.ensure_neighbors(neighbor_ids)
        rho_vec = np.array(
            [1.0] + [self.rho_by_id[neighbor_id] for neighbor_id in neighbor_ids],
            dtype=float,
        )
        return self._project_rho_vector(rho_vec)

    def _standardize(self, value: float, stats: RunningStats) -> float:
        r"""
        Standardize a value to zero mean and unit variance.

        Implements the standardization equations from the manuscript:
        \tilde{m}(x_t^j) = \frac{m(x_t^j) - \bar{m}^i_{1:t}}{\sqrt{v(m_{1:t}^j) + \sigma^2_\zeta}}
        \tilde{r}_t^j = \frac{r_t^j - \bar{r}^j_{1:t}}{\sqrt{v(r_{1:t}^j) + \sigma^2_\zeta}}

        Returns 0 if insufficient data or numerical issues.
        """
        if stats.count < 2:
            return 0.0
        denom = math.sqrt(stats.var + self.sigma_zeta)
        if (not math.isfinite(denom)) or denom <= 0.0:
            return 0.0
        return (value - stats.mean) / denom

    def _compute_z(
        self, mean_pred: float, reward_obs: float, stats_m: RunningStats, stats_r: RunningStats
    ) -> float:
        """
        Compute trial-wise correlation evidence z_t^j.

        From the manuscript:
        z_t^j = \tilde{m}(x_t^j) · \tilde{r}_t^j

        "Intuitively, z_t^j captures the degree of alignment between the posterior
        predicted mean m(x_t^j) and the observed reward r_t^j."

        When both are high or low (same sign) → increases correlation estimate
        When mismatched (different signs) → decreases correlation estimate

        Returns: correlation evidence clipped to [-1, 1]
        """
        m_tilde = self._standardize(mean_pred, stats_m)
        r_tilde = self._standardize(reward_obs, stats_r)
        z = m_tilde * r_tilde
        if not math.isfinite(z):
            return 0.0
        return float(np.clip(z, -1.0, 1.0))

    def update(
        self,
        neighbor_ids: list[int],
        mean_list: list[np.ndarray],
        var_list: list[np.ndarray],
        reward_list: list[np.ndarray],
    ) -> None:
        raise NotImplementedError


class RhoKalmanUpdater(BaseRhoUpdater):
    r"""
    Trust learning with adaptive (Kalman-like) learning rate.

    Implements the main trust learning model from the manuscript:

    \hat{\rho}_{t+1}^j = \hat{\rho}_t^j + \alpha_t^j (z_t^j - \hat{\rho}_t^j)

    where the learning rate is dynamically computed as:

    \alpha_t^j = \frac{v(x_t^j)}{v(x_t^j) + \sigma^2_\epsilon + \sigma^2_\zeta}

    "in a similar spirit to the Kalman gain" - adapts based on prediction uncertainty.
    """
    def __init__(
        self,
        *,
        rho_init: float,
        rho_max: float,
        sigma_zeta: float,
        observation_noise: float,
        learning_rate_type: str = "kalman",  # "kalman", "fixed", "dampened"
        learning_rate_value: float = 0.1,  # Used for "fixed" or dampening factor
    ):
        super().__init__(rho_init=rho_init, rho_max=rho_max, sigma_zeta=sigma_zeta)
        self.observation_noise = float(observation_noise)  # \sigma^2_\epsilon
        self.learning_rate_type = learning_rate_type
        self.learning_rate_value = float(learning_rate_value)
        self._last_update_info = {}  # For tracking internal variables

    def update(
        self,
        neighbor_ids: list[int],
        mean_list: list[np.ndarray],
        var_list: list[np.ndarray],
        reward_list: list[np.ndarray],
    ) -> None:
        r"""
        Update rho estimates for each neighbor based on new observations.

        For each new observation from neighbor j at location x_t^j:
        1. Compute z_t^j (correlation evidence)
        2. Compute \alpha_t^j (adaptive learning rate based on prediction variance)
        3. Update \hat{\rho}^j using: \hat{\rho}_{t+1}^j = \hat{\rho}_t^j + \alpha_t^j (z_t^j - \hat{\rho}_t^j)
        4. Update running statistics for future standardization
        """
        self.ensure_neighbors(neighbor_ids)
        self._last_update_info = {}  # Clear previous tracking info

        for neighbor_id, means, vars_, rewards in zip(
            neighbor_ids, mean_list, var_list, reward_list
        ):
            stats_m = self.stats_m[neighbor_id]
            stats_r = self.stats_r[neighbor_id]
            rho = self.rho_by_id[neighbor_id]

            for mean_pred, var_pred, reward_obs in zip(means, vars_, rewards):
                # Compute standardized values
                m_tilde = self._standardize(mean_pred, stats_m)
                r_tilde = self._standardize(reward_obs, stats_r)

                # Compute correlation evidence z_t^j
                z = self._compute_z(mean_pred, reward_obs, stats_m, stats_r)

                # Compute learning rate based on type
                if self.learning_rate_type == "fixed":
                    # Fixed learning rate
                    alpha = self.learning_rate_value
                elif self.learning_rate_type == "dampened":
                    # Dampened Kalman: multiply by dampening factor
                    denom = var_pred + self.observation_noise + self.sigma_zeta
                    alpha_kalman = var_pred / denom if denom > 0.0 else 0.0
                    alpha = alpha_kalman * self.learning_rate_value
                else:  # "kalman" (default)
                    # Adaptive learning rate: \alpha_t^j = v(x) / (v(x) + \sigma^2_\epsilon + \sigma^2_\zeta)
                    denom = var_pred + self.observation_noise + self.sigma_zeta
                    alpha = var_pred / denom if denom > 0.0 else 0.0

                # Store tracking info (last observation for this neighbor)
                self._last_update_info[neighbor_id] = {
                    'z_t': float(z),
                    'm_pred': float(mean_pred),
                    'v_pred': float(var_pred),
                    'r_obs': float(reward_obs),
                    'alpha': float(alpha),
                    'm_tilde': float(m_tilde),
                    'r_tilde': float(r_tilde),
                }

                # Update trust: \hat{\rho}_{t+1}^j = \hat{\rho}_t^j + \alpha_t^j (z_t^j - \hat{\rho}_t^j)
                rho = rho + alpha * (z - rho)
                rho = float(np.clip(rho, -self.rho_max, self.rho_max))

                # Update running statistics for next trial's standardization
                stats_m.update(float(mean_pred))
                stats_r.update(float(reward_obs))

            self.rho_by_id[neighbor_id] = rho
