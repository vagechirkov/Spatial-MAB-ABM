import math

import numpy as np


class RunningStats:
    """Online mean/variance via Welford's algorithm."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    @property
    def var(self):
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2


class BaseRhoUpdater:
    def __init__(self, *, rho_init: float, rho_max: float, sigma_zeta: float):
        self.rho_init = float(rho_init)
        self.rho_max = float(rho_max)
        self.sigma_zeta = float(sigma_zeta)
        self.rho_by_id: dict[int, float] = {}
        self.stats_m: dict[int, RunningStats] = {}
        self.stats_r: dict[int, RunningStats] = {}

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
        if stats.count < 2:
            return 0.0
        denom = math.sqrt(stats.var + self.sigma_zeta)
        if (not math.isfinite(denom)) or denom <= 0.0:
            return 0.0
        return (value - stats.mean) / denom

    def _compute_z(
        self, mean_pred: float, reward_obs: float, stats_m: RunningStats, stats_r: RunningStats
    ) -> float:
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
    def __init__(
        self,
        *,
        rho_init: float,
        rho_max: float,
        sigma_zeta: float,
        observation_noise: float,
    ):
        super().__init__(rho_init=rho_init, rho_max=rho_max, sigma_zeta=sigma_zeta)
        self.observation_noise = float(observation_noise)

    def update(
        self,
        neighbor_ids: list[int],
        mean_list: list[np.ndarray],
        var_list: list[np.ndarray],
        reward_list: list[np.ndarray],
    ) -> None:
        self.ensure_neighbors(neighbor_ids)

        for neighbor_id, means, vars_, rewards in zip(
            neighbor_ids, mean_list, var_list, reward_list
        ):
            stats_m = self.stats_m[neighbor_id]
            stats_r = self.stats_r[neighbor_id]
            rho = self.rho_by_id[neighbor_id]

            for mean_pred, var_pred, reward_obs in zip(means, vars_, rewards):
                z = self._compute_z(mean_pred, reward_obs, stats_m, stats_r)
                denom = var_pred + self.observation_noise + self.sigma_zeta
                alpha = var_pred / denom if denom > 0.0 else 0.0
                rho = rho + alpha * (z - rho)
                rho = float(np.clip(rho, -self.rho_max, self.rho_max))

                stats_m.update(float(mean_pred))
                stats_r.update(float(reward_obs))

            self.rho_by_id[neighbor_id] = rho


class TrustRWUpdater(BaseRhoUpdater):
    def __init__(
        self,
        *,
        rho_init: float,
        rho_max: float,
        sigma_zeta: float,
        eta: float,
    ):
        super().__init__(rho_init=rho_init, rho_max=rho_max, sigma_zeta=sigma_zeta)
        self.eta = float(eta)
        self.q_by_id: dict[int, float] = {}

    def ensure_neighbors(self, neighbor_ids: list[int]) -> None:
        super().ensure_neighbors(neighbor_ids)
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.q_by_id:
                continue
            if self.rho_max <= 0:
                q_init = 0.5
            else:
                q_init = (self.rho_init / self.rho_max + 1.0) / 2.0
            self.q_by_id[neighbor_id] = float(np.clip(q_init, 0.0, 1.0))

    def update(
        self,
        neighbor_ids: list[int],
        mean_list: list[np.ndarray],
        var_list: list[np.ndarray],
        reward_list: list[np.ndarray],
    ) -> None:
        self.ensure_neighbors(neighbor_ids)

        for neighbor_id, means, rewards in zip(neighbor_ids, mean_list, reward_list):
            stats_m = self.stats_m[neighbor_id]
            stats_r = self.stats_r[neighbor_id]
            q = self.q_by_id[neighbor_id]

            for mean_pred, reward_obs in zip(means, rewards):
                z = self._compute_z(mean_pred, reward_obs, stats_m, stats_r)
                o = 0.5 * (math.tanh(z) + 1.0)
                q = q + self.eta * (o - q)
                q = float(np.clip(q, 0.0, 1.0))

                stats_m.update(float(mean_pred))
                stats_r.update(float(reward_obs))

            self.q_by_id[neighbor_id] = q
            self.rho_by_id[neighbor_id] = (2.0 * q - 1.0) * self.rho_max
