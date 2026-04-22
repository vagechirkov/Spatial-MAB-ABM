import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import Kernel, RBF
from statsmodels.stats.weightstats import DescrStatsW


def plot_summary(results):
    columns = results[0].columns
    fig, axes = plt.subplots(1, len(columns), figsize=(4*len(columns), 5))  # 1 row, 5 columns
    for result in results:


        for idx, col in enumerate(columns):
            axes[idx].plot(result[col])
            axes[idx].set_title(col.replace('_', ' '))  # line break for readability
            axes[idx].set_xlabel('Index')
            axes[idx].set_ylabel(col)

    plt.tight_layout()
    plt.show()


def robust_weighted_correlation(x, y, x_err, y_err):
    """
    Calculates Pearson correlation weighted by inverse error variance.
    Robust to high noise because noisy points get near-zero weight.
    """
    if len(x) < 2:
        return 0.0

    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0

    # Calculate weights: High error = Low weight
    # We combine x_err and y_err into a single "unreliability" metric
    weights = 1.0 / (x_err**2 + y_err**2 + 1e-9)

    # Normalize weights (optional but good for stability)
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0
    weights /= w_sum

    # Stack data
    data = np.stack([x, y], axis=1)

    # Calculate weighted correlation
    weighted_stats = DescrStatsW(data, weights=weights)

    return weighted_stats.corrcoef[0, 1]


class ICMKernel(Kernel):
    """Intrinsic Coregionalization Model kernel"""

    def __init__(self, length_scale=1.0, rho=0.5):
        self.length_scale = length_scale
        self.rho = rho
        self.base = RBF(length_scale=length_scale)

    def __call__(self, X, Y=None, eval_gradient=False):
        X, Y = np.atleast_2d(X), X if Y is None else np.atleast_2d(Y)
        X_feat, X_out = X[:, :-1], X[:, -1].astype(int)
        Y_feat, Y_out = Y[:, :-1], Y[:, -1].astype(int)

        Kxy = self.base(X_feat, Y_feat, eval_gradient=eval_gradient)

        if np.array(self.rho).size == 1:
            # Coregionalization matrix (all off-diags = rho, diags = 1)
            Bxy = self.rho * (X_out[:, None] != Y_out[None, :]) + (
                X_out[:, None] == Y_out[None, :]
            )
        else:
            # Task Correlation Matrix (Size T x T)
            rho_vec = np.array(self.rho)
            n_tasks = len(rho_vec)

            # Start with Identity (1s on diagonal, 0s elsewhere)
            B_task = np.eye(n_tasks)

            # Apply the "Hub" logic: Row 0 and Col 0 take values from rho
            B_task[0, :] = rho_vec
            B_task[:, 0] = rho_vec

            # Project onto the full sample size
            Bxy = B_task[X_out[:, None], Y_out[None, :]]

            # Alternative (slow) solution:
            # Bxy = 0.0 * (X_out[:, None] != Y_out[None, :]) + (
            #         X_out[:, None] == Y_out[None, :]
            # )
            #
            # for i, _rho in enumerate(np.array(self.rho)):
            #     rows_task_0 = (X_out == 0)
            #     cols_task_1 = (Y_out == i)
            #
            #     Bxy[np.ix_(rows_task_0, cols_task_1)] = _rho
            #
            #     rows_task_i = (X_out == i)
            #     cols_task_0 = (Y_out == 0)
            #     Bxy[np.ix_(rows_task_i, cols_task_0)] = _rho

        K = Bxy * Kxy

        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True


class LMCKernel(Kernel):
    def __init__(self, length_scale_private=1.0, length_scale_social=1.0, rho=0.5):
        self.length_scale_private = length_scale_private
        self.length_scale_social = length_scale_social
        self.rho = float(rho)
        self.rbf_private = RBF(length_scale=length_scale_private)
        self.rbf_social = RBF(length_scale=length_scale_social)

    def _mixing(self, out_idx):
        # Returns mixing weights [w_private, w_social] for a given output index
        if out_idx == 0:
            return np.array([1.0, 0.0])   # private output: only private GP
        else:
            return np.array([self.rho, 1.0])  # social outputs: mix of private & social GPs

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        X_feat, X_out = X[:, :-1], X[:, -1].astype(int)
        Y_feat, Y_out = Y[:, :-1], Y[:, -1].astype(int)

        # Build all mixing weights (shape: n_points x 2)
        W_X = np.array([self._mixing(i) for i in X_out])
        W_Y = np.array([self._mixing(j) for j in Y_out])

        # Two RBF kernel matrices
        K_priv = self.rbf_private(X_feat, Y_feat)
        K_soc = self.rbf_social(X_feat, Y_feat)

        # Compose LMC: sum_q w_iq * w_jq * K_q(x_i, x_j)
        K = (W_X[:, [0]] * W_Y[:, [0]].T) * K_priv + (W_X[:, [1]] * W_Y[:, [1]].T) * K_soc
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True


class MultiOutputICMKernel(Kernel):
    """ICM kernel: output 0=private, output 1...N=social (all share one social length scale)."""
    def __init__(self, length_scale_private=1.0, length_scale_social=1.0, rho=0.5):
        self.length_scale_private = length_scale_private
        self.length_scale_social = length_scale_social
        self.rho = rho

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        X_feat, X_out = X[:, :-1], X[:, -1].astype(int)
        Y_feat, Y_out = Y[:, :-1], Y[:, -1].astype(int)

        n_outputs = int(max(np.max(X_out), np.max(Y_out)) + 1)
        rho_min = -1/(n_outputs-1)
        if not (rho_min < self.rho < 1):
            raise ValueError(f"rho={self.rho} not in ({rho_min:.3f}, 1) for n_outputs={n_outputs}")

        dists = np.sum((X_feat[:, None, :] - Y_feat[None, :, :]) ** 2, axis=2)
        nX, nY = X.shape[0], Y.shape[0]
        L_mat = np.empty((nX, nY))

        # Block-wise fill
        mask_pp = (X_out[:, None] == 0) & (Y_out[None, :] == 0)
        mask_ss = (X_out[:, None] >= 1) & (Y_out[None, :] >= 1)
        mask_ps = (X_out[:, None] == 0) & (Y_out[None, :] >= 1)
        mask_sp = (X_out[:, None] >= 1) & (Y_out[None, :] == 0)

        L_mat[mask_pp] = self.length_scale_private
        L_mat[mask_ss] = self.length_scale_social
        L_mat[mask_ps] = np.sqrt(self.length_scale_private * self.length_scale_social)
        L_mat[mask_sp] = np.sqrt(self.length_scale_private * self.length_scale_social)

        K_base = np.exp(-0.5 * dists / (L_mat ** 2))
        same_output = X_out[:, None] == Y_out[None, :]
        K = np.where(same_output, K_base, self.rho * K_base)
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True


def _stack_tasks(X_private, X_social_list):
    X_p = np.hstack([X_private, np.zeros((len(X_private), 1))])
    X_s = [
        np.hstack([Xs, np.full((len(Xs), 1), k + 1)])
        for k, Xs in enumerate(X_social_list)
    ]
    return np.vstack([X_p] + X_s)


def _stack_targets(y_private, y_social_list):
    return np.vstack([y_private] + y_social_list)
