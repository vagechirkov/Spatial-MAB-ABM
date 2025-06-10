import itertools
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from matplotlib import gridspec

def _global_min_max(parent, children):
    # global min–max scaling
    flat_all = np.concatenate([parent.ravel()] + [c.ravel() for c in children])
    lo, hi = flat_all.min(), flat_all.max()
    parent = (parent - lo) / (hi - lo)
    children = [(c - lo) / (hi - lo) for c in children]
    return parent, children

def _min_max(arr):
    return  (arr - arr.min()) / (arr.max() - arr.min())


def make_parent_and_children_cholesky(
    rng,
    grid_size=11,
    n_children=4,
    length_scale=2.0,
    corr_parent=0.60,
    corr_children=0.60
):
    # Build the spatial kernel
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    Xstar = np.column_stack([x.ravel(), y.ravel()])  # (M, 2)
    kernel = RBF(length_scale)
    Sigma = kernel(Xstar)  # (M, M)
    M = Sigma.shape[0]  # M = grid_size^2

    # Desired (n+1) x (n+1) cross-surface correlation matrix R
    n_total = n_children + 1
    R = np.full((n_total, n_total), corr_children)
    R[0, 1:] = R[1:, 0] = corr_parent  # parent–child correlations
    np.fill_diagonal(R, 1.0)

    # Positive-definiteness check (needed for the Cholesky in the next step)
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 0:
        raise ValueError(
            f"The correlation matrix is not positive-definite. "
            f"Try smaller |corr| or fewer children.\n"
            f"Smallest eigenvalue: {eigvals.min():.3g}"
        )

    # Draw (n_total) independent samples  W ~ N(0, Sigma)
    W = rng.multivariate_normal(np.zeros(M), Sigma, size=n_total)  # (n_total, M)

    # Mix them with a Cholesky factor
    L = np.linalg.cholesky(R)  # (n_total, n_total)
    Y = L @ W  # (n_total, M)

    # Split, reshape, optionally min-max normalise
    parent = Y[0].reshape(grid_size, grid_size)
    children = [Y[i + 1].reshape(grid_size, grid_size) for i in range(n_children)]
    return parent, children


def make_parent_and_children_cholesky2(
    rng,
    grid_size=11,
    n_children=4,
    length_scale=2.0,
    corr_parent=0.60,
    corr_children=0.60
):
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    Xstar = np.column_stack([x.ravel(), y.ravel()])  # (M, 2)
    Sigma = RBF(length_scale)(Xstar)  # (M, M)
    # add a tiny jitter for numerical stability
    LSigma = np.linalg.cholesky(Sigma + 1e-10 * np.eye(Sigma.shape[0]))
    M = Sigma.shape[0]
    n_total = n_children + 1

    # Task-level correlation matrix R and its Cholesky LR
    R = np.full((n_total, n_total), corr_children)
    R[0, 1:] = R[1:, 0] = corr_parent
    np.fill_diagonal(R, 1.0)

    # Positive-definiteness check (needed for the Cholesky in the next step)
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 0:
        raise ValueError(
            f"The correlation matrix is not positive-definite. "
            f"Try smaller |corr| or fewer children.\n"
            f"Smallest eigenvalue: {eigvals.min():.3g}"
        )

    LR = np.linalg.cholesky(R)  # (n_total, n_total)
    z = rng.standard_normal((n_total, M))  # (n_total, M)
    Y = (LR @ z) @ LSigma.T  # same shape

    parent = Y[0].reshape(grid_size, grid_size)
    children = [Y[i + 1].reshape(grid_size, grid_size) for i in range(n_children)]
    return parent, children


def check_correlations(parent, children, corr_parent, corr_children, tol=0.05):
    p = parent.ravel()
    children = [c.ravel() for c in children]

    for c in children:
        if not np.isclose(np.corrcoef(p, c)[0, 1], corr_parent, atol=tol):
            return False

    for (c1, c2) in itertools.combinations(children, 2):
        if not np.isclose(np.corrcoef(c1, c2)[0, 1], corr_children, atol=tol):
            return False

    return True


def make_parent_and_children_corr(
        rng,
        grid_size=11,
        n_children=4,
        length_scale=2.0,
        corr_parent=0.60,
        corr_children=0.60
):
    # Build the spatial kernel
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    Xstar = np.column_stack([x.ravel(), y.ravel()])  # (M, 2)
    Sigma = RBF(length_scale)(Xstar)  # (M, M)
    M = Sigma.shape[0]

    parent_vec = rng.multivariate_normal(mean=np.zeros(M), cov=Sigma)  # (1, M)
    noise_mat = rng.multivariate_normal(mean=np.zeros(M), cov=Sigma, size=n_children)

    scale_p = corr_parent
    scale_n = np.sqrt(1.0 - corr_parent**2)

    children = [
        (scale_p * parent_vec + scale_n * noise_mat[i]).reshape(grid_size, grid_size)
        for i in range(n_children)
    ]
    parent = parent_vec.reshape(grid_size, grid_size)
    return parent, children

def _fix_corr(parent, child, corr_target, eps=1e-12):
    p = parent.ravel()
    c = child.ravel()

    p = p - p.mean()
    c = c - c.mean()

    # projection of c onto p
    proj = (p @ c) / (p @ p + eps) * p
    res = c - proj  # component orthogonal to p

    # rescale so that corr(p, new) = rho_target
    p_norm = np.linalg.norm(p) + eps
    res_norm = np.linalg.norm(res) + eps
    a = corr_target  # along parent
    b = np.sqrt(1 - corr_target ** 2) * p_norm / res_norm
    child_fixed = (a * p + b * res).reshape(parent.shape)
    return child_fixed

def plot_reward_environments(parent, children, cmap="viridis", figsize=None):
    n_children = len(children)
    if figsize is None:
        figsize = (4 * (n_children + 1), 4)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_children + 1, wspace=0.05)

    # parent
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(parent, cmap=cmap, origin="lower", interpolation="nearest", vmin=0, vmax=1)
    ax0.set_title("Parent")
    ax0.axis("off")

    # Colour-bar only once (parent)
    # cbar = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    # cbar.ax.set_ylabel("payoff", rotation=-90, va="bottom")

    # children
    for i, child in enumerate(children, 1):
        ax = fig.add_subplot(gs[i])
        ax.imshow(
            child,
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            vmin=0, vmax=1
            # vmin=im0.get_clim()[0],
            # vmax=im0.get_clim()[1],
        )
        ax.set_title(f"Child {i}")
        ax.axis("off")

    plt.show()


if __name__ == "__main__":
    random_state = 42
    # corr_PC, corr_CC = 0.4, 0.4
    corr_PC = corr_CC = 0.6

    rng = np.random.default_rng(random_state)

    n = 0
    no_luck = 0
    while n < 5:
        parent, kids = (
            # make_parent_and_children_corr
            make_parent_and_children_cholesky2(
                rng,
                grid_size=11,
                n_children=4,
                length_scale=2.0,
                corr_parent=corr_PC,
                corr_children=corr_CC
            )
        )
        kids = [_fix_corr(parent, ch, corr_CC) for ch in kids]

        parent, kids = _min_max(parent), [_min_max(k) for k in kids]

        if not check_correlations(parent, kids, corr_PC, corr_CC, tol=0.1):
            no_luck += 1
            continue

        n += 1
        print(no_luck)

        plot_reward_environments(parent, kids)

        print(np.corrcoef([parent.ravel()] + [k.ravel() for k in kids]))

        p = parent.ravel()
        c1, c2 = (kids[0].ravel(), kids[1].ravel())
        print("corr(P, C1) =", np.corrcoef(p, c1)[0, 1])
        print("corr(P, C2) =", np.corrcoef(p, c2)[0, 1])
        print("corr(C1, C2) =", np.corrcoef(c1, c2)[0, 1])

