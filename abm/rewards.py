import itertools
import warnings

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
    rng=None,
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
    if rng is None:
        W = np.radom.multivariate_normal(np.zeros(M), Sigma, size=n_total)
    else:
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
    corr_children=0.60,
    corr_matrix=None, # add correlation matrix
):
    """
    Generate one parent map + n_children child maps with a specified
    spatial covariance (RBF kernel) and a task-level correlation structure.

    Parameters
    ----------
    corr_matrix : np.ndarray or None
        Optional full task-level correlation matrix R of shape
        (n_total, n_total), where n_total = n_children + 1 (parent + children).
        If provided, it is used directly instead of corr_parent/corr_children.
        Must be symmetric with ones on the diagonal and positive-definite.
    """
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    Xstar = np.column_stack([x.ravel(), y.ravel()])  # (M, 2)
    Sigma = RBF(length_scale)(Xstar)  # (M, M)
    # add a tiny jitter for numerical stability
    LSigma = np.linalg.cholesky(Sigma + 1e-10 * np.eye(Sigma.shape[0]))
    M = Sigma.shape[0]

    # Task-level correlation matrix R
    if corr_matrix is not None:
        n_total = n_children
        n_children -= 1
        # Use the provided full correlation matrix
        R = np.asarray(corr_matrix, dtype=float)
        if R.shape != (n_total, n_total):
            raise ValueError(
                f"corr_matrix must have shape {(n_total, n_total)}, "
                f"but got {R.shape}."
            )
        # Symmetry check
        if not np.allclose(R, R.T, atol=1e-8):
            raise ValueError("corr_matrix must be symmetric.")
        # Diagonal ones check
        if not np.allclose(np.diag(R), 1.0, atol=1e-8):
            raise ValueError("corr_matrix must have ones on the diagonal.")
    else:
        n_total = n_children + 1
        # Original behavior: Task-level correlation matrix R and its Cholesky LR
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
    if rng is None:
        z = np.random.standard_normal((n_total, M))  # (n_total, M)
    else:
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

def check_correlations_matrix(parent, children, R_target, tol=0.1):
    """
    Check if the empirical correlation matrix of (parent + children) is close
    to the target task-level correlation matrix R_target, up to tol.

    R_target should be a (n_total, n_total) array with ones on the diagonal.
    """
    flats = [parent.ravel()] + [c.ravel() for c in children]
    C = np.corrcoef(flats)  # (n_total, n_total)

    if C.shape != R_target.shape:
        raise ValueError(
            f"Shape mismatch: empirical C has shape {C.shape}, "
            f"but R_target has shape {R_target.shape}."
        )

    # We only care about off-diagonal entries; diagonal should be 1 anyway
    i, j = np.triu_indices(R_target.shape[0], k=1)
    diffs = np.abs(C[i, j] - R_target[i, j])
    return np.all(diffs <= tol)

def sample_children_with_corr(
        rng: np.random.Generator | None,
        n_children: int,
        length_scale: float,
        rho_parent_child: float,
        rho_child_child: float,
        tol: float = 0.10,
        max_tries: int = 50,
        grid_size=11,
):
    """
    Draw (parent, children) reward maps until every pair of children
    correlates with |corr - rho_child_child| ≤ tol.

    The parent’s correlation is *not* checked (specification).
    """
    for _ in range(max_tries):
        parent, children = make_parent_and_children_cholesky2(
            rng=rng,
            grid_size=grid_size,
            n_children=n_children,
            length_scale=length_scale,
            corr_parent=rho_parent_child,
            corr_children=rho_child_child,
        )

        # pair-wise child correlations
        flats = [c.ravel() for c in children]
        C = np.corrcoef(flats)
        ok = np.all(
            np.abs(C[np.triu_indices(n_children, k=1)] - rho_child_child) <= tol
        )
        if ok:
            return _min_max(parent), [_min_max(c) for c in children]

    warnings.warn(
        "Could not sample child maps with desired correlations within "
        f"{max_tries} attempts (tol={tol}). Using the last draw."
    )

    return _min_max(parent), [_min_max(c) for c in children]


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

def build_corr_matrix_option1():
    """
    Option 1: AS–AS correlations (B–C, B–D, C–D) = 0.

    A = parent (index 0)
    B, C, D = children (indices 1, 2, 3)
    """
    corr_AB = 0.6   # r(A,B)
    corr_AC = 0.0  # r(A,C)
    corr_AD = -0.6   # r(A,D)

    corr_BC = 0.0   # r(B,C)
    corr_BD = 0.0   # r(B,D)
    corr_CD = 0.0   # r(C,D)

    R = np.array([
        [1.0,      corr_AB, corr_AC, corr_AD],
        [corr_AB,  1.0,     corr_BC, corr_BD],
        [corr_AC,  corr_BC, 1.0,     corr_CD],
        [corr_AD,  corr_BD, corr_CD, 1.0    ],
    ])
    return R


def build_corr_matrix_option2(eps=0.2, max_tries=20):
    """
    Option 2: AS–AS correlations (B–C, B–D, C–D) random in [-eps, eps],
    so they are ~0 in expectation, but R is enforced to be PD.
    """
    corr_AB = 0.0
    corr_AC = -0.6
    corr_AD = 0.6

    for _ in range(max_tries):
        corr_BC = np.random.uniform(-eps, eps)
        corr_BD = np.random.uniform(-eps, eps)
        corr_CD = np.random.uniform(-eps, eps)

        R = np.array([
            [1.0,      corr_AB, corr_AC, corr_AD],
            [corr_AB,  1.0,     corr_BC, corr_BD],
            [corr_AC,  corr_BC, 1.0,     corr_CD],
            [corr_AD,  corr_BD, corr_CD, 1.0    ],
        ])

        eigvals = np.linalg.eigvalsh(R)
        if eigvals.min() > 0:
            return R

    raise RuntimeError(
        f"Could not sample a positive-definite R in {max_tries} attempts, try smaller eps."
    )


def build_corr_matrix_option3():
    """
    Option 3: fully custom AS–AS correlations.
    """
    corr_AB = 0.0
    corr_AC = -0.6
    corr_AD = 0.6

    # customize correlations for AS agnets
    corr_BC = 0.2
    corr_BD = -0.3
    corr_CD = 0.1

    R = np.array([
        [1.0,      corr_AB, corr_AC, corr_AD],
        [corr_AB,  1.0,     corr_BC, corr_BD],
        [corr_AC,  corr_BC, 1.0,     corr_CD],
        [corr_AD,  corr_BD, corr_CD, 1.0    ],
    ])

    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() <= 0:
        raise ValueError(
            f"Custom R is not positive-definite; smallest eigenvalue={eigvals.min():.3g}"
        )

    return R


def generate_env(env_seed, rho, corr_matrix, grid_size=11, max_tries=10000, tol: float = 0.10):
    """
    Generate environments with target correlation (rho) between parent and child maps.

    This function ensures:
    1. All maps (parent and children) are normalized to the same scale [0, 1]
    2. Correlation between maps matches the target rho within tolerance
    3. A warning is issued if the desired correlation cannot be achieved
    """
    n_children = corr_matrix.shape[0] - 1
    rng = np.random.default_rng(env_seed)
    best_deviation = float('inf')
    best_parent, best_child_maps = None, None

    for attempt in range(max_tries):
        parent, child_maps = make_parent_and_children_cholesky2(
            rng=rng,
            grid_size=grid_size,
            n_children=n_children + 1,
            length_scale=2.0,
            corr_matrix=corr_matrix,
        )
        flats = [c.ravel() for c in [parent] + child_maps]
        C = np.corrcoef(flats)

        deviation = np.abs(C[0, :] - rho)
        max_deviation = deviation.max()

        # Keep track of the best attempt
        if max_deviation < best_deviation:
            best_deviation = max_deviation
            best_parent = parent
            best_child_maps = child_maps

        ok = np.all(deviation <= tol)
        if ok:
            break

    # Warn if we didn't meet the tolerance
    if best_deviation > tol:
        warnings.warn(
            f"Could not generate environment with desired correlation within "
            f"{max_tries} attempts (tol={tol:.3f}). Best max deviation: {best_deviation:.4f}."
        )

    parent_normalized = _min_max(best_parent)
    child_maps_normalized = [_min_max(c) for c in best_child_maps]

    return parent_normalized, child_maps_normalized


if __name__ == "__main__":
    random_state = 42
    # corr_PC, corr_CC = 0.4, 0.4
    corr_PC = corr_CC = 0.6  

    rng = np.random.default_rng(random_state)

    # Choose which correlation-matrix option
    R = build_corr_matrix_option1()
    # R = build_corr_matrix_option2(eps=0.2)
    # R = build_corr_matrix_option3()

    n = 0
    no_luck = 0
    while n < 5:
        parent, kids = make_parent_and_children_cholesky2(
            rng,
            grid_size=11,
            n_children=3,      
            length_scale=2.0,
            corr_parent=corr_PC,   # ignored when corr_matrix is provided
            corr_children=corr_CC, # ignored when corr_matrix is provided
            corr_matrix=R,         # <-- new: full task correlation matrix
        )

        parent, kids = _min_max(parent), [_min_max(k) for k in kids]

        # New: check full correlation matrix instead of scalar corr_PC/corr_CC
        if not check_correlations_matrix(parent, kids, R, tol=0.1):
            no_luck += 1
            continue

        n += 1
        print("Number of failed attempts so far:", no_luck)

        plot_reward_environments(parent, kids)

        flats = [parent.ravel()] + [k.ravel() for k in kids]
        C = np.corrcoef(flats)
        print("Target R:")
        print(R)
        print("Empirical C:")
        print(C)

        p = parent.ravel()
        c1, c2 = (kids[0].ravel(), kids[1].ravel())
        print("corr(P, C1) =", np.corrcoef(p, c1)[0, 1])
        print("corr(P, C2) =", np.corrcoef(p, c2)[0, 1])
        print("corr(C1, C2) =", np.corrcoef(c1, c2)[0, 1])