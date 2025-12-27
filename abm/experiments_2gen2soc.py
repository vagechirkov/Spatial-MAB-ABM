import argparse
from rewards import (
    sample_children_with_corr,
    make_parent_and_children_cholesky2,
    build_corr_matrix_option1,
    build_corr_matrix_option2,
    build_corr_matrix_option3,
    _min_max,
)

def _build_corr_matrix(option: str, eps: float):
    if option == "option1":
        return build_corr_matrix_option1()
    if option == "option2":
        return build_corr_matrix_option2(eps=eps)
    if option == "option3":
        return build_corr_matrix_option3()
    return None


def run_simulations(_model, n_samples, corr_matrix):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",
                        type=int, default=2_000, help="Number of samples to draw from the prior")
    parser.add_argument("--model", type=str, default="SG-ICM")
    parser.add_argument(
        "--corr_matrix_option",
        type=str,
        default="none",
        choices=["none", "option1", "option2", "option3"],
        help="Select a predefined correlation matrix; 'none' keeps scalar correlation sampling.",
    )
    parser.add_argument(
        "--corr_eps",
        type=float,
        default=0.2,
        help="Epsilon for option2 random AS-AS correlations.",
    )
    args = parser.parse_args()

    corr_matrix = _build_corr_matrix(args.corr_matrix_option, args.corr_eps)

    run_simulations(args.model, args.n_samples, corr_matrix)