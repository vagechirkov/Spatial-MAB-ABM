from experiments import effect_of_heterogeneity


if __name__ == "__main__":
    # np.linspace(0.01, 0.06, 5)
    tau_values = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    for n_agents in (4, 8):
        for n_mutants in (1, 2):
            effect_of_heterogeneity(
                experiment_name=f"{n_mutants}-tau-mutant_{n_agents}-agents",
                mutant_value=tau_values,
                resident_value=tau_values,
                n_mutants=n_mutants,
                n_seeds=200,
                n_iterations=100,
                n=n_agents,
                length_scale_private=2
            )
