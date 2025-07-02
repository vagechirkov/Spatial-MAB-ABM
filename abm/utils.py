from matplotlib import pyplot as plt


def plot_summary(results):
    columns = [
        'avg_reward',
        'private_step_distance',
        'social_step_distance',
        'private_landscape_reconstruction_mse',
        'social_landscape_reconstruction_mse'
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 1 row, 5 columns

    for idx, col in enumerate(columns):
        axes[idx].plot(results[col])
        axes[idx].set_title(col.replace('_', ' '))  # line break for readability
        axes[idx].set_xlabel('Index')
        axes[idx].set_ylabel(col)

    plt.tight_layout()
    plt.show()