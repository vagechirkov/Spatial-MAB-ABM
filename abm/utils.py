from matplotlib import pyplot as plt


def plot_summary(results):
    columns = results.columns
    fig, axes = plt.subplots(1, len(columns), figsize=(4*len(columns), 5))  # 1 row, 5 columns

    for idx, col in enumerate(columns):
        axes[idx].plot(results[col])
        axes[idx].set_title(col.replace('_', ' '))  # line break for readability
        axes[idx].set_xlabel('Index')
        axes[idx].set_ylabel(col)

    plt.tight_layout()
    plt.show()