from matplotlib import pyplot as plt


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