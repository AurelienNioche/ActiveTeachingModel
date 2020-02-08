import numpy as np
import matplotlib.pyplot as plt

from utils.plot import save_fig


def plot_scatter_metric(data, y_label, x_tick_labels, title=None,
                        ax=None,
                        colors=None, alpha=0.5, dot_size=20,
                        fig_name=None, fig_folder=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Extract from data...
    n = data.shape[-1]

    # Colors
    if colors is None:
        colors = np.array([f"C{i}" for i in range(n)])

    # Containers for boxplot
    positions = list(range(n))
    values_box_plot = [[] for _ in range(n)]

    # Containers for scatter
    x_scatter = []
    y_scatter = []
    colors_scatter = []

    # For each boxplot
    for i in range(n):

        # For every value
        for v in data[:, i]:

            # Add value to the boxplot container
            values_box_plot[i].append(v)

            # Add value to the scatter plot
            x_scatter.append(i + np.random.uniform(-0.05*n, 0.05*n))
            y_scatter.append(v)
            colors_scatter.append(colors[i])

    # Plot the scatter
    ax.scatter(x_scatter, y_scatter, c=colors_scatter, s=dot_size,
               alpha=alpha, linewidth=0.0, zorder=1)

    # Plot the boxplot
    bp = ax.boxplot(values_box_plot, positions=positions,
                    labels=x_tick_labels, showfliers=False, zorder=2)

    # Set the color of the boxplot
    for e in ['boxes', 'caps', 'whiskers', 'medians']:
        for b in bp[e]:
            b.set(color='black')

    # Set the label of the y axis
    ax.set_ylabel(y_label)

    # Set the title
    ax.set_title(title)

    if fig_name is not None and fig_folder is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
