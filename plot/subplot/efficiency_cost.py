import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plot.utils as utils


def plot(df: pd.DataFrame, fig_path: str) -> None:
    """Efficiency vs. computational cost"""

    print("Plotting efficiency vs. cost...")
    # Mean
    efficiency_mean = df.groupby("Teacher").mean()["Items learnt"].sort_index()
    cost_mean = df.groupby("Teacher").mean()["Computation time"].sort_index()

    # Std
    efficiency_std = df.groupby("Teacher").std()["Items learnt"].sort_index()
    cost_std = df.groupby("Teacher").std()["Computation time"].sort_index()

    # Colors
    color_mappings = utils.map_teacher_colors()
    colors = tuple(color_mappings[teacher] for teacher in efficiency_mean.index)

    # Subplots
    eff_cost, ax = plt.subplots()

    # Error bars
    ax.errorbar(
        efficiency_mean.values,
        cost_mean.values,
        fmt="o",
        ecolor=colors,
        xerr=efficiency_std.values,
        yerr=cost_std.values,
        zorder=0,
    )

    # Scatter
    ax.scatter(efficiency_mean.values, cost_mean.values, c=colors, zorder=1)

    # Labels
    ax.set_xlabel(efficiency_mean.name)
    ax.set_ylabel(cost_mean.name)

    # Save
    print("Saving fig...")
    eff_cost.savefig(os.path.join(fig_path, "efficiency_cost.pdf"))
    print("Done!")


# ax = sns.catplot(x="Items learnt", y="Computation time", hue="Psychologist", data=df)
# eff_cost = sns.relplot(x="Items learnt", y="Computation time", hue=eff_cost_data.index,  data=eff_cost_data)
