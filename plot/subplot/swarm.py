import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_swarm(df: pd.DataFrame, fig_path: str) -> None:
    """Swarm plot total items recalled per agent"""

    plt.close("all")
    box = sns.catplot(x="Teacher", y="Items learnt", kind="swarm", data=df)
    #for ax in g.axes.flat:
    #    ax.plot((0, 50), (0, .2 * 50), c=".2", ls="--")
    box.savefig(os.path.join(fig_path, "swarm.pdf"))

