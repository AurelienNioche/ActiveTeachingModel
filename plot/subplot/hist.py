import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


def plot(df: pd.DataFrame, learnt_label: str, fig_path: str) -> None:
    """Swarm plot total items recalled per agent"""

    print("Plotting hist...")
    fig, ax = plt.subplots()
    sns.histplot(x=learnt_label, hue="Teacher", ax=ax, data=df,
                 lw=0, alpha=0.5)

    # plt.show()
    print("Saving fig...")
    fig.savefig(fig_path)
    print("Done!")