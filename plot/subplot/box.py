import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(df: pd.DataFrame, fig_path: str) -> None:
    """Swarm plot total items recalled per agent"""

    print("Plotting box and swarm...")
    box, ax = plt.subplots()
    ax = sns.boxplot(x="Teacher", y="Items learnt", data=df)
    ax = sns.swarmplot(x="Teacher", y="Items learnt", data=df, color="0.25", alpha=0.7)

    print("Saving fig...")
    box.savefig(os.path.join(fig_path, "box.pdf"))
    print("Done!")
