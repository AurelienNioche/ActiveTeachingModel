import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


def plot(df: pd.DataFrame, learnt_label: str, fig_path: str) -> None:
    """Swarm plot total items recalled per agent"""

    print("Plotting box and swarm...")
    fig, ax = plt.subplots()
    df = df.sort_values("Teacher")
    sns.boxplot(x="Teacher", y=learnt_label, data=df, ax=ax)
    sns.stripplot(x="Teacher", y=learnt_label, data=df,
                  color="0.25", alpha=0.7, ax=ax)

    print("Saving fig...")
    fig.savefig(fig_path, dpi=300)
    print("Done!")
