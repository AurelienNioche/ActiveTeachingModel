import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(df: pd.DataFrame, learnt_label: str, fig_path: str) -> None:
    """Swarm plot total items recalled per agent"""

    print("Plotting box and swarm...")
    box, ax = plt.subplots()
    df = df.sort_values("Teacher")
    ax = sns.boxplot(x="Teacher", y=learnt_label, data=df)
    ax = sns.swarmplot(x="Teacher", y=learnt_label, data=df, color="0.25",
                       alpha=0.7,)

    print("Saving fig...")
    box.savefig(fig_path)
    print("Done!")
