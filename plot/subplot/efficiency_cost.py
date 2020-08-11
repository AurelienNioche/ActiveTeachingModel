import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_efficiency_and_cost(df: pd.DataFrame, fig_path: str) -> None:
    """Efficiency vs. computational cost"""

    plt.close("all")
    eff_cost_data = df.groupby("Teacher").sum()
    eff_cost = sns.relplot(x="Items learnt", y="Computation time", hue=eff_cost_data.index,  data=eff_cost_data)
    eff_cost.savefig(os.path.join(fig_path, "efficiency_cost.pdf"))
