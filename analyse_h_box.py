import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    data = pd.read_csv("data_summary.csv", index_col=[0])
    data = data[data.n_ss_done == 14]

    row_list = []

    for i, row in data.iterrows():
        row_list.append({
            "id": row.user,
            "n_learnt": row.n_recall_leitner,
            "kind": "item spec" if row.is_item_specific else "non item spec",
            "planning": "leitner"
        })
        row_list.append({
            "id": row.user,
            "n_learnt": row.n_recall_act,
            "kind": "item spec" if row.is_item_specific else "non item spec",
            "planning": row.teacher_md
        })

    df = pd.DataFrame(row_list)
    sns.catplot(x="planning", y="n_learnt", hue="kind",
                data=df, kind="box", legend_out=False)
    plt.tight_layout()
    plt.savefig("human_box.png", dpi=300)


if __name__ == "__main__":

    main()
