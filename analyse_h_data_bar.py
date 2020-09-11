import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    data = pd.read_csv("r.csv", index_col=[0])

    data = data[data.n_ss_done == 14]

    is_spec = data.is_item_specific == True
    is_rec = data.teacher_md == "recursive"

    n_leitner = data.n_recall_leitner
    n_active = data.n_recall_act

    min_v = min(min(n_active), min(n_leitner))
    max_v = max(max(n_active), max(n_leitner))

    fig, axes = plt.subplots(nrows=2, ncols=2)

    dic_spec = {
        "no spec": np.invert(is_spec),
        "spec": is_spec}

    dic_rec = {
        "no rec": np.invert(is_rec),
        "rec": is_rec}


    # for i, spec in enumerate(dic_spec):
    #     for j, rec in enumerate(dic_rec):
    #
    #         ax = axes[i, j]
    #
    #         slc = dic_rec[rec] & dic_spec[spec]
    #         y_label = f"{spec} & {rec}"
    #
    #         x = n_leitner[slc]
    #         y = n_active[slc]
    #         ax.scatter(x, y, alpha=0.5)
    #         ax.set_xlim(min_v, max_v)
    #         ax.set_ylim(min_v, max_v)
    #         ax.set_xlabel("leitner")
    #         ax.set_ylabel(y_label)
    #
    #         ax.plot((min_v, max_v), (min_v, max_v), ls="--", color="black",
    #                 alpha=0.1)
    #
    # plt.tight_layout()
    # plt.savefig("wonderful.pdf")

    exercise = sns.load_dataset("exercise")
    print(exercise.head())

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
    g = sns.catplot(x="planning", y="n_learnt", hue="kind",
                    data=df, kind="box", legend_out=True)
    plt.tight_layout()
    plt.savefig("wonderful_violin.pdf")
    plt.show()
if __name__ == "__main__":

    main()