import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    data = pd.read_csv("data_summary.csv", index_col=[0])
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

    for i, spec in enumerate(dic_spec):
        for j, rec in enumerate(dic_rec):

            ax = axes[i, j]

            slc = dic_rec[rec] & dic_spec[spec]
            y_label = f"{spec} & {rec}"

            x = n_leitner[slc]
            y = n_active[slc]
            ax.scatter(x, y, alpha=0.5)
            ax.set_xlim(min_v, max_v)
            ax.set_ylim(min_v, max_v)
            ax.set_xlabel("leitner")
            ax.set_ylabel(y_label)

            ax.plot((min_v, max_v), (min_v, max_v), ls="--", color="black",
                    alpha=0.1)

    plt.tight_layout()
    plt.savefig("human_scatter.png", dpi=300)


if __name__ == "__main__":

    main()