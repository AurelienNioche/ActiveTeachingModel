import os
import pandas as pd
import numpy as np

"""
To import data: 
scp <user>@<server>:/var/www/html/ActiveTeachingServer/results.csv ~/Desktop
"""


def main():

    df = pd.read_csv(os.path.join("data", "results.csv"), index_col=[0])
    users = np.unique(df["user"])

    for u in users:
        u_data = df[df["user"] == u]
        eval_session = u_data[u_data["is_eval"]]
        teachers = np.unique(eval_session["teacher_md"])
        for t in teachers:
            eval_t = eval_session[eval_session["teacher_md"] == t]
            success = eval_t["success"]
            print(u, t, np.sum(success), len(success))

    # df = df[df["user"] == "carlos2@test.com"]
    # print(np.unique(df["teacher_md"], return_counts=True))


if __name__ == "__main__":
    main()
