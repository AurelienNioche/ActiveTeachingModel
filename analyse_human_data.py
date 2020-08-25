import os
import pandas as pd
import numpy as np

"""
To import data: 
scp <user>@<server>:/var/www/html/ActiveTeachingServer/results.csv ~/Desktop
"""


def main():

    df = pd.read_csv(os.path.join("data", "results.csv"), index_col=[0])
    users = np.unique["user"]

    for u in users:
        u_data = df[df["user"] == "carlos@test.com"]



if __name__ == "__main__":
    main()
