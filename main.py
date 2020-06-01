import os

from task_param.task_param import TaskParam

from run.make_fig import make_fig
from run.make_data import make_data


def main(force=False):

    configs = \
        ("2020_06_01_day_longbreak",
         "2020_06_01_day",
         "2020_06_01_single_session")

    for config in configs:
        task_param = TaskParam.get(os.path.join("config", f"{config}.json"))
        make_fig(**make_data(tk=task_param, force=force))


if __name__ == "__main__":
    main(force=True)
