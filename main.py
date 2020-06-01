import os

from task_param.task_param import TaskParam

from run.make_fig import make_fig
from run.make_data import make_data


def main(force):

    configs = \
        ("2020_06_01_single_session",
         "2020_06_01_day_longbreak",
         "2020_06_01_day")

    for config in configs:
        task_param = TaskParam.get(os.path.join("config", f"{config}.json"))
        make_fig(**make_data(tk=task_param, force=force))


if __name__ == "__main__":
    main(force=True)
