import os

from task_param.task_param import TaskParam

from run.make_fig import make_fig
from run.make_data import make_data


def main(force):

    configs = "config",

    for config in configs:
        task_param = TaskParam.get(os.path.join("config", f"{config}.json"))
        make_fig(**make_data(tk=task_param, force=force))


if __name__ == "__main__":
    main(force=True)
