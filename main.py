import os
import sys

from task_param.task_param import TaskParam

from run.make_fig import make_fig
from run.make_data import make_data


def main(force):

    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            config = sys.argv[1]
        else:
            raise ValueError
    else:

        config = os.path.join("config", "config.json")
    task_param = TaskParam.get(config)
    make_fig(**make_data(tk=task_param, force=force))


if __name__ == "__main__":

    main(force=True)
