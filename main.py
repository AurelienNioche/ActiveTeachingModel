import os

from task_param.task_param import TaskParam

from run.make_fig import make_fig
from run.make_data import make_data


def main(force=False):

    task_param = TaskParam.get("config/2020_05_25_hom_day_rollout_thr_fixed_window_longer_session.json")
    make_fig(**make_data(tk=task_param, force=force))


if __name__ == "__main__":
    main(force=True)
