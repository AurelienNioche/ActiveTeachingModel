import os

from task_param.task_param import TaskParam

from run.make_fig import make_fig
from run.make_data import make_data


def main(force=False):

    task_param = TaskParam.get("config/2020_05_25_hom_single_ss_rollout_rdm_fixed_window.json")
    make_fig(**make_data(tk=task_param, force=force))


if __name__ == "__main__":
    main(force=True)
