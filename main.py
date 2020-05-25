import os

from task_param.task_param import TaskParam

from teacher import Leitner, ThresholdTeacher, MCTSTeacher, \
    ThresholdPsychologist, MCTSPsychologist

from run.make_fig import make_fig
from run.make_data import make_data


def main(force=False):

    task_param = TaskParam.get("config/2020_05_22_hom_day_rollout_thr_fixed_window.json")
    teachers = \
        (Leitner, ThresholdTeacher, MCTSTeacher,)
    make_fig(**make_data(tk=task_param, teachers=teachers,
                         force=force))


if __name__ == "__main__":
    main(force=True)
