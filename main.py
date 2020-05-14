import os

from task_param.task_param import TaskParam

from teacher import Leitner, ThresholdTeacher, MCTSTeacher, \
    ThresholdPsychologist, MCTSPsychologist

from run.make_fig import make_fig
from run.make_data import make_data


def main(force=False):
    for name in (
        'het_days_14_05',
        'het_single_session_14_05',
        'homo_days_14_05',
        'homo_single_session_14_05',

    ):
        task_param = TaskParam.get(os.path.join("config", f"{name}.json"))
        teachers = \
            (Leitner, ThresholdTeacher, ThresholdPsychologist,
             MCTSTeacher, MCTSPsychologist)
        make_fig(**make_data(tk=task_param, teachers=teachers,
                             force=force))


if __name__ == "__main__":
    main(force=True)
