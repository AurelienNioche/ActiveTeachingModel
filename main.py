import os

from task_param.task_param import TaskParam

from teacher import Leitner, ThresholdTeacher, MCTSTeacher, \
    ThresholdPsychologist, MCTSPsychologist

from run.make_fig import make_fig
from run.make_data import make_data


def main(force=False):
    for name in (
        'homo_day_18_05_long_horizon_threshold_rollout',
        'homo_single_session_18_05_long_horizon_threshold_rollout'
    ):
        task_param = TaskParam.get(os.path.join("config", f"{name}.json"))
        teachers = \
            (Leitner, ThresholdTeacher, MCTSTeacher,)
        make_fig(**make_data(tk=task_param, teachers=teachers,
                             force=force))


if __name__ == "__main__":
    main(force=True)
