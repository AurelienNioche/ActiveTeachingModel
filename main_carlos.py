import os

from task_param.task_param_carlos import TaskParam
from run.make_data_carlos import make_data


def save(job_idx: int, kwarg1: object, kwarg2: object) -> None:

    # bkp_file = os.path.join(PICKLE_FOLDER, f"results.p")
    pass


def main(job_idx: int) -> None:

    config = os.path.join("config", f"config{job_idx}.json")

    task_param = TaskParam.get(config)
    save(job_idx, **make_data(tk=task_param))


if __name__ == "__main__":

    main(int(os.getenv("JOB_IDX")))
