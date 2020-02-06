import os

import matplotlib.pyplot as plt
import numpy as np

from learner.act_r_custom import ActRMeaning

from teacher.random import RandomTeacher

from simulation.fake import generate_fake_task_param
from simulation.run import run

from fit.scipy import Minimize

from datetime import timedelta
from time import time

DATA_FOLDER = os.path.join("../bkp", "model_evaluation")
FIG_FOLDER = "fig"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


def main():

    student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}
    student_model = ActRMeaning

    teacher_model = RandomTeacher

    fit_class = Minimize

    # param = {"d": 0.5, "tau": 0.01, "s": 0.06}
    # model = ActR

    n_iteration = 500
    n_item = 79

    init_evals = 3
    max_iter = 50 - init_evals

    verbose = True

    task_param = generate_fake_task_param(n_item=n_item)

    r = run(
        student_param=student_param,
        student_model=student_model,
        teacher_model=teacher_model,
        n_item=n_item,
        n_iteration=n_iteration,
        task_param=task_param)

    questions = r["questions"]
    successes = r["successes"]

    f = fit_class(model=student_model)
    t0 = time()

    obj_with_true_param = objective(
        student_param=student_param,
        student_model=student_model,
        questions=r["questions"]
    )
    print('obj with true param', obj_with_true_param)

    f.evaluate(max_iter=max_iter, init_evals=init_evals, verbose=verbose)

    # print(f.evaluate(n_iter=max_iter, init_points=init_evals))

    print("Time:", timedelta(seconds=time() - t0))

    n_param = len(param)

    param_key = sorted(list(param.keys()))

    history = f.history_eval_param
    objective = f.obj_values
    fig, axes = plt.subplots(nrows=n_param+1)

    x = np.arange(len(history))

    for i in range(n_param):
        ax = axes[i]

        key = param_key[i]

        y = [v[key] for v in history]
        ax.plot(x, y, color=f"C{i}")
        ax.axhline(param[key], linestyle='--', color="black", alpha=0.5,
                   zorder=-1)
        ax.set_ylabel(key)

        ax.set_xlim(min(x), max(x))

        ax.set_xticks([])

    ax = axes[-1]

    ax.set_xlim(min(x), max(x))
    ax.plot(x, objective)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Objective")

    ax.axhline(obj_with_true_param, linestyle='--', color="black", alpha=0.5,
               zorder=-1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    main()
