import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats
from tqdm import tqdm

from fit.scipy import Minimize
from learner.act_r import ActR
from teacher.random import RandomTeacher

from simulation.fake import generate_fake_task_param
from simulation.run import run

from utils.plot import save_fig

from utils.backup import dump, load

import os

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]


class FitFidelity:

    def __init__(self, t_min=25, n_iteration=30, n_item=79,
                 student_model=ActR,
                 fit_class=Minimize,
                 teacher_model=RandomTeacher,
                 n_agent=3):

        self.t_min = t_min
        self.n_iteration = n_iteration
        self.n_item = n_item

        if (self.n_iteration - self.t_min) < 4:
            raise ValueError("The difference between n_iteration "
                             "and t_min must be "
                             "of at least 4 to compute fidelity")

        self.student_model = student_model
        self.teacher_model = teacher_model

        self.fit_class = fit_class

        self.n_param = len(self.student_model.bounds)
        self.param = {}

        self.n_agent = n_agent

    def compute_fidelity(self):
        """
        :return: 2D array with each parameter as rows and the mean of the best
        value difference as columns.
        """

        seeds = np.arange(self.n_agent)

        changes = np.zeros((self.n_param,
                            self.n_agent,
                            self.n_iteration - self.t_min - 1))

        for i in range(self.n_agent):

            seed = seeds[i]

            np.random.seed(seed)

            param = self.student_model.generate_random_parameters()
            task_param = generate_fake_task_param(n_item=self.n_item)

            r = run(
                student_param=param,
                student_model=self.student_model,
                teacher_model=self.teacher_model,
                n_item=self.n_item,
                n_iteration=self.n_iteration, task_param=task_param)

            questions = r["questions"]
            successes = r["successes"]

            f = self.fit_class(model=self.student_model)
            fit_r = f.evaluate(
                hist_question=questions[:self.t_min],
                hist_success=successes[:self.t_min],
                task_param=task_param,
            )

            best_v = {}
            best_v.update(fit_r["best_param"])

            t_idx = 0

            for t in tqdm(range(self.t_min+1, self.n_iteration)):

                fit_r = f.evaluate(
                     hist_question=questions[:t],
                     hist_success=successes[:t],
                     task_param=task_param)

                for p_idx, k in\
                        enumerate(sorted(fit_r["best_param"].keys())):

                    changes[p_idx, i, t_idx] =\
                        np.abs(best_v[k] - fit_r["best_param"][k])

                best_v.update(fit_r["best_param"])

                t_idx += 1

        return changes


def _plot(changes, t_min, fig_name, model):

    x = t_min + np.arange(changes.shape[2])

    n_param = len(model.bounds)
    array_item = np.arange(n_param)

    max_n_item_per_figure = 100
    n_fig = n_param // max_n_item_per_figure + \
        int(n_param % max_n_item_per_figure != 0)

    item_groups = np.array_split(array_item, n_fig)

    assert 'pdf' in fig_name
    root = fig_name.split('.pdf')[0]

    for idx_fig, item_gp in enumerate(item_groups):

        fig, axes = plt.subplots(nrows=n_param, figsize=(5, 1.5*n_param))

        for ax_idx, item in enumerate(item_gp):

            ax = axes[ax_idx]

            param_name = model.bounds[ax_idx][0]
            ax.set_ylabel(f'{param_name}-change')

            data = changes[ax_idx, :]

            mean = np.mean(data, axis=0)
            # std = np.std(data, axis=0)

            ax.plot(x, mean,
                    alpha=1, color='C0')

            # ax.fill_between(
            #     mean,
            #     y1=mean - std,
            #     y2=mean + std,
            #     alpha=0.2,
            #     color='C0'
            # )

            if ax_idx != n_param-1:
                ax.set_xticks([])

        axes[-1].set_xlabel('Time')

        fig_name_idx = root + f'_{idx_fig}.pdf'
        save_fig(fig_name_idx, sub_folder=SCRIPT_NAME)


def main(n_agent, t_min, n_iteration, student_model, force=False):

    extension = f"fit-fidelity-{student_model.__name__}-t_min={t_min}" \
                f"-n_iteration={n_iteration}"
    bkp_file = os.path.join("bkp", "fit_fidelity", f"{extension}.p")

    mean_array = load(bkp_file)
    if mean_array is None or force:
        np.random.seed(123)

        ff = FitFidelity(student_model=student_model, n_iteration=n_iteration,
                         n_agent=n_agent)
        mean_array = ff.compute_fidelity()
        dump(mean_array, bkp_file)

    _plot(
        changes=mean_array, student_model=student_model, t_min=t_min,
        fig_name=f'{extension}.pdf')


if __name__ == "__main__":
    main(n_agent=10, t_min=25, n_iteration=300, student_model=ActR)
