import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats
from tqdm import tqdm

from fit.fit import Fit
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning
from plot.generic import save_fig
from simulation.data import SimulatedData, Data
from simulation.task import Task

from utils.utils import load, dump

import os


class FitFidelity:

    def __init__(self, t_min=25, n_iteration=30, n_kanji=79, grades=(1, ),
                 model=ActR,
                 normalize_similarity=False,
                 n_agent=3):

        self.t_min = t_min
        self.n_iteration = n_iteration

        if (self.n_iteration - self.t_min) < 4:
            raise ValueError("The difference between n_iteration and t_min must be "
                             "of at least 4 to compute fidelity")

        self.n_kanji = n_kanji
        self.grades = grades
        self.model = model
        self.normalize_similarity = normalize_similarity

        self.n_param = len(self.model.bounds)
        self.param = {}

        self.n_agent = n_agent

    def _simulate_data(self, seed):

        np.random.seed(seed)

        for bound in self.model.bounds:
            self.param[bound[0]] = np.random.uniform(bound[1], bound[2])

        self.tk = Task(n_iteration=self.n_iteration, n_kanji=self.n_kanji,
                       grades=self.grades,
                       normalize_similarity=self.normalize_similarity,
                       verbose=False,
                       generate_full_task=True)

        self.data = SimulatedData(model=self.model, param=self.param,
                                  tk=self.tk, verbose=False)

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

            self._simulate_data(seed=seeds[i])

            data_view = Data(n_item=self.n_kanji,
                             questions=self.data.questions[:self.t_min],
                             replies=self.data.replies[:self.t_min],
                             possible_replies=
                             self.data.possible_replies[:self.t_min, :])

            self.tk.n_iteration = self.t_min
            f = Fit(model=self.model, tk=self.tk, data=data_view)
            fit_r = f.evaluate()

            best_v = {}
            best_v.update(fit_r["best_param"])

            t_idx = 0

            for t in tqdm(range(self.t_min+1, self.n_iteration)):

                data_view = Data(n_item=self.n_kanji,
                                 questions=self.data.questions[:t],
                                 replies=self.data.replies[:t],
                                 possible_replies=
                                 self.data.possible_replies[:t, :])

                self.tk.n_iteration = t

                f = Fit(model=self.model, tk=self.tk, data=data_view)
                fit_r = f.evaluate()

                for p_idx, k in\
                        enumerate(sorted(fit_r["best_param"].keys())):

                    changes[p_idx, i, t_idx] =\
                        np.abs(best_v[k] - fit_r["best_param"][k])

                best_v.update(fit_r["best_param"])

                t_idx += 1

        return changes


def _run(model=model, n_iteration=n_iteration, n_agent=n_agent):

    np.random.seed(seed)

    for bound in self.model.bounds:
        self.param[bound[0]] = np.random.uniform(bound[1], bound[2])

    self.tk = Task(n_iteration=self.n_iteration, n_kanji=self.n_kanji,
                   grades=self.grades,
                   normalize_similarity=self.normalize_similarity,
                   verbose=False,
                   generate_full_task=True)

    self.data = SimulatedData(model=self.model, param=self.param,
                              tk=self.tk, verbose=False)


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
        save_fig(fig_name_idx)


def main(force=False):

    student_model = ActRMeaning

    teacher_model = RandomTeacher

    extension = f"fit-fidelity-{student_model.__name__}-t_min={t_min}-n_iteration={n_iteration}"
    bkp_file = os.path.join("../bkp", "fit_fidelity", f"{extension}.p")

    r = load(bkp_file)
    if r is None or force:
        r = _run(r)
        dump(mean_array, bkp_file)

    _plot(
        r,
        fig_name=f'{extension}.pdf')


if __name__ == "__main__":
    main()
