import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tqdm import tqdm

from fit.fit import Fit
from learner.act_r import ActR
from plot.generic import save_fig
from simulation.data import SimulatedData, Data
from simulation.task import Task


class FitFidelity(Fit):
    def __init__(self, t_min=25, t_max=30, n_kanji=79, grade=1, model=ActR,
                 normalize_similarity=False, verbose=False):
        super().__init__

        self.t_min = t_min
        self.t_max = t_max

        if (self.t_max - self.t_min) < 4:
            raise IndexError("The difference between t_max and t_min must be "
                             "of at least 4 to compute fidelity")

        self.n_kanji = n_kanji
        self.grade = grade
        self.model = model
        self.normalize_similarity = normalize_similarity
        self.verbose = verbose

        self.n_param = len(self.model.bounds)
        self.param = {}
        self.changes = np.zeros((self.n_param,
                                self.t_max - self.t_min - 1))
        self.mean_array = np.zeros((self.n_param, self.t_max - self.t_min - 2))

        self.compute_fidelity()

    def _simulate_data(self):

        for bound in self.model.bounds:
            self.param[bound[0]] = np.random.uniform(bound[1], bound[2])

        self.tk = Task(t_max=self.t_max, n_kanji=self.n_kanji,
                       grade=self.grade,
                       normalize_similarity=self.normalize_similarity,
                       verbose=False)

        self.data = SimulatedData(model=self.model, param=self.param,
                                  tk=self.tk, verbose=False)

    def compute_fidelity(self, n_agents=3):

        self.mean_array = np.zeros((self.n_param, self.t_max - self.t_min - 2))
        # - 2 because we will compute the difference and then the mean

        for i in range(n_agents):
            self._simulate_data()

            self.changes = np.zeros((self.n_param,
                                     self.t_max - self.t_min - 1))

            data_view = Data(n_items=self.n_kanji,
                             questions=self.data.questions[:self.t_min],
                             replies=self.data.replies[:self.t_min],
                             possible_replies=
                             self.data.possible_replies[:self.t_min, :])

            self.tk.t_max = self.t_min
            f = Fit(model=self.model, tk=self.tk, data=data_view)
            fit_r = f.evaluate()

            best_v = {}
            best_v.update(fit_r["best_param"])

            i = 0

            for t in tqdm(range(self.t_min+1, self.t_max)):

                data_view = Data(n_items=self.n_kanji,
                                 questions=self.data.questions[:t],
                                 replies=self.data.replies[:t],
                                 possible_replies=
                                 self.data.possible_replies[:t, :])

                self.tk.t_max = t

                f = Fit(model=self.model, tk=self.tk, data=data_view)
                fit_r = f.evaluate()

                for iteration, k in\
                        enumerate(sorted(fit_r["best_param"].keys())):
                    self.changes[iteration, i] =\
                        np.abs(best_v[k] - fit_r["best_param"][k])

                    if i != 0:
                        self.mean_array[iteration, i - 1] = (
                                                    last_change +
                                                    self.changes[iteration, i]
                                                    ) / 2
                        print("arr", self.mean_array, "arrr")
                        # print("chachacha", self.changes)

                    last_change = self.changes[iteration, i]

                best_v.update(fit_r["best_param"])

                i += 1

            if self.verbose:
                print(self.changes)
                print(fit_r)

        self._plot()

    def _plot(self, font_size=42):

        # if p_recall_time is None:
        #     p_recall_time = np.arange(p_recall_value.shape[1])

        fig = plt.figure(figsize=(15, 12))
        ax = fig.subplots()

        mean = self.mean_array[0, :]
        sem = scipy.stats.sem(self.mean_array[0, :], axis=0)
        n_trial = np.arange(0, self.mean_array[0, :].size, 1)

        ax.plot(n_trial, mean, lw=1.5)
        ax.fill_between(
            mean,
            y1=mean - sem,
            y2=mean + sem,
            alpha=0.2
        )

        ax.plot(mean, linestyle=':', color='C0')
        ax.plot(mean, linestyle=':', color='C0')

        ax.set_xlabel('Iteration', fontsize=font_size)
        ax.set_ylabel('Mean', fontsize=font_size)

        fig_name = f'fit-fidelity-{self.model.__name__}-t_max={self.t_max}.pdf'

        save_fig(fig_name=fig_name)


def main():

    np.random.seed(123)

    fit_fidelity = FitFidelity(verbose=False)


if __name__ == "__main__":
    main()


# Suppose you have an array 'data' with each line being the values for
# a specific agent, and each column, values for a fit including data until
# specific time step.
#
# mean = np.mean(data, axis=0)
# sem = scipy.stats.sem(data, axis=0)
# ax.plot(mean, lw=1.5)
# ax.fill_between(
#     range(len(mean)),
#     y1=mean - sem,
#     y2=mean + sem,
#     alpha=0.2)
