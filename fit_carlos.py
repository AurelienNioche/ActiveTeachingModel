import matplotlib as plt
import numpy as np
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

        self. t_min = t_min
        self.t_max = t_max
        self.n_kanji = n_kanji
        self.grade = grade
        self.model = model
        self.normalize_similarity = normalize_similarity
        self.verbose = verbose

        self.n_param = len(self.model.bounds)
        self.param = {}
        self.changes = np.zeros((self.n_param,
                                     self.t_max - self.t_min - 1))

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
            f = Fit(model=ActR, tk=self.tk, data=data_view)
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

                for j, k in enumerate(sorted(fit_r["best_param"].keys())):
                    self.changes[j, i] =\
                        np.abs(best_v[k] - fit_r["best_param"][k])

                best_v.update(fit_r["best_param"])

                i += 1

            if self.verbose:
                print(self.changes)
                print(fit_r)

    def _plot(self, font_size=42):

        # if p_recall_time is None:
        #     p_recall_time = np.arange(p_recall_value.shape[1])

        fig = plt.figure(figsize=(15, 12))
        ax = fig.subplots()

        mean = np.mean(p_recall_value, axis=0)
        sem = scipy.stats.sem(p_recall_value, axis=0)

        min_ = np.min(p_recall_value, axis=0)
        max_ = np.max(p_recall_value, axis=0)

        ax.plot(p_recall_time, mean, lw=1.5)
        ax.fill_between(
            p_recall_time,
            y1=mean - sem,
            y2=mean + sem,
            alpha=0.2
        )

        ax.plot(p_recall_time, min_, linestyle=':', color='C0')
        ax.plot(p_recall_time, max_, linestyle=':', color='C0')

        ax.set_xlabel('Time', fontsize=font_size)
        ax.set_ylabel('Prob. of recall',fontsize=font_size)

        ax.set_ylim((-0.01, 1.01))

        fig_name=f'fit-fidelity-{self.t_max}.pdf'

        save_fig(fig_name=fig_name)


def main():

    np.random.seed(123)

    fit_fidelity = FitFidelity(verbose=True)


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


