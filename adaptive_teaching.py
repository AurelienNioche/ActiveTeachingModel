import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from adaptive_design.engine.teacher_half_life import TeacherHalfLife
from adaptive_design.plot import create_fig

from utils.utils import dump, load

from learner.half_life import FastHalfLife


def run():

    force = True

    engine_model = TeacherHalfLife

    grid_size = 100
    n_design = 100
    num_trial = 200

    possible_design = np.arange(n_design)
    learner_model = FastHalfLife

    true_param = {
        "beta": 0.02,
        "alpha": 0.2
    }

    design_types = [
        'optimal',
        'pure_teaching',
        'adaptive_teaching',
        'random']

    param = sorted(learner_model.bounds.keys())

    post_means = load(file_path='bkp/adaptive/post_means.p')
    post_sds = load(file_path='bkp/adaptive/post_sds.p')
    recall_means = load(file_path='bkp/adaptive/recall_means.p')
    recall_sds = load(file_path='bkp/adaptive/recall_sds.p')

    if not post_means or not post_sds or not recall_means or not recall_sds \
            or force:

        learner = learner_model(param=true_param, n_item=n_design)
        engine = engine_model(
            learner_model=learner_model,
            possible_design=possible_design,
            grid_size=grid_size)

        post_means = {d: {pr: np.zeros(num_trial)
                          for pr in param}
                      for d in design_types}
        post_sds = {d: {pr: np.zeros(num_trial)
                        for pr in param}
                    for d in design_types}

        recall_means = {d: np.zeros(num_trial)
                        for d in design_types}

        recall_sds = {d: np.zeros(num_trial)
                      for d in design_types}

        # Run simulations for every design
        for design_type in design_types:

            print("Resetting the engine...")
            np.random.seed(123)

            # Reset the engine as an initial state
            engine.reset()

            print(f"Computing results for design '{design_type}'...")

            for trial in tqdm(range(num_trial)):

                # Compute an optimal design for the current trial
                design = engine.get_design(design_type)

                # Get a response using the optimal design
                p = learner.p_recall(item=design)

                response = int(p > np.random.random())

                # Update the engine
                engine.update(design, response)

                for i, pr in enumerate(param):
                    post_means[design_type][pr][trial] = engine.post_mean[i]
                    post_sds[design_type][pr][trial] = engine.post_sd[i]

                p_recall = np.zeros(len(possible_design))
                for i, item in enumerate(possible_design):
                    p_recall[i] = learner.p_recall(item=item)
                recall_means[design_type][trial] = np.mean(p_recall)
                recall_sds[design_type][trial] = np.std(p_recall)

                # Make the user learn
                learner.learn(item=design)

        dump(obj=post_means, file_path='bkp/adaptive/post_means.p')
        dump(obj=post_sds, file_path='bkp/adaptive/post_sds.p')
        dump(obj=recall_means, file_path='bkp/adaptive/recall_means.p')
        dump(obj=recall_sds, file_path='bkp/adaptive/recall_sds.p')

    create_fig(param=param, design_types=design_types,
               post_means=post_means, post_sds=post_sds,
               true_param=true_param, num_trial=num_trial,
               fig_name=
               f"adaptive_teaching_"
               f"{learner_model.__name__}_n_trial_"
               f"{num_trial}_n_item_{n_design}.pdf")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [f'C{i}' for i in range(len(design_types))]

    for i, dt in enumerate(design_types):

        means = recall_means[dt]
        stds = recall_sds[dt]

        ax.plot(means, color=colors[i], label=dt)
        ax.fill_between(range(num_trial), means-stds,
                        means+stds, alpha=.2, color=colors[i])

    ax.set_xlabel("time")
    ax.set_ylabel(f"probability or recall")

    plt.legend()
    plt.tight_layout()

    fig_name = f"adaptive_teaching_probability_recall_" + \
               f"{learner_model.__name__}_n_trial_" + \
               f"{num_trial}_n_item_{n_design}.pdf"

    FIG_FOLDER = os.path.join("fig", "adaptive")
    os.makedirs(FIG_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(FIG_FOLDER, fig_name))


if __name__ == "__main__":
    run()
