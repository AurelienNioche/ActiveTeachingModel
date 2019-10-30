import numpy as np
from tqdm import tqdm

from adaptive_design.engine.teacher import Teacher
from adaptive_design.plot import create_fig

from utils.utils import dump

from learner.half_life import HalfLife


def run():

    grid_size = 100
    n_design = 100
    num_trial = 500

    possible_design = np.arange(n_design)
    learner_model = HalfLife
    true_param = {
        "beta": 0.02,
        "alpha": 0.2
    }

    np.random.seed(123)

    param = sorted(learner_model.bounds.keys())

    learner = learner_model(param=true_param)
    engine = Teacher(
        learner_model=learner_model,
        possible_design=possible_design,
        grid_size=grid_size)

    design_types = [
        'adaptive_teaching',
        'pure_teaching', 'optimal', 'random']

    post_means = {d: {pr: np.zeros(num_trial)
                      for pr in param}
                  for d in design_types}
    post_sds = {d: {pr: np.zeros(num_trial)
                    for pr in param}
                for d in design_types}

    recall_means = {d: {pr: np.zeros(num_trial)
                      for pr in param}
                  for d in design_types}

    recall_sds = {d: {pr: np.zeros(num_trial)
                    for pr in param}
                for d in design_types}


    # Run simulations for three designs
    for design_type in design_types:

        print("Resetting the engine...")

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
    dump(obj=recall_means, file_path='bkp/adaptive/recall_sds.p')

    create_fig(param=param, design_types=design_types,
               post_means=post_means, post_sds=post_sds,
               true_param=true_param, num_trial=num_trial,
               fig_name=
               f"adaptive_teaching_"
               f"{learner_model.__name__}.pdf")


if __name__ == "__main__":
    run()
