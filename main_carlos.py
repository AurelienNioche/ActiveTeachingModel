#%%
"""
Run simulations and save results
"""

import os

from itertools import product
from typing import Iterable, Hashable

import pandas as pd

import model.parameters
import settings.paths as paths
import utils.file_loader as file_loader
from task_param.task_param_carlos import TaskParam
from run.make_data_carlos import make_data


def main(job_id: int, num_agents: int) -> None:

    # Select config file
    file_paths = file_loader.get_file_paths(paths.AUTO_JSON_DIR, ".json")
    file_paths = tuple(file_loader.get_file_paths_not_empty(file_paths))
    selected_file_path = file_paths[job_id]
    selected_file_name = os.path.basename(selected_file_path)
    save_file_name = f"{selected_file_name.split('.')[0]}.pickle"

    # Run simulation with selected parameters
    task_param = TaskParam.get(selected_file_path)
    results_df = make_data(task_param, job_id, False)

    # Save
    results_df.to_pickle(os.path.join(paths.RESULTS_DIR, save_file_name))


if __name__ == "__main__":
    try:
        job_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))  # Changes cluster simulation
    except:
        job_id = 0  # Default for non-cluster use

    main(job_id, num_agents=100)

#####
# # !! For cell usage !!
# config = os.path.join("config", f"config.json")  # Protyping
#
# task_param = TaskParam.get(config)
# data = make_data(tk=task_param, agent_id=23, human=False)
# print(data)
# data.to_csv("sample_results.csv")
#######

# def make_saving_df(learner_model: str) -> pd.MultiIndex:
#
#     # Get the parameter names for the learner
#     learner_parameters = model.parameters.make_param_learners_df()[
#         learner_model
#     ].dropna()
#
#     # Each element is the level 0 column multiindex
#     agents = tuple(product({"Agent"}, sorted(("Human", "ID"))))
#     models = tuple(product({"Model"}, ("Learner", "Teacher", "Psychologist")))
#     parameters = tuple(product({"Parameter label"}, learner_parameters))
#     session = tuple(
#         product(
#             {"Session"},
#             sorted(
#                 (
#                     "Time delta",
#                     "Iteration",
#                     "Seed",
#                     "Number",
#                     "Item ID",
#                     "P recall error",
#                     "Success",
#                 )
#             ),
#         )
#     )
#     bounds = tuple(product({"Bounds"}, learner_parameters))
#     initial_guess = tuple(product({"Initial guess"}, learner_parameters))
#
#     # Combine all in a multiindex for dataframe
#     multiindex = pd.MultiIndex.from_tuples(
#         (*agents, *models, *parameters, *session, *bounds, *initial_guess)
#     )
#     return pd.DataFrame(columns=multiindex)

# def write_results(job_idx: int, kwarg1: object, kwarg2: object) -> None:
#     """Compose the correct saving stucture and write file"""
#
#     saving_df = make_saving_df(learner_model)
#     saving_df.to_csv(os.path.join("/dev/null", "test.csv"))
#     # bkp_file = os.path.join(PICKLE_FOLDER, f"results.p")
