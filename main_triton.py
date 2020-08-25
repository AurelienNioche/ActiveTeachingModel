#%%
"""
Run simulations and save results
"""
import os
import sys

from settings.config_triton import Config

import settings.paths as paths
import utils.file_loader as file_loader

from run.make_data_triton import run


def main(job_id: int) -> None:

    f_names = sorted(
        file_name
        for file_name in os.listdir(paths.CONFIG_CLUSTER_DIR)
        if ".json" in file_name
    )
    f_paths = sorted(
        os.path.join(paths.CONFIG_CLUSTER_DIR, f_name) for f_name in f_names
    )
    f_path = f_paths[job_id]
    config = Config.get(f_path)
    df = run(config=config)
    f_name = f"{config.config_file.split('.')[0]}.csv"
    df.to_csv(os.path.join(paths.DATA_DIR, f_name))


if __name__ == "__main__":
    try:
        job_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))  # Changes cluster simulation
    except:
        job_id = 0  # Default for non-cluster use
    main(job_id)

# Legacy argv
# def main():
#
#     if len(sys.argv) > 1:
#         if os.path.exists(sys.argv[1]):
#             config_file = sys.argv[1]
#         else:
#             raise ValueError
#     else:
#
#         config_file = os.path.join("config",
#                                    "config_triton_sampling_grid_walsh.json")
#     config = Config.get(config_file)
#     df = run(config=config)
#     f_name = f"{os.path.splitext(os.path.basename(config_file))[0]}.csv"
#     df.to_csv(os.path.join("data", f_name))
