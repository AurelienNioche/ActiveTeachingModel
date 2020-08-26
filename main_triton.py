"""
Run simulations and save results
"""
import os

# import sys

from settings.config_triton import Config
import settings.paths as paths

# import utils.file_loader as file_loader

from run.make_data_triton import run


EXT = "_incr_samples"
DIR_NAME = f"triton + {EXT}"
PATH = os.path.join(paths.DATA_CLUSTER_DIR, DIR_NAME)


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
    df.to_csv(os.path.join(PATH, f_name))


if __name__ == "__main__":
    try:
        # Changes cluster simulation
        job_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    except:
        # Default for non-cluster use
        job_id = 0
    main(job_id)
