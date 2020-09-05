"""
Run simulations and save results
"""

import os

import settings.paths as paths
from run.make_data_triton import run
from settings.config_triton import Config


def main(job_id: int) -> None:
    """Launch job and save the files"""

    f_names = sorted(
        file_name
        for file_name in os.listdir(paths.CONFIG_CLUSTER_DIR)
        if ".json" in file_name
    )
    f_paths = sorted(
        os.path.join(paths.CONFIG_CLUSTER_DIR, f_name)
        for f_name in f_names
    )
    f_path = f_paths[job_id]
    config = Config.get(f_path)
    saving_df = run(config=config)
    f_name = f"{config.config_file.split('.')[0]}.csv"
    os.makedirs(config.data_folder, exist_ok=True)
    saving_df.to_csv(os.path.join(config.data_folder, f_name))


if __name__ == "__main__":

    try:  # Changes cluster simulation
        JOB_ID = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    except ValueError:  # Default for non-cluster use
        JOB_ID = 0

    main(JOB_ID)
