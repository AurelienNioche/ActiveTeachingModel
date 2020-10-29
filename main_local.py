"""
Run simulations and save results
"""

import os
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from run.make_data_triton import run
from settings.config_triton import Config


def make_data(config_file):

    config = Config.get(config_file)
    saving_df = run(config=config)
    f_name = f"{config.config_file.split('.')[0]}.csv"
    os.makedirs(config.data_folder, exist_ok=True)
    saving_df.to_csv(os.path.join(config.data_folder, f_name))


def main():

    files = [
        p.path
        for p in os.scandir(os.path.join("config", "triton"))
        if os.path.splitext(p.path)[1] == ".json"
    ]
    file_count = len(files)
    assert file_count > 0

    # for file in tqdm(files):
    #     make_data(file)

    with Pool(processes=cpu_count()) as p:
        max_ = file_count
        with tqdm(total=max_) as pbar:
            for i, _ in enumerate(p.imap_unordered(make_data, files)):
                pbar.update()


if __name__ == "__main__":
    main()
