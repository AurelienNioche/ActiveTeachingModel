"""
Run simulations and save results
"""
import os
import sys

from settings.config_triton import Config
from run.make_data_triton import run


def main():

    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            config_file = sys.argv[1]
        else:
            raise ValueError
    else:

        config_file = os.path.join("config",
                                   "triton",
                                   "at-2020_08_25_20_29_25_914475-exp_decay-psy_grid-sampling-29.json")
                                   # "config_triton_trial.json")
    config = Config.get(config_file)
    df = run(config=config)
    f_name = f"{os.path.splitext(os.path.basename(config_file))[0]}.csv"
    df.to_csv(os.path.join("data", f_name))


if __name__ == "__main__":
    main()

