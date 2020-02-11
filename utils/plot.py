import os

from matplotlib import pyplot as plt

FIG_FOLDER = 'fig'


def save_fig(fig_name, fig_folder=None, sub_folder=None, tight_layout=True,
             **kwargs):

    if tight_layout:
        plt.tight_layout()

    if fig_folder is None:
        fig_folder = FIG_FOLDER
    if sub_folder is None:
        sub_folder = ''

    file_name = os.path.join(fig_folder, sub_folder, fig_name)

    dir_path = os.path.dirname(file_name)
    os.makedirs(dir_path, exist_ok=True)

    plt.savefig(fname=file_name, **kwargs)

    print(f'Figure "{file_name}" created.\n')
    plt.close()
