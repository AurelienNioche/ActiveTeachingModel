import matplotlib.pyplot as plt
import os

FIG_FOLDER = 'fig'


def save_fig(fig_name):

    plt.tight_layout()

    file_name = os.path.join(FIG_FOLDER, fig_name)
    dir_path = os.path.dirname(file_name)

    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(fname=file_name)

    print(f'Figure "{file_name}" created.\n')
    plt.close()
