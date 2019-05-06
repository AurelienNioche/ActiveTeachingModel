import matplotlib.pyplot as plt
import os

dir_path = 'fig'


def save_fig(fig_name):

    os.makedirs(dir_path, exist_ok=True)

    file_name = f'{dir_path}/{fig_name}'
    plt.savefig(fname=f'{dir_path}/{fig_name}')

    print(f'Figure "{file_name}" created.')
    plt.close()
