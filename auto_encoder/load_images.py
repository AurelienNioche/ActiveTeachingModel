import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt


def load_training_data(img_size=28, folder="image"):

    n_images = len(os.listdir(folder))

    train_data = np.zeros((n_images, img_size, img_size))

    for i, img in enumerate(os.listdir(folder)):

        # label = label_img(img)
        path = os.path.join(folder, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        train_data[i] = np.array(img)

        # # Basic Data Augmentation - Horizontal Flipping
        # flip_img = Image.open(path)
        # flip_img = flip_img.convert('L')
        # flip_img = flip_img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        # flip_img = np.array(flip_img)
        # flip_img = np.fliplr(flip_img)
        # train_data.append(flip_img)

    # random.shuffle(train_data)
    return train_data


def main():

    train_data = load_training_data()

    print(train_data)
    print(train_data[0].shape)
    plt.imshow(train_data[0], cmap='gist_gray')
    plt.show()


if __name__ == "__main__":

    main()
