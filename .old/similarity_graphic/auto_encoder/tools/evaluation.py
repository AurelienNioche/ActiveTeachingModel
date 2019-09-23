import os
import matplotlib.pyplot as plt

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
FIG_FOLDER = f'{SCRIPT_FOLDER}/../fig'

os.makedirs(FIG_FOLDER, exist_ok=True)


def show_result(x_test, decoded_images, decoded_images_auto,
                fig_name="autoencoder_image_reconstruction.pdf"):

    size = x_test.shape[1]

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(size, size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_images[i].reshape(size, size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_images_auto[i].reshape(size, size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()

    plt.savefig(f'{FIG_FOLDER}/{fig_name}')


def show_accuracy(train, fig_name="autoencoder_validation_loss.pdf"):

    # Plot the loss plots between training and validation data:
    loss = train.history['loss']
    val_loss = train.history['val_loss']
    epochs = range(len(loss))

    fig, ax = plt.subplots()

    ax.plot(epochs, loss, 'bo', label='Training loss')
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    ax.set_title('Training and validation loss')
    ax.legend()

    plt.tight_layout()

    plt.savefig(f'{FIG_FOLDER}/{fig_name}')
