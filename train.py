
import os
import numpy as np
from vae import VAE
import tensorflow as tf

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS =  100

PATH_TO_SPECTROGRAMS = "data_source/lj_speech/libopus/audio/16k/spectrograms/"

# def load_mnist():
#     """
#     Loads the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is commonly used for training and testing machine learning models in the field of computer vision.
#     The function returns the training and testing data, including both the images (x_train, x_test) and their corresponding labels (y_train, y_test).
#     """
#     from tensorflow.keras.datasets import mnist

#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
#     x_train = x_train.astype('float32') / 255.0
#     x_train = x_train.reshape(x_train.shape + (1,))  # Reshape to (num_samples, height, width, channels)
#     x_test = x_test.astype('float32') / 255.0
#     x_test = x_test.reshape(x_test.shape + (1,))  # Reshape to (num_samples, height, width, channels)

#     return x_train, y_train, x_test, y_test

def load_lj_speech(spectrograms_path):
    """
    Loads the spectrogram data from the specified path. The function reads the spectrogram files, processes them, and returns the data in a format suitable for training machine learning models.
    """
    x_train = []  # for evaluation, should split into train and test sets

    for root, _, files in os.walk(spectrograms_path):
        for file in files[:500]:
            file_path = os.path.join(root, file)
            spectrogram = np.load(file_path)   # (n_bins, n_frames)
            # add a channel dimension if it doesn't exist
            if spectrogram.ndim == 2:
                spectrogram = spectrogram[..., np.newaxis]  # (n_bins, n_frames, 1)
            # reshape to (256, 512) using tf.image.resize
            spectrogram = tf.image.resize(spectrogram, (256, 512)).numpy()
            x_train.append(spectrogram)

    return np.array(x_train)      # Convert list to numpy array
    return x_train[..., np.newaxis]  # (num_samples, n_bins, n_frames, 1)

def train(x_train, learning_rate, batch_size, epochs):
    """
    Trains the autoencoder model using the provided training data and hyperparameters. The function initializes an instance of the AutoEncoder class, compiles the model with the specified learning rate, and then fits the model to the training data for a given number of epochs and batch size.
    After training, the function returns the trained autoencoder model.
    """
    vae = VAE(
        input_shape=(256, 512, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, 2),
        latent_space_dim=128,
    )
    # vae = VAE(
    #     input_shape=(256, 512, 1),
    #     conv_filters=(32, 64, 64, 64),
    #     conv_kernels=(3, 3, 3, 3),
    #     conv_strides=(1, 2, 2, 1),
    #     latent_space_dim=2,
    # )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size=batch_size, epochs=epochs)
    return vae

if __name__ == "__main__":
    x_train = load_lj_speech(PATH_TO_SPECTROGRAMS)
    vae = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    vae.save('model')
