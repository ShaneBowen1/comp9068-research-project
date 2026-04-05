import argparse
import os
import numpy as np
from vae import VAE
import tensorflow as tf
from s3_utils import S3Client

def load_mnist():
    """
    Loads the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is commonly used for training and testing machine learning models in the field of computer vision.
    The function returns the training and testing data, including both the images (x_train, x_test) and their corresponding labels (y_train, y_test).
    """
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))  # Reshape to (num_samples, height, width, channels)
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))  # Reshape to (num_samples, height, width, channels)

    return x_train, y_train, x_test, y_test

def load_lj_speech(spectrograms_path):
    """
    Loads the spectrogram data from the specified path. The function reads the spectrogram files, processes them, and returns the data in a format suitable for training machine learning models.
    """
    x_train = []  # for evaluation, should split into train and test sets
    file_paths = []

    for root, _, files in os.walk(spectrograms_path):
        for file in files:
            file_path = os.path.join(root, file)
            spectrogram = np.load(file_path, allow_pickle=True)   # (n_bins, n_frames)
            x_train.append(spectrogram)
            file_paths.append(file_path.replace("../", ""))

    x_train = np.array(x_train)      # Convert list to numpy array
    return x_train[..., np.newaxis], file_paths  # (num_samples, n_bins, n_frames, 1)

def train(x_train, learning_rate, batch_size, epochs):
    """
    Trains the autoencoder model using the provided training data and hyperparameters. The function initializes an instance of the AutoEncoder class, compiles the model with the specified learning rate, and then fits the model to the training data for a given number of epochs and batch size.
    After training, the function returns the trained autoencoder model.
    """
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, 2),
        latent_space_dim=128
    )
    # vae = VAE(
    #     input_shape=(28, 28, 1),
    #     conv_filters=(32, 64, 64, 64),
    #     conv_kernels=(3, 3, 3, 3),
    #     conv_strides=(1, 2, 2, 1),
    #     latent_space_dim=2,
    # )

    if LOCAL_FALSE_S3_TRUE:
        vae.s3_client = S3Client(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', None),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', None),
            region_name=os.getenv('AWS_REGION', 'eu-west-1')
        )

    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size=batch_size, epochs=epochs)
    return vae

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a VAE on the LJ Speech dataset.')

    # hyperparameters for training the VAE    
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for training the VAE.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training the VAE.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training the VAE.')

    # Use SageMaker default if not passed
    parser.add_argument('--model_dir', type=str, default=f"output/{os.environ.get('TRAINING_JOB_NAME')}", help='Directory to save the trained model.')
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'), help='Directory containing the training data.')

    args = parser.parse_args()

    LOCAL_FALSE_S3_TRUE = True  # Set to True to use S3 for saving/loading model, False to use local filesystem

    x_train, _ = load_lj_speech(args.train_dir)
    #x_train, _, _, _ = load_mnist()

    print(x_train.shape)
    vae = train(x_train, args.learning_rate, args.batch_size, args.epochs)
    vae.save(args.model_dir)
