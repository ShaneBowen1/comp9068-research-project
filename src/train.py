import argparse
import os
import numpy as np
from vae import VAE
import tensorflow as tf
from s3_utils import S3Client
from sklearn.model_selection import train_test_split

def make_same_amount(x, y):
    """
    This function takes two arrays as input and returns two arrays with the same number of samples.
    """
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    return x, y

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
    file_paths = []  # for storing the file paths of the spectrograms

    for root, _, files in os.walk(spectrograms_path):
        for file in files:
            file_path = os.path.join(root, file)
            spectrogram = np.load(file_path, allow_pickle=True)   # (n_bins, n_frames)
            x_train.append(spectrogram)
            file_paths.append(file_path.replace("../", ""))  # Store relative path for later use in S3Client

    x_train = np.array(x_train)      # Convert list to numpy array
    return x_train[..., np.newaxis], file_paths  # (num_samples, n_bins, n_frames, 1)

def prepare_dataset(data, val_size=0.15, test_size=0.15):
    """
    Prepares the dataset for training by splitting it into training and testing sets. The function takes the noisy and clean spectrogram data as input, along with a specified test size, and returns the training and testing data for both the noisy and clean spectrograms.
    """
    train_data, test_data = train_test_split(data, test_size=test_size+val_size, random_state=42)                # Split into train and test+val sets
    test_data, val_data = train_test_split(test_data, test_size=val_size/(test_size+val_size), random_state=42)  # Split into val and test sets
    print(f"Training samples: {len(train_data)},  Validation samples: {len(val_data)}, Testing samples: {len(test_data)}")

    # Unzip the data into separate arrays for noisy and clean spectrograms
    x_train_noisy, x_train_clean = zip(*train_data)
    x_val_noisy, x_val_clean = zip(*val_data)
    x_test_noisy, x_test_clean = zip(*test_data)

    # Convert to numpy arrays
    x_train = (
        np.array(x_train_noisy),
        np.array(x_train_clean)
    )
    x_val = (
        np.array(x_val_noisy),
        np.array(x_val_clean)
    )
    x_test = (
        np.array(x_test_noisy),
        np.array(x_test_clean)
    )
    print("X Train Shapes:", x_train[0].shape, x_train[1].shape)
    print("X Val Shapes:", x_val[0].shape, x_val[1].shape)
    print("X Test Shapes:", x_test[0].shape, x_test[1].shape)

    return x_train, x_val, x_test

def train(
        x_train,
        x_val,
        learning_rate,
        batch_size,
        epochs,
        reconstruction_loss_weight,
        latent_space_dim,
        base_filters,
        n_layers
    ):
    """
    Trains the autoencoder model using the provided training data and hyperparameters. The function initializes an instance of the AutoEncoder class, compiles the model with the specified learning rate, and then fits the model to the training data for a given number of epochs and batch size.
    After training, the function returns the trained autoencoder model.
    """
    
    base_filters = args.base_filters
    conv_filters = tuple(base_filters // (2 ** i) for i in range(n_layers))  # Example: (512, 256, 128, 64, 32) for n_layers=5
    conv_kernels = tuple(3 for _ in range(n_layers))  # Example: (3, 3, 3, 3, 3) for n_layers=5
    conv_strides = tuple(2 for _ in range(n_layers-1)) + (1,)  # Example: (2, 2, 2, 2, 1) for n_layers=5
    
    input_shape = x_train[0].shape[1:]  # Get the shape of the spectrograms (n_bins, n_frames, 1)

    vae = VAE(
        input_shape=input_shape,
        conv_filters=conv_filters,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        latent_space_dim=latent_space_dim,
        reconstruction_loss_weight=reconstruction_loss_weight
    )

    if LOCAL_FALSE_S3_TRUE:
        vae.s3_client = S3Client(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', None),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', None),
            region_name=os.getenv('AWS_REGION', 'eu-west-1')
        )

    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, x_val, batch_size=batch_size, epochs=epochs)
    return vae

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a VAE on the LJ Speech dataset.')

    # hyperparameters for training the VAE    
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for training the VAE.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training the VAE.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training the VAE.')
    parser.add_argument('--reconstruction_loss_weight', type=float, default=1000.0, help='Weight for the reconstruction loss in the combined loss function.')
    parser.add_argument('--latent_space_dim', type=int, default=128, help='Dimensionality of the latent space in the VAE.')
    parser.add_argument('--base_filters', type=int, default=512, help='Number of filters in the first convolutional layer of the VAE.')
    parser.add_argument('--n_layers', type=int, default=5, help='Number of convolutional layers in the VAE.')
    parser.add_argument('--val_size', type=float, default=0.15, help='Fraction of training data for validation')
    parser.add_argument('--test_size', type=float, default=0.15, help='Fraction of all samples for held-out test')
    parser.add_argument('--save_model', type=bool, default=True, help='Whether to save the trained model to the specified model directory.')

    # Use SageMaker default if not passed
    parser.add_argument('--model_dir', type=str, default=f"output/{os.environ.get('TRAINING_JOB_NAME')}", help='Directory to save the trained model.')
    parser.add_argument('--noisy_dir', type=str, default=os.environ.get('SM_CHANNEL_NOISY'), help='Directory containing the noisy training data.')
    parser.add_argument('--clean_dir', type=str, default=os.environ.get('SM_CHANNEL_CLEAN'), help='Directory contain the clean training data.')

    args = parser.parse_args()

    LOCAL_FALSE_S3_TRUE = True  # Set to True to use S3 for saving/loading model, False to use lfs

    x_noisy, _ = load_lj_speech(args.noisy_dir)
    x_clean, _ = load_lj_speech(args.clean_dir)
    x_noisy, x_clean = make_same_amount(x_noisy, x_clean)
    x_train, x_val, x_test = prepare_dataset(
        list(zip(x_noisy, x_clean)),
        val_size=args.val_size,
        test_size=args.test_size
    )
    vae = train(
        x_train,
        x_val,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.reconstruction_loss_weight,
        args.latent_space_dim,
        args.base_filters,
        args.n_layers
    )

    # x_train, _, _, _ = load_mnist()
    # x_train = x_train[:2000]  # Use a subset of the data for faster training during development
    # x_train, x_val, x_test = prepare_dataset(
    #     list(zip(x_train, x_train)),  # Using x_train as both noisy and clean
    #     val_size=args.val_size,
    #     test_size=args.test_size
    # )
    # vae = train(
    #     x_train,
    #     x_val,
    #     args.learning_rate,
    #     args.batch_size,
    #     args.epochs,
    #     args.reconstruction_loss_weight,
    #     args.latent_space_dim,
    #     args.base_filters,
    #     args.n_layers
    # )

    if args.save_model:
        vae.save(args.model_dir)

    total_loss, reconstruction_loss, kl_loss = vae.evaluate(x_test, batch_size=args.batch_size)
    print(f"Test Loss: {total_loss}")
    print(f"Reconstruction Loss: {reconstruction_loss}")
    print(f"KL Loss: {kl_loss}")
