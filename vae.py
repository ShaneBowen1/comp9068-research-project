import os
import json
import pickle
import numpy as np

# Import TensorFlow
import tensorflow as tf

# Import Model from Keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, BatchNormalization, Lambda, Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam

class CombinedLoss(Layer):
    """
    Custom layer that adds VAE loss and metrics from model internals
    Receives [inputs, reconstructed, mu, log_var] and returns reconstructed;
    adds total loss and updates reconstruction_loss / kl_loss metrics
    """
    def __init__(self, reconstruction_loss_weight=1000, **kwargs):
        super().__init__(**kwargs)
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.reconstruction_loss_metric = Mean(name="reconstruction_loss")
        self.kl_loss_metric = Mean(name="kl_loss")
    
    def call(self, inputs, **kwargs):
        x, reconstructed, mu, log_var = inputs
        # Per-sample reconstruction loss (MSE over H, W, C)
        recon_per_sample = tf.reduce_mean(tf.square(x - reconstructed), axis=[1, 2, 3])
        # Per-sample KL
        kl_per_sample = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        recon_loss = tf.reduce_mean(recon_per_sample)
        kl_loss = tf.reduce_mean(kl_per_sample)
        total_loss = self.reconstruction_loss_weight * recon_loss + kl_loss
        self.add_loss(total_loss)
        self.reconstruction_loss_metric.update_state(recon_loss)
        self.kl_loss_metric.update_state(kl_loss)
        return reconstructed

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "reconstruction_loss_weight": self.reconstruction_loss_weight}

class VAE:
    """
    Variational Autoencoder is a type of neural network used for unsupervised learning. It consists of an encoder that compresses the input data into a lower-dimensional
    representation, and a decoder that reconstructs the original data from this compressed representation. The goal of an VAE is to learn a compact
    and efficient representation of the input data, which can be useful for tasks such as dimensionality reduction, anomaly detection, and data denoising.
    """

    def __init__(self,
                input_shape,
                conv_filters,
                conv_kernels,
                conv_strides,
                latent_space_dim):
        """
        Initializes the VAE with the specified parameters.

        :param input_shape: A tuple representing the shape of the input data (e.g., (height, width, channels)).
        :param conv_filters: A list of integers specifying the number of filters for each convolutional layer in the encoder.
        :param conv_kernels: A list of integers specifying the kernel size for each convolutional layer in the encoder.
        :param conv_strides: A list of integers specifying the stride for each convolutional layer in the encoder.
        :param latent_space_dim: An integer representing the dimensionality of the latent space (the compressed representation).
        """

        self.input_shape = input_shape    # (28, 28, 1)
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2
        self.reconstruction_loss_weight = 1000000  # Weight for the reconstruction loss in the combined loss function

        self.encoder = None
        self.decoder = None
        self.model = None

        # Private attributes
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        """
        Prints a summary of the VAE model, including the architecture of the encoder and decoder. This method is useful for understanding the structure of the model and
        verifying that it has been built correctly.
        """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        """
        Compiles the VAE model with the specified learning rate. This method sets up the optimizer, loss function, and any metrics that will be used during training.
        The choice of loss function is typically mean squared error (MSE) for autoencoders, as it measures the difference between the original input and the reconstructed output.
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate)
        )

    def train(self, x_train, batch_size=32, epochs=50):
        """
        Trains the VAE model on the provided training data (x_train). This method uses the fit function of the Keras model to perform the training process, which involves feeding
        the input data through the encoder and decoder, calculating the loss, and updating the model weights using backpropagation. The batch_size parameter determines how many
        samples are processed before the model weights are updated, and the epochs parameter specifies how many times the entire training dataset is passed through the model.
        """
        self.model.fit(x_train, x_train,  # For autoencoders, the target output is the same as the input
                       batch_size=batch_size,
                       epochs=epochs)

    def save(self, folder_path="."):
        """
        Saves the trained VAE model to the specified file path. This method allows you to persist the model architecture and weights so that it can be loaded and used later without needing to retrain.
        The file format for saving can be HDF5 (.h5) or TensorFlow SavedModel format, depending on your preference and requirements.
        """
        self._create_folder_if_not_exists(folder_path)
        self._save_parameters(folder_path)
        self._save_weights(folder_path)

    def load_weights(self, weights_path):
        """
        Loads the weights of the VAE model from a file. This method is used to restore the learned parameters of the model after it has been trained and saved. The weights are essential for making predictions or reconstructing data using the VAE.
        The weights can be loaded from a file in HDF5 (.h5) format or TensorFlow SavedModel format, depending on how they were saved.
        """
        self.model.load_weights(weights_path)

    def reconstruct(self, x):
        """
        Reconstructs the input data using the VAE model. This method takes in the original input data, passes it through the encoder to obtain the compressed representation in the latent space, and then passes this representation through the decoder to produce the reconstructed output.
        The method returns both the reconstructed output and the latent representations (z), which can be useful for analyzing how well the VAE has learned to compress and reconstruct the data.
        """
        _, _, z = self.encoder.predict(x)
        reconstructed = self.decoder.predict(z)
        return reconstructed, z

    @classmethod
    def load(cls, folder_path="."):
        """
        Loads a trained VAE model from the specified file path. This method reconstructs the model architecture using the saved parameters and loads the weights into the model.
        After loading, the model is ready to be used for making predictions or further training.
        """
        with open(os.path.join(folder_path, 'parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)
        
        vae = cls(*parameters)  # Unpack parameters to initialize the vae
        vae.load_weights(os.path.join(folder_path, 'model.weights.h5'))
        return vae

    def _create_folder_if_not_exists(self, folder_path):
        """
        Creates the specified folder if it does not already exist. This is a helper method used by the save function to ensure that the directory for saving the model is available.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def _save_parameters(self, folder_path):
        """
        Saves the parameters of the vae model (such as input shape, convolutional layer configurations, and latent space dimensionality) to a file. This information is necessary for reconstructing the model architecture when loading the model later.
        The parameters can be saved in a JSON file or any other format that allows for easy retrieval and parsing.
        """
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        with open(os.path.join(folder_path, 'parameters.pkl'), 'wb') as f:
            pickle.dump(parameters, f)

    def _save_weights(self, folder_path):
        """
        Saves the weights of the vae model to a file. The weights represent the learned parameters of the model after training and are essential for making predictions or reconstructing data using the vae.
        The weights can be saved in a format such as HDF5 (.h5) or TensorFlow SavedModel format, depending on your preference and requirements.
        """
        self.model.save_weights(os.path.join(folder_path, 'model.weights.h5'))

    def _build(self):
        """
        Builds the encoder, decoder, and the complete vae model. This method is responsible for constructing the architecture of the vae based on the
        provided parameters.
        """
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        """
        Builds the complete vae model by connecting the encoder and decoder. The vae model takes the input data, passes it through the encoder to obtain the
        compressed representation in the latent space, and then passes this representation through the decoder to reconstruct the original data. CombinedLoss layer
        uses model internals (mu, log_var) for loss and metrics.
        """
        mu, log_var, z = self.encoder(self._model_input)
        reconstructed = self.decoder(z)
        model_output = CombinedLoss(reconstruction_loss_weight=self.reconstruction_loss_weight, name='combined_loss')(
            [self._model_input, reconstructed, mu, log_var]
        )
        self.model = Model(self._model_input, model_output, name='vae')

    def _build_decoder(self):
        """
        Builds the decoder part of the vae. The decoder consists of a series of convolutional transpose layers that reconstruct the original data from the compressed
        representation in the latent space. The architecture of the decoder is designed to mirror the encoder, but in reverse order, using the same parameters for filters, kernels,
        and strides.
        """
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def _add_decoder_input(self):
        """
        Adds the input layer for the decoder. This layer defines the shape of the compressed representation (the latent space) that the decoder will take as input. The input shape
        for the decoder is determined by the latent_space_dim parameter provided during initialization.
        """
        return Input(shape=(self.latent_space_dim,), name='decoder_input')

    def _add_dense_layer(self, decoder_input):
        """
        Adds a dense (fully connected) layer to the decoder. This layer takes the compressed representation from the latent space and transforms it into a shape that can be reshaped
        into the dimensions required for the convolutional transpose layers. The number of units in this dense layer is determined by the shape of the output from the last convolutional
        layer in the encoder, which is stored in the _shape_before_bottleneck attribute.
        """
        units = int(np.prod(self._shape_before_bottleneck))  # Calculate the number of units needed to reshape back to the shape before the bottleneck
        return Dense(units, activation='relu', name='decoder_dense')(decoder_input)

    def _add_reshape_layer(self, dense_layer):
        """
        Adds a reshape layer to the decoder. This layer reshapes the output of the dense layer into the dimensions required for the convolutional transpose layers. The target shape for
        reshaping is determined by the _shape_before_bottleneck attribute, which stores the shape of the output from the last convolutional layer in the encoder.
        """
        return Reshape(self._shape_before_bottleneck, name='decoder_reshape')(dense_layer)

    def _add_conv_transpose_layers(self, reshape_layer):
        """
        Adds convolutional transpose layers to the decoder. The number of convolutional transpose layers and their configurations (number of filters, kernel size, and stride) are determined
        by the conv_filters, conv_kernels, and conv_strides parameters, but in reverse order compared to the encoder. Each convolutional transpose layer applies a transposed convolution operation
        to the input data, followed by an activation function (e.g., ReLU) to introduce non-linearity.
        """
        x = reshape_layer
        for i in range(self._num_conv_layers - 1, -1, -1):  # Iterate in reverse order to mirror the encoder architecture e.g., if there are 3 conv layers, iterate with i = 2, 1, 0
            x = self._add_conv_transpose_layer(i, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        """
        Adds a single convolutional transpose layer to the decoder. This method is called iteratively for each convolutional transpose layer defined by the conv_filters, conv_kernels, and conv_strides
        parameters, but in reverse order compared to the encoder. The layer_index parameter indicates which convolutional transpose layer is being added, and x represents the input to this layer (which is the output of the previous layer).
        """
        conv_transpose_layer = Conv2DTranspose(filters=self.conv_filters[layer_index],
                               kernel_size=self.conv_kernels[layer_index], 
                               strides=self.conv_strides[layer_index],
                               activation='relu',
                               padding='same',
                               name=f'decoder_conv_transpose_{self._num_conv_layers - layer_index}')(x)
        conv_transpose_layer = BatchNormalization(name=f'decoder_bn_{self._num_conv_layers - layer_index}')(conv_transpose_layer)
        return conv_transpose_layer

    def _add_decoder_output(self, conv_transpose_layers):
        """
        Adds the output layer for the decoder. This layer produces the final reconstructed output of the vae. The number of filters in this layer is determined by the number of channels
        in the original input data (the last element of the input_shape), and it uses a sigmoid activation function to ensure that the output values are between 0 and 1, which is appropriate for image data.
        """
        return Conv2DTranspose(filters=1,   # 1 for number of channels in the input data
                               kernel_size=self.conv_kernels[0],  # Use the kernel size of the first convolutional layer in the encoder
                               #strides=self.conv_strides[0],  # Use the stride of the first convolutional layer in the encoder
                               strides=1,
                               activation='sigmoid',
                               padding='same',
                               name=f'decoder_conv_transpose_layer_{self._num_conv_layers}')(conv_transpose_layers)

    def _build_encoder(self):
        """
        Builds the encoder part of the vae. The encoder consists of a series of convolutional layers that compress the input data into a lower-dimensional
        representation (the latent space). The architecture of the encoder is determined by the conv_filters, conv_kernels, and conv_strides parameters.
        """
        encoder_input = self._add_encoder_input()
        self._model_input = encoder_input  # Store the encoder input as the model input for building the vae later
        conv_layers = self._add_conv_layers(encoder_input)
        mu, log_var, z = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, [mu, log_var, z], name='encoder')

    def _add_encoder_input(self):
        """
        Adds the input layer for the encoder. This layer defines the shape of the input data that the encoder will process. The input shape is specified by the input_shape
        parameter provided during initialization.
        """
        return Input(shape=self.input_shape, name='encoder_input')

    def _add_conv_layers(self, encoder_input):
        """
        Adds convolutional layers to the encoder. The number of convolutional layers and their configurations (number of filters, kernel size, and stride) are determined by
        the conv_filters, conv_kernels, and conv_strides parameters. Each convolutional layer applies a convolution operation to the input data, followed by an activation
        function (e.g., ReLU) to introduce non-linearity.
        """
        x = encoder_input
        for i in range(self._num_conv_layers):
            x = self._add_conv_layer(i, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        Adds a single convolutional layer to the encoder. This method is called iteratively for each convolutional layer defined by the conv_filters, conv_kernels, and conv_strides
        parameters. The layer_index parameter indicates which convolutional layer is being added, and x represents the input to this layer (which is the output of the previous layer).
        """

        # add batch normalization
        conv_layer = Conv2D(filters=self.conv_filters[layer_index],
                      kernel_size=self.conv_kernels[layer_index],
                      strides=self.conv_strides[layer_index],
                      activation='relu',
                      padding='same',
                      name=f'encoder_conv_{layer_index + 1}')(x)
        conv_layer = BatchNormalization(name=f'encoder_bn_{layer_index + 1}')(conv_layer)
        return conv_layer

    def _add_bottleneck(self, x):
        """
        Adds the bottleneck layer to the encoder. The bottleneck layer is responsible for compressing the input data into a lower-dimensional representation (the latent space).
        This method calculates the shape of the output from the last convolutional layer, flattens it, and then creates two dense layers to represent the mean (mu) and log variance
        of the latent space distribution. The reparameterization trick is used to sample a point from this distribution.
        Returns (mu, log_variance, z) for use in CombinedLoss and encoder output.
        """
        self._shape_before_bottleneck = x.shape[1:]  # (7, 7, 8) (width ,height, channels)
        x = Flatten(name='encoder_flatten')(x)
        mu = Dense(self.latent_space_dim, name='latent_mu')(x)  # Mean of the latent space distribution
        log_variance = Dense(self.latent_space_dim, name='latent_log_variance')(x)  # Log variance of the latent space distribution

        def sample_point_from_normal_distribution(args):
            mu_val, log_variance = args
            epsilon = tf.random.normal(shape=tf.shape(mu_val), mean=0., stddev=1.)  # Sample from standard normal distribution
            return mu_val + tf.exp(0.5 * log_variance) * epsilon  # Reparameterization trick

        z = Lambda(sample_point_from_normal_distribution, output_shape=(self.latent_space_dim,), name='encoder_output')([mu, log_variance])
        return mu, log_variance, z

if __name__ == "__main__":
    input_shape = (28, 28, 1)
    conv_filters = [32, 64, 64, 64]
    conv_kernels = [3, 3, 3, 3]
    conv_strides = [1, 2, 2, 1]
    latent_space_dim = 2

    vae = VAE(input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim)
    vae.summary()
