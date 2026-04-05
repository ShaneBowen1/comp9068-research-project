import os
import pickle

import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from vae import VAE
from train import load_lj_speech

import s3_utils

def select_spectrograms(spectrograms, file_paths, min_max_values, num_samples):
    """Selects a subset of spectrograms for audio generation."""
    sampled_indices = np.random.choice(len(spectrograms), num_samples)
    selected_spectrograms = spectrograms[sampled_indices]
    selected_file_paths = [file_paths[i] for i in sampled_indices]
    selected_min_max_values = [min_max_values[file_paths[i]] for i in sampled_indices]
    print(selected_file_paths)
    print(selected_min_max_values)
    return selected_spectrograms, selected_min_max_values

def save_signal(signals, save_dir, sample_rate=22050):
    """Saves the generated audio signals to disk."""
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, f"generated_{i}.wav")
        sf.write(save_path, signal, sample_rate)
        print(f"Saved generated audio to: {save_path}")

if __name__ == "__main__":

    HOP_LENGTH = 256
    SAVE_DIR_ORIGINAL = "samples/original/"
    SAVE_DIR_GENERATED = "samples/generated/"
    SAVED_MODEL_PATH = "output/tensorflow-training-job-20260403183117/"
    TRAIN_DATA_PATH = "data_source/lj_speech/libopus/audio/4k/"

    # Instantiate the S3 client
    s3_client = s3_utils.S3Client(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', None),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', None),
        region_name=os.getenv('AWS_REGION', 'eu-west-1')
    )

    # Download the model weights from S3
    s3_client.download_file(
        bucket_name=os.getenv('S3_BUCKET_NAME'),
        object_name=os.path.join(SAVED_MODEL_PATH, "model.weights.h5"),
        file_path="model/model.weights.h5"
    )

    # Download the model parameters from S3
    s3_client.download_file(
        bucket_name=os.getenv('S3_BUCKET_NAME'),
        object_name=os.path.join(SAVED_MODEL_PATH, "parameters.pkl"),
        file_path="model/parameters.pkl"
    )

    # Download the min-max values from S3
    s3_client.download_file(
        bucket_name=os.getenv('S3_BUCKET_NAME'),
        object_name=os.path.join(TRAIN_DATA_PATH, "min_max_values.pkl"),
        file_path='min_max_values.pkl'
    )

    # Load the saved model
    model = VAE.load("model")

    # Initialize the SoundGenerator
    sound_generator = SoundGenerator(model, HOP_LENGTH)

    # Load the spectrograms + min max values
    specs, file_paths = load_lj_speech(os.path.join("../", TRAIN_DATA_PATH, "spectrograms/"))
    with open("min_max_values.pkl", "rb") as f:
        min_max_values = pickle.load(f)

    # Sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(
        specs,
        file_paths,
        min_max_values,
        num_samples=5
    )

    # Generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(
        sampled_specs,
        sampled_min_max_values
    )

    # Convert the generated spectrograms back to audio signals
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs,
        sampled_min_max_values
    )

    # Save audio signals
    save_signal(signals, SAVE_DIR_GENERATED)
    save_signal(original_signals, SAVE_DIR_ORIGINAL)
