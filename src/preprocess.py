import argparse
import pickle
import os
import librosa
import numpy as np
from s3_utils import S3Client

class Loader:
    """
    Loader is responsible for loading an audio file.
    """
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        return librosa.load(file_path, sr=self.sample_rate, duration=self.duration, mono=self.mono)[0]


class Padder:
    """
    Padder is responsible for padding the audio signal.
    """
    def __init__(self, mode="constant"):
        self.mode = mode
    
    def left_pad(self, array, num_missing_items):
        return np.pad(array, (num_missing_items, 0), mode=self.mode)

    def right_pad(self, array, num_missing_items):
        return np.pad(array, (0, num_missing_items), mode=self.mode)


# We may have an abstract class for the extractor, as we may want to use different types of extractors in the future (e.g., Mel spectrogram, MFCC, etc.)
class LogSpectrogramExtractor:
    """
    Log Spectrogram Extractor is responsible for extracting the log spectrogram (in dB) from a time-series signal.
    """
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, signal):
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]  # (1 + frame_size/2, num_frames) 1024 -? 513 -> 512
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))
        return log_spectrogram

class MelSpectrogramExtractor:
    """
    Mel Spectrogram Extractor is responsible for extracting the Mel spectrogram from a time-series signal.
    """
    def __init__(self, sample_rate, n_mels, frame_size, hop_length):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, signal):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

class MFCCExtractor:
    """
    MFCC Extractor is responsible for extracting the MFCC features from a time-series signal.
    """
    def __init__(self, sample_rate, n_mfcc, frame_size, hop_length):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mfcc  # Set n_mels to n_mfcc to ensure we have enough Mel bands for MFCC extraction
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, signal):
        mfccs = librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_length,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels
        )
        return mfccs

class MinMaxNormaliser:
    """
    MinMaxNormaliser is responsible for normalising the spectrogram using min-max normalisation.
    """
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min = min_value
        self.max = max_value

    def normalise(self, array):
        array_min = array.min()
        array_max = array.max()
        return (array - array_min) / (array_max - array_min) * (self.max - self.min) + self.min

    def denormalise(self, norm_array, original_min, original_max):
        return (norm_array - self.min) / (self.max - self.min) * (original_max - original_min) + original_min


class Saver:
    """
    Saver is responsible to save features, and the min max values.
    """
    def __init__(self, feature_save_dir, min_max_values_save_dir, local_false_s3_true):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
        self.local_false_s3_true = local_false_s3_true
        self.s3_client = None
        self.s3_bucket_name = os.getenv('S3_BUCKET_NAME', None)

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        if self.local_false_s3_true:
            self.s3_client.save_object(feature, self.s3_bucket_name, save_path)
        else:
            self._create_folder_if_not_exists(self.feature_save_dir)
            np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, 'min_max_values.pkl')
        if self.local_false_s3_true:
            self.s3_client.save_object(min_max_values, self.s3_bucket_name, save_path)
        else:
            self._create_folder_if_not_exists(self.min_max_values_save_dir)
            self._save(min_max_values, save_path)

    def _save(self, data, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def _create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def _generate_save_path(self, file_path):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(self.feature_save_dir, file_name + '.npy')
        return save_path


class PreprocessingPipeline:
    """
    PreprocessingPipeline is responsible for running the entire preprocessing pipeline, which includes the following steps:
    1- load a file
    2- pad the signal (if necessary)
    3- extracting log spectrogram from signal
    4- normalise spectrogram
    5- save the normalised spectrogram

    Storing the min max values for all the log spectrograms is also important for denormalisation during inference.
    """
    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self.s3_client = None
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir, samples=0):
        for root, _, files in os.walk(audio_files_dir):
            files = files[:samples] if samples > 0 else files
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed {file_path}")

        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_values(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        return len(signal) < self._num_expected_samples

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        return self.padder.right_pad(signal, num_missing_samples)

    def _store_min_max_values(self, save_path, min_value, max_value):
        self.min_max_values[save_path] = {
            "min": min_value,
            "max": max_value
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess the LJ Speech dataset.')

    parser.add_argument('--frame_size', type=int, default=512, help='Frame size for STFT.')
    parser.add_argument('--hop_length', type=int, default=256, help='Hop length for STFT.')
    parser.add_argument('--duration', type=float, default=0.74, help='Duration of audio to load (in seconds).')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate for loading audio.')
    parser.add_argument('--n_mels', type=int, default=256, help='Number of Mel bands to generate (only for MelSpectrogramExtractor).')
    parser.add_argument('--n_mfcc', type=int, default=256, help='Number of MFCCs to generate (only for MFCCExtractor).')
    parser.add_argument('--extractor_type', type=str, default='log', choices=['log', 'mel', 'mfcc'], help='Type of feature extractor to use.')
    parser.add_argument('--mono', type=bool, default=True, help='Whether to convert audio to mono.')
    parser.add_argument('--samples', type=int, default=0, help='Number of samples to preprocess (0 for all).')

    # Use SageMaker default if not passed
    parser.add_argument('--files_dir', type=str, required=True, help='Directory containing the audio files to preprocess.')
    parser.add_argument('--spectrogram_save_dir', type=str, required=True, help='Directory to save the spectrogram features.')
    parser.add_argument('--min_max_values_save_dir', type=str, required=True, help='Directory to save the min-max values for denormalisation.')

    args = parser.parse_args()
    LOCAL_FALSE_S3_TRUE = True  # Set to True to use S3 for saving features, False to use local filesystem

    args.spectrogram_save_dir = os.path.join(args.spectrogram_save_dir, args.extractor_type) + "/"
    args.min_max_values_save_dir = os.path.join(args.min_max_values_save_dir, args.extractor_type) + "/"

    # Instantiate all objects
    loader = Loader(sample_rate=args.sample_rate, duration=args.duration, mono=args.mono)
    padder = Padder()
    if args.extractor_type == 'log':
        extractor = LogSpectrogramExtractor(
            frame_size=args.frame_size,
            hop_length=args.hop_length)
    elif args.extractor_type == 'mel':
        extractor = MelSpectrogramExtractor(
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_length=args.hop_length,
            n_mels=args.n_mels
        )
    elif args.extractor_type == 'mfcc':
        extractor = MFCCExtractor(
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_length=args.hop_length,
            n_mfcc=args.n_mfcc
        )
    normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(
        feature_save_dir=args.spectrogram_save_dir,
        min_max_values_save_dir=args.min_max_values_save_dir,
        local_false_s3_true=LOCAL_FALSE_S3_TRUE
    )

    if LOCAL_FALSE_S3_TRUE:
        saver.s3_client = S3Client(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', None),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', None),
            region_name=os.getenv('AWS_REGION', 'eu-west-1')
        )

    # Create the pipeline and set the objects
    pipeline = PreprocessingPipeline()
    pipeline.loader = loader
    pipeline.padder = padder
    pipeline.extractor = extractor
    pipeline.normaliser = normaliser
    pipeline.saver = saver

    # Run the pipeline
    pipeline.process(args.files_dir, samples=args.samples)
