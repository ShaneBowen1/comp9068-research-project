from librosa import db_to_amplitude, istft
from preprocess import MinMaxNormaliser
import torch

# HIFIGAN imports
from speechbrain.inference.vocoders import HIFIGAN

class SoundGenerator:
    """SoundGenerator is responsible for generating audios from spectrograms"""

    def __init__(self, model, hop_length):
        self.model = model
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        """Generates audio from a spectrogram using the trained model."""

        generated_spectrograms, latent_representations = self.model.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    # def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
    #     """Converts spectrograms to audio signals using the Griffin-Lim algorithm."""

    #     signals = []
    #     for spectrogram, min_max in zip(spectrograms, min_max_values):
    #         # Reshape the log spectrogram
    #         log_spec = spectrogram[:,:,0]

    #         # Denormalize the spectrogram
    #         denorm_log_spec = self._min_max_normaliser.denormalise(
    #             log_spec,
    #             min_max['min'],
    #             min_max['max']
    #         )

    #         # Move log-spectrogram to spectrogram
    #         denorm_spec = db_to_amplitude(denorm_log_spec)

    #         # Convert the spectrogram back to a time-domain signal using the Griffin-Lim algorithm
    #         signal = istft(denorm_spec, hop_length=self.hop_length)
    
    #         # Append the generated signal to the list of signals
    #         signals.append(signal)

# 
# class MelSpectrogramExtractor:
#     """
#     Mel Spectrogram Extractor is responsible for extracting the Mel spectrogram from a time-series signal.
#     """
#     def __init__(self, sample_rate, n_mels, frame_size, hop_length):
#         self.sample_rate = sample_rate
#         self.n_mels = n_mels
#         self.frame_size = frame_size
#         self.hop_length = hop_length
    
#     def extract(self, signal):
#         mel_spectrogram = librosa.feature.melspectrogram(
#             y=signal,
#             sr=self.sample_rate,
#             n_fft=self.frame_size,
#             hop_length=self.hop_length,
#             n_mels=self.n_mels
#         )
#         log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
#         return log_mel_spectrogram

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        """HIFIGAN: Converts spectrograms to audio signals using the HIFIGAN vocoder."""

        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")

        signals = []
        for spectrogram, min_max in zip(spectrograms, min_max_values):
            # Reshape the log spectrogram
            log_spec = spectrogram[:,:,0]

            # Denormalize the spectrogram
            denorm_log_spec = self._min_max_normaliser.denormalise(
                log_spec,
                min_max['min'],
                min_max['max']
            )

            # Move log-spectrogram to spectrogram
            #denorm_spec = db_to_amplitude(denorm_log_spec)

            denorm_log_spec = torch.from_numpy(denorm_log_spec).unsqueeze(0)  # Add batch dimension

            # Convert the spectrogram back to a time-domain signal using HIFIGAN
            signal = hifi_gan.decode_batch(denorm_log_spec)

            # Convert the signal from a PyTorch tensor to a NumPy array
            signal = signal[0].cpu().numpy()
    
            # Append the generated signal to the list of signals
            signals.append(signal)

        return signals