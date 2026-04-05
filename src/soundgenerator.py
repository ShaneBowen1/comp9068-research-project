from librosa import db_to_amplitude, istft
from preprocess import MinMaxNormaliser

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

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        """Converts spectrograms to audio signals using the Griffin-Lim algorithm."""

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
            denorm_spec = db_to_amplitude(denorm_log_spec)

            # Convert the spectrogram back to a time-domain signal using the Griffin-Lim algorithm
            signal = istft(denorm_spec, hop_length=self.hop_length)
    
            # Append the generated signal to the list of signals
            signals.append(signal)

        return signals
