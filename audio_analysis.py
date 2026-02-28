import argparse
import os
import resampy
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from collections import defaultdict
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio
)

class AudioLoader:
    @staticmethod
    def load_audio(file_path, target_sr=None):
        """
        :param file_path: Path to the audio file
        :param target_sr: Target sample rate (optional)
        :return: Normalized audio waveform as a float32 numpy array and the sample rate
        """
        wav_data, sr = sf.read(file_path, dtype=np.int16)

        # Check if audio data is in the expected format
        if wav_data.dtype != np.int16:
            raise ValueError(f'Bad sample type: {wav_data.dtype}')

        # Resample if target sample rate is specified
        if target_sr and sr != target_sr:
            wav_data = resampy.resample(wav_data, sr, target_sr)
            sr = target_sr

        # Convert to mono if audio is stereo
        if wav_data.ndim > 1:
            wav_data = np.mean(wav_data, axis=1)  # Convert to mono by averaging channels

        wav = wav_data / 32768.0  # Normalize to [-1, 1]
        wav = wav.astype('float32')  # Convert to float32 for processing
        return wav, sr


class AudioMetric:
    def __init__(self, name, metric_obj):
        self.name = name
        self.metric_obj = metric_obj
        self.scores = []
    
    def calculate(self, ref_tensor, deg_tensor):
        """
        :param ref_tensor: Reference audio tensor
        :param deg_tensor: Degraded audio tensor
        :return: Calculated metric score
        """
        try:
            score = self.metric_obj(ref_tensor, deg_tensor).item()
            self.scores.append(score)
            return score
        except Exception as e:
            print(f"An error occurred while calculating {self.name} score: {str(e)}")
            return None
    
    def average_score(self):
        """
        :return: Average score across all samples
        """
        if len(self.scores) > 0:
            return round(sum(self.scores) / len(self.scores), 4)
        else:
            return None
    
    def count(self):
        """
        :return: Number of samples processed for this metric
        """
        return len(self.scores)


class MetricFactory:
    @staticmethod
    def create_metrics(metric):
        """
        :param metric_name: List of metric names to create
        :return: List of AudioMetric instances
        """
        available_metrics = {
            'PESQ': PerceptualEvaluationSpeechQuality(fs=16000, mode='nb'),
            'STOI': ShortTimeObjectiveIntelligibility(fs=16000),
            'SI-SDR': ScaleInvariantSignalDistortionRatio(),
            'SI-SNR': ScaleInvariantSignalNoiseRatio()
        }

        metrics = []
        for name in metric:
            if name in available_metrics:
                metrics.append(AudioMetric(name, available_metrics[name]))
            else:
                raise ValueError(f"Unsupported metric: {name}")
        
        return metrics


class AudioAnalyzer:
    def __init__(self, input_folder, degraded_folder, metrics):
        self.input_folder = input_folder
        self.degraded_folder = degraded_folder
        self.metrics = metrics

    def analyze(self, format, samples=0):
        """
        :param format: Format of the degraded audio files (e.g., 'opus')
        :param samples: Number of samples to process for each metric (0 for all samples)
        """
        for idx, filename in enumerate(os.listdir(self.input_folder), start=1):
            if filename.endswith(".wav"):
                input_file = os.path.join(self.input_folder, filename)
                degraded_file = os.path.join(self.degraded_folder, filename.replace(".wav", f".{format}"))

            # check if degraded file exists
            if not os.path.isfile(degraded_file):
                print(f"Degraded file {degraded_file} does not exist. Stopping...")
                break

            # 2. Load reference and degraded files
            deg_wav, deg_sr = AudioLoader.load_audio(degraded_file)
            ref_wav, _ = AudioLoader.load_audio(input_file, target_sr=deg_sr)  # Resample reference to match degraded sample rate if necessary

            # Trim to ensure same length
            if len(ref_wav) != len(deg_wav):
                min_length = min(len(ref_wav), len(deg_wav))
                ref_wav = ref_wav[:min_length]
                deg_wav = deg_wav[:min_length]

            # Convert to tensor
            ref_tensor = torch.tensor(ref_wav, dtype=torch.float32)
            deg_tensor = torch.tensor(deg_wav, dtype=torch.float32)

            # Compute all metrics for current sample
            for metric in self.metrics:
                score = metric.calculate(ref_tensor, deg_tensor)
                if score is not None:
                    print(f"{metric.name} score for {degraded_file}: {score}")

            if samples > 0 and idx == samples:
                break

    def display_results(self, bitrate=None):
        """
        :param bitrate: Bitrate of the degraded audio files (optional, for display purposes)
        :return: None
        """
        print(f"\nResults for bitrate: {bitrate}")
        for metric in self.metrics:
            print(f"Processed {metric.count()} samples. Average {metric.name} score: {metric.average_score()}")

    def get_avg_scores(self):
        """
        :return: Dictionary of average scores for each metric
        """
        return {metric.name: metric.average_score() for metric in self.metrics}


def plot_results(results):
    """
    :param results: Dictionary of the form {metric_name: {bitrate: avg_score}}
    """
    for metric_name, scores in results.items():
        plt.plot(list(scores.keys()), list(scores.values()), marker='o', label=metric_name)
    
    plt.title('Average Metric Scores by Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid()
    plt.show()

    # save figure
    plt.savefig('metrics.png')


def main(format, metric, bitrate=["16k"], samples=0):
    """
    :param format: Format of the degraded audio files (e.g., 'opus')
    :param bitrate: List of bitrates of the degraded audio files (e.g., ['16k', '32k'])
    :param samples: Number of samples to process for each metric (0 for all samples)
    :param metric: List of metric names to calculate (e.g., ['PESQ', 'STOI'])
    """
    
    print("Starting audio analysis...")

    input_folder = "./data_source/clean/wavs/"
    results = defaultdict(dict)

    for bitrate in bitrate:
        degraded_folder = f"./data_source/{format}/{bitrate}/"
        metrics = MetricFactory.create_metrics(metric)
        analyzer = AudioAnalyzer(input_folder, degraded_folder, metrics)
        analyzer.analyze(format, samples)
        analyzer.display_results(bitrate)
        
        # Store average scores for graphing
        avg_scores = analyzer.get_avg_scores()
        for metric_name, avg_score in avg_scores.items():
            results[metric_name][bitrate] = avg_score
    
    # Plot results
    plot_results(results)

    print("Audio analysis completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Analysis Script")
    parser.add_argument("--format", type=str, required=True, help="Specify format to convert (e.g., wav, opus) <Required>")
    parser.add_argument("--metric", type=str, nargs="+", required=True, help="Specify metric to calculate (e.g., PESQ, STOI, SI-SDR, SI-SNR) <Required>")
    parser.add_argument("--bitrate", type=int, nargs="+", default=[16], help="Specify bitrate (e.g., 16) <Optional>")
    parser.add_argument("--samples", type=int, default=0, help="Specify number of samples to calculate PESQ for (e.g., 100) <Optional>")
    args = parser.parse_args()

    if args.bitrate:
        args.bitrate = [f"{bitrate}k" for bitrate in args.bitrate]

    main(args.format, args.metric, args.bitrate, args.samples)
