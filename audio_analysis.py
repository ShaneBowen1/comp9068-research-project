import argparse
import os
import time
import resampy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import torch
from collections import defaultdict
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalDistortionRatio,
    SignalNoiseRatio
)

custom_labels = {
    'normalized_PESQ': 'PESQ',
    'normalized_SI-SDR': 'SI-SDR',
    'normalized_SI-SNR': 'SI-SNR',
    'normalized_SDR': 'SDR',
    'normalized_SNR': 'SNR'
}

class AudioLoader:
    @staticmethod
    def load_audio(file_path, target_sr=None):
        """
        :param file_path: Path to the audio file
        :param target_sr: Target sample rate (optional)
        :return: Normalized audio waveform as a float32 numpy array and the sample rate
        """
        wav_data, sr = sf.read(file_path, dtype='float32')  # Normalized -1.0 to 1.0

        # Resample if target sample rate is specified
        if target_sr and sr != target_sr:
            wav_data = resampy.resample(wav_data, sr, target_sr)
            sr = target_sr

        # Convert to mono if audio is stereo
        if wav_data.ndim > 1:
            wav_data = np.mean(wav_data, axis=1)  # Convert to mono by averaging channels

        return wav_data, sr


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
            'SI-SNR': ScaleInvariantSignalNoiseRatio(),
            'SDR': SignalDistortionRatio(),
            'SNR': SignalNoiseRatio()
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
        print()

    def get_avg_scores(self):
        """
        :return: Dictionary of average scores for each metric
        """
        return {metric.name: metric.average_score() for metric in self.metrics}

def plot_results(results_df, title, xlabel, ylabel, y_limit=None):
    """
    :param results_df: DataFrame of the form {bitrate: {metric_name: avg_score}}
    :param title: Title of the plot 
    :param xlabel: Label for x-axis
    :param ylabel: Label for y-axis
    :param y_limit: Tuple specifying y-axis limits (optional)
    """

    # Custom labels for the legend
    results_df = results_df.rename(columns=custom_labels)

    # Sort Bitrate column from highest to lowest for better visualization
    results_df = results_df.sort_index(ascending=False)

    for col in results_df.columns:
        label = f"{col[0]} ({col[1]})" if isinstance(results_df.columns, pd.MultiIndex) else col
        plt.plot(results_df.index, results_df[col], marker='o', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # Set y-axis limits if specified
    if y_limit:
        plt.ylim(y_limit)

    # check if results directory exists, if not create it
    if not os.path.exists('./results/plots'):
        os.makedirs('./results/plots')

    # save figure
    plt.savefig(f'./results/plots/{title}_{int(time.time())}.png')
    plt.show()
    plt.close()

def normalise_and_plot_results(results_df):
    """
    :param results_df: DataFrame of the form {bitrate: {metric_name: avg_score}}
    """

    for codec, df in results_df.items():
        print(f"\nNormalised results for codec: {codec}")
        if 'PESQ' in df.columns:
            MIN = -0.5
            MAX = 4.5
            normalized_pesq = (df['PESQ'] - MIN) / (MAX - MIN)  # Normalize to [0, 1]
            normalized_df = normalized_pesq.to_frame()  # Convert Series to DataFrame for plotting
            plot_results(normalized_df, title=f"{codec} - Normalised PESQ Scores", xlabel="Bitrate (kbps)", ylabel="Normalised Score", y_limit=(0, 1))

        if all(col in df.columns for col in ['PESQ', 'STOI']):
            # Combine normalized PESQ and STOI into a single DataFrame for plotting
            combined_normalized_df = pd.DataFrame({
                'PESQ': normalized_pesq,
                'STOI': df['STOI']  # STOI is already in the range [0, 1]
            })
            plot_results(combined_normalized_df, title=f"{codec} - Normalised PESQ and STOI Scores", xlabel="Bitrate (kbps)", ylabel="Normalised Score", y_limit=(0, 1))

        if any(col in df.columns for col in ['SI-SDR', 'SI-SNR', 'SDR', 'SNR']):
            normalized_df = pd.DataFrame()

            df = df[[col for col in ['SI-SDR', 'SI-SNR', 'SDR', 'SNR'] if col in df.columns]]  # Select only the columns that are present in results_df
            min_score = df.min().min()  # Get the minimum score across all metrics
            max_score = df.max().max()  # Get the maximum score across all metrics

            for metric_name, series in df.items():
                normalised_metric = (series - min_score) / (max_score - min_score)  # Normalize to [0, 1]
                normalized_df[metric_name] = normalised_metric  # Add to DataFrame with metric name as column name
            plot_results(normalized_df, title=f"{codec} - Normalised {', '.join(normalized_df.columns)} Scores", xlabel="Bitrate (kbps)", ylabel="Normalised Score", y_limit=(0, 1))

def run_avg_score_calculation(format, metric, bitrates=["16k"], codecs=["libopus"], application="audio", samples=0):
    """
    :param format: Format of the degraded audio files (e.g., 'opus')
    :param metric: List of metric names to calculate (e.g., ['PESQ', 'STOI'])
    :param bitrates: List of bitrates of the degraded audio files (e.g., ['16k', '32k'])
    :param codecs: List of codecs for encoding (e.g., ['libopus'])
    :param application: Application type for encoding (e.g., 'voip', 'audio', 'lowdelay')
    :param samples: Number of samples to process for each metric (0 for all samples)
    :return results_dfs: Dictionary of DataFrames containing average scores for each codec and bitrate
    """

    input_folder = "./data_source/clean/wavs/"
    results_dfs = {}  # Dict to store average scores for each bitrate and metric
    results = defaultdict(dict)

    for codec in codecs:
        print(f"Calculating average scores for codec: {codec}")
        for bitrate in bitrates:
            degraded_folder = f"./data_source/{codec}/{application}/{bitrate}/"
            metrics = MetricFactory.create_metrics(metric)
            analyzer = AudioAnalyzer(input_folder, degraded_folder, metrics)
            analyzer.analyze(format, samples)
            analyzer.display_results(bitrate)

            # Store average scores for graphing
            avg_scores = analyzer.get_avg_scores()
            for metric_name, avg_score in avg_scores.items():
                results[metric_name][bitrate] = avg_score

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Store in dfs
        results_dfs[codec] = results_df

    return results_dfs

def plots_per_codec(results_dfs):
    """
    :param results_dfs: Dictionary of DataFrames containing average scores for each codec and bitrate
    """
    for codec, df in results_dfs.items():

        # Plot All metrics on a single graph
        plot_results(df, title=f"{codec} - Mean {', '.join(df.columns)} Scores", xlabel="Bitrate (kbps)", ylabel="Score")

        # Individual plots for each metric
        if 'PESQ' in df.columns:
            plot_results(df['PESQ'].to_frame(), title=f"{codec} - Mean PESQ Scores", xlabel="Bitrate (kbps)", ylabel="PESQ Score")
        if 'STOI' in df.columns:
            plot_results(df['STOI'].to_frame(), title=f"{codec} - Mean STOI Scores", xlabel="Bitrate (kbps)", ylabel="STOI Score")
        if any(col in df.columns for col in ['SI-SDR', 'SI-SNR', 'SDR', 'SNR']):
            df = df[[col for col in ['SI-SDR', 'SI-SNR', 'SDR', 'SNR'] if col in df.columns]]
            plot_results(df, title=f"{codec} - Mean {', '.join(df.columns)} Scores", xlabel="Bitrate (kbps)", ylabel="Score (dB)")

def merge_results_df_and_plot(results_dfs):
    """
    :param results_dfs: Dictionary of DataFrames containing average scores for each codec and bitrate
    """

    # use plot_results to plot all codecs on the same graph for each metric    
    merged_df = pd.concat(results_dfs.values(), keys=results_dfs.keys(), names=['Codec', 'Bitrate'])
    merged_df.reset_index(inplace=True)

    for metric_name in merged_df.columns[2:]:  # Skip 'Codec' and 'Bitrate' columns
        
        pivot_df = merged_df.pivot(index='Bitrate', columns='Codec', values=metric_name)
        plot_results(pivot_df, title=f"Comparison of {metric_name} Scores Across Codecs", xlabel="Bitrate (kbps)", ylabel=f"{metric_name} Score")

def normalise_and_merge_plot_results(results_dfs):
    """
    :param results_dfs: Dictionary of DataFrames containing average scores for each codec and bitrate
    """
    merged_df = pd.concat(results_dfs.values(), keys=results_dfs.keys(), names=['Codec', 'Bitrate'])
    merged_df.reset_index(inplace=True)

    if 'PESQ' in merged_df.columns:
        MIN = -0.5
        MAX = 4.5
        merged_df['normalized_PESQ'] = (merged_df['PESQ'] - MIN) / (MAX - MIN)  # Normalize to [0, 1]
        pesq_pivot_df = merged_df.pivot(index='Bitrate', columns='Codec', values='normalized_PESQ')
        plot_results(pesq_pivot_df, title=f"Comparison of Normalised PESQ Scores Across Codecs", xlabel="Bitrate (kbps)", ylabel=f"Normalised PESQ Score", y_limit=(0, 1))

    # only include STOI and PESQ if both are present in the merged_df
    if all(col in merged_df.columns for col in ['PESQ', 'STOI']):
        combined_pivot_df = merged_df.pivot(index='Bitrate', columns='Codec', values=['normalized_PESQ', 'STOI'])
        plot_results(combined_pivot_df, title=f"Comparison of Normalised PESQ and STOI Scores Across Codecs", xlabel="Bitrate (kbps)", ylabel="Normalised Score", y_limit=(0, 1))

    if any(col in merged_df.columns for col in ['SI-SDR', 'SI-SNR', 'SDR', 'SNR']):
        columns_to_normalize = [col for col in ['SI-SDR', 'SI-SNR', 'SDR', 'SNR'] if col in merged_df.columns]

        df = merged_df[columns_to_normalize]  # Select only the columns that are present in results_df
        min_score = df.min().min()  # Get the minimum score across all metrics
        max_score = df.max().max()  # Get the maximum score across all metrics

        for metric_name in columns_to_normalize:
            merged_df[f'normalized_{metric_name}'] = (merged_df[metric_name] - min_score) / (max_score - min_score)  # Normalize to [0, 1]

        normalized_pivot_df = merged_df.pivot(index='Bitrate', columns='Codec', values=[f'normalized_{col}' for col in columns_to_normalize])
        plot_results(normalized_pivot_df, title=f"Comparison of Normalised {', '.join([f'{col}' for col in columns_to_normalize])} Scores Across Codecs", xlabel="Bitrate (kbps)", ylabel="Normalised Score", y_limit=(0, 1))

def main(format, metric, bitrates=["16k"], codecs=["libopus"], application="audio", samples=0):
    """
    :param format: Format of the degraded audio files (e.g., 'opus')
    :param bitrates: List of bitrates of the degraded audio files (e.g., ['16k', '32k'])
    :param codecs: List of codecs for encoding (e.g., ['libopus'])
    :param application: Application type for encoding (e.g., 'voip', 'audio', 'lowdelay')
    :param samples: Number of samples to process for each metric (0 for all samples)
    :param metric: List of metric names to calculate (e.g., ['PESQ', 'STOI'])
    """

    print("Starting audio analysis...")

    # Calculate average scores for each metric and bitrate
    results_dfs = run_avg_score_calculation(format, metric, bitrates, codecs, application, samples)

    # Plotting per codec results
    plots_per_codec(results_dfs)

    # Normalize and plot results for each codec
    normalise_and_plot_results(results_dfs)

    # Merge codecs into a single DataFrame for comparison
    merge_results_df_and_plot(results_dfs)

    # Normalize merged codecs and plot comparison
    normalise_and_merge_plot_results(results_dfs)

    print("Audio analysis completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Analysis Script") 
    parser.add_argument("--format", type=str, required=True, help="Specify format to convert (e.g., wav, opus) <Required>")
    parser.add_argument("--metric", type=str, nargs="+", required=True, help="Specify metric to calculate (e.g., PESQ, STOI, SI-SDR, SI-SNR) <Required>")
    parser.add_argument("--bitrate", type=int, nargs="+", default=[16], help="Specify bitrate (e.g., 16) <Optional>")
    parser.add_argument("--codec", type=str, nargs="+", default=["libopus"], help="Method for encoding (e.g., libopus)")
    parser.add_argument("--application", type=str, default="audio", help="Application type for encoding (e.g., voip, audio, lowdelay)")
    parser.add_argument("--samples", type=int, default=0, help="Specify number of samples to calculate PESQ for (e.g., 100) <Optional>")
    args = parser.parse_args()

    if args.bitrate:
        args.bitrate = [f"{bitrate}k" for bitrate in args.bitrate]

    main(args.format, args.metric, args.bitrate, args.codec, args.application, args.samples)
