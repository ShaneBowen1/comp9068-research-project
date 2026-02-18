import argparse
import os
import resampy
import numpy as np
import soundfile as sf
import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='nb')  # Initialize PESQ metric with default sample rate and narrowband mode

torch.set_printoptions(precision=8)

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

def main(target_format, target_bitrate="16k", num_of_samples=0):

    assert target_format, "Target format is required. Please specify using --target_format."

    # Array for storing PESQ scores
    pesq_scores = []

    input_folder = "./data_source/clean/wavs/"
    degraded_folder = f"./data_source/{target_format}/{target_bitrate}/"

    # 1. Read input files
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_folder, filename)
            degraded_file = os.path.join(degraded_folder, filename.replace(".wav", f".{target_format}"))

        # check if degraded file exists
        if not os.path.isfile(degraded_file):
            print(f"Degraded file {degraded_file} does not exist. Stopping...")
            if len(pesq_scores) > 0:
                print(f"Processed {len(pesq_scores)} samples. Average PESQ score: {round(sum(pesq_scores) / len(pesq_scores), 4)}")
            break

        # 2. Load reference and degraded files
        deg_wav, deg_sr = load_audio(degraded_file)
        ref_wav, _ = load_audio(input_file, target_sr=deg_sr)  # Resample reference to match degraded sample rate if necessary

        # Trim to ensure same length
        if len(ref_wav) != len(deg_wav):
            min_length = min(len(ref_wav), len(deg_wav))
            ref_wav = ref_wav[:min_length]
            deg_wav = deg_wav[:min_length]

        # Convert to tensor
        ref_tensor = torch.tensor(ref_wav, dtype=torch.float32)
        deg_tensor = torch.tensor(deg_wav, dtype=torch.float32)

        # 3. Calculate PESQ score
        try:
            pesq_score = pesq(ref_tensor, deg_tensor).item()
            pesq_scores.append(pesq_score)
            print(f"PESQ score for {degraded_file}: {pesq_score}")
        except Exception as e:
            print(f"An error occurred while calculating PESQ score for {degraded_file}: {str(e)}")
            continue

        if len(pesq_scores) == num_of_samples:
            print(f"Processed {len(pesq_scores)} samples. Average PESQ score: {round(sum(pesq_scores) / len(pesq_scores), 4)}")
            break

        input("Press Enter to continue...")

    if len(pesq_scores) > 0:
        print(f"Processed {len(pesq_scores)} samples. Average PESQ score: {round(sum(pesq_scores) / len(pesq_scores), 4)}")
    print("Audio analysis completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Analysis Script")
    parser.add_argument("--target_format", type=str, required=True, help="Specify format to convert (e.g., wav, opus) <Required>")
    parser.add_argument("--target_bitrate", type=int, default=16, help="Specify bitrate (e.g., 16) <Optional>")
    parser.add_argument("--num_of_samples", type=int, default=0, help="Specify number of samples to calculate PESQ for (e.g., 100) <Optional>")
    args = parser.parse_args()

    if args.target_bitrate:
        args.target_bitrate = f"{args.target_bitrate}k"
    target_bitrate = args.target_bitrate

    main(args.target_format, args.target_bitrate, args.num_of_samples)
