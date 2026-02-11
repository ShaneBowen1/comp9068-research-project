import os
import ffmpeg
import resampy
from pesq import pesq
from scipy.io import wavfile

if __name__ == "__main__":

    # Array for storing PESQ scores
    pesq_scores = []

    # Set target format and bitrate for degraded audio files
    target_format = input("Specify format to convert <Required>: ")
    target_bitrate = input("Specify bitrate (e.g., 128) <Optional>: ")
    target_sample_rate = input("Specify sample rate (e.g., 44100) <Optional>: ")
    target_audio_channels = input("Specify number of audio channels (e.g., 2) <Optional>: ")  # 1 for mono, 2 for stereo
    num_of_samples = input("Specify number of samples to calculate PESQ for (e.g., 100) <Optional>: ")

    # Set default values for optional parameters if not provided
    if not target_bitrate:
        target_bitrate = "default"  # Default to 'default' if not specified
    elif target_bitrate and not target_bitrate.endswith("k"):
        target_bitrate += "k"
    if not target_sample_rate:
        target_sample_rate = 16000  # Default to 16 kHz if not specified
    if not target_audio_channels:
        target_audio_channels = "1"  # Default to mono if not specified
    num_of_samples = int(num_of_samples) if num_of_samples else 0  # Default to 0 (all samples) if not specified

    band = 'nb' if target_audio_channels == "1" else 'wb'  # Use narrowband for mono and wideband for stereo

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
                print(f"Average PESQ score for {len(pesq_scores)} samples: {round(sum(pesq_scores) / len(pesq_scores), 4)}")
            break

        f_name = os.path.splitext(filename)[0]

        # 2. Convert degraded audio file to WAV if necessary for PESQ calculation
        if not degraded_file.endswith(".wav"):
            temp_wav_file = os.path.join(degraded_folder, f"{f_name}_temp.wav")
            stream = ffmpeg.input(degraded_file)
            stream = ffmpeg.output(stream, temp_wav_file, acodec='pcm_s16le', ac=1, ar=target_sample_rate)
            try:
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                print(f"Converted {degraded_file} to temporary WAV file for PESQ calculation.")
            except ffmpeg.Error as e:
                print(f"An error occurred while converting {degraded_file} to WAV: {e.stderr.decode()}")
                continue

        # Read reference and degraded audio files
        if not degraded_file.endswith(".wav"):
            deg_rate, deg_data = wavfile.read(temp_wav_file)
            os.remove(temp_wav_file)  # Clean up temporary file
        else:
            deg_rate, deg_data = wavfile.read(degraded_file)

        # Resample reference audio if sample rates do not match
        ref_rate, ref_data = wavfile.read(input_file)
        if ref_rate != deg_rate:
            ref_data = resampy.resample(ref_data, ref_rate, deg_rate)

        # 3. Calculate PESQ score
        try:
            pesq_score = pesq(deg_rate, ref_data, deg_data, band)
            pesq_scores.append(pesq_score)
            print(f"PESQ score for {degraded_file}: {pesq_score}")
        except Exception as e:
            print(f"An error occurred while calculating PESQ score for {degraded_file}: {str(e)}")
            continue

        if len(pesq_scores) == num_of_samples:
            print(f"Average PESQ score for {len(pesq_scores)} samples: {round(sum(pesq_scores) / len(pesq_scores), 4)}")
            break

        input("Press Enter to continue...")

    print("Audio analysis completed.")
