import os
import ffmpeg

if __name__ == "__main__":

    # Read user inputs
    target_format = input("Specify format to convert <Required>: ")
    target_bitrate = input("Specify bitrate (e.g., 128) <Optional>: ")
    target_sample_rate = input("Specify sample rate (e.g., 44100) <Optional>: ")
    target_audio_channels = input("Specify number of audio channels (e.g., 2) <Optional>: ")  # 1 for mono, 2 for stereo
    method_for_encoding = input("Specify method for encoding (e.g., libopus) <Optional>: ")

    # Set target format
    output_kwargs = {
        "f": target_format
    }

    # Set target bitrate
    if target_bitrate and not target_bitrate.endswith("k"):
        target_bitrate += "k"
        output_kwargs["audio_bitrate"] = target_bitrate

    # Set target sample rate
    if target_sample_rate:
        output_kwargs["ar"] = target_sample_rate

    # Set target audio channels
    if target_audio_channels:
        output_kwargs["ac"] = target_audio_channels

    # Set method for encoding
    if method_for_encoding:
        output_kwargs["acodec"] = method_for_encoding

    # 1. Read input files
    input_folder = "./data_source/clean/wavs/"

    # 2. Convert audio files
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_folder, filename)

            output_folder = f"./data_source/{target_format}/{target_bitrate if target_bitrate else 'default'}/"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            f_name = os.path.splitext(filename)[0]

            output_file = os.path.join(output_folder, f"{f_name}.{target_format}")
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file, **output_kwargs)

            try:
                # 3. Run the ffmpeg command
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                print(f"Converted {input_file} to {output_file} in {target_format} format.")
            except ffmpeg.Error as e:

                if e.stderr:
                    print(f"An error occurred while converting {input_file}: {e.stderr.decode()}")
                
                if e.stdout:
                    print(f"ffmpeg output: {e.stdout.decode()}")
                
                raise RuntimeError(f"Failed to convert {input_file} to {output_file}.") from e

            input("Press Enter to continue...")

    print("Audio conversion completed.")
