import argparse
import os
import ffmpeg


def main(output_kwargs, samples=0):
    """
    :param output_kwargs: Dictionary of ffmpeg output parameters (e.g., format, bitrate, sample rate, channels, codec)
    :param samples: Number of samples to convert (0 for all)
    """

    print("Starting audio transcoding...")

    # 1. Read input files
    input_folder = "./data_source/clean/wavs/"

    # 2. Convert audio files
    for idx, filename in enumerate(os.listdir(input_folder), start=1):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_folder, filename)

            output_folder = f"./data_source/{args.format}/{target_bitrate if target_bitrate else 'default'}/"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            f_name = os.path.splitext(filename)[0]

            output_file = os.path.join(output_folder, f"{f_name}.{args.format}")
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file, **output_kwargs)

            try:
                # 3. Run the ffmpeg command
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                print(f"Converted {input_file} to {output_file} in {args.format} format.")
            except ffmpeg.Error as e:

                if e.stderr:
                    print(f"An error occurred while converting {input_file}: {e.stderr.decode()}")
                
                if e.stdout:
                    print(f"ffmpeg output: {e.stdout.decode()}")
                
                raise RuntimeError(f"Failed to convert {input_file} to {output_file}.") from e

            if samples > 0 and idx == samples:
                break

            #input("Press Enter to continue...")

    print("Audio conversion completed.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audio Transcoding Script")
    parser.add_argument("--format", type=str, required=True, help="Target audio format (e.g., mp3, opus)")
    parser.add_argument("--bitrate", type=int, help="Target bitrate (e.g., 128)")
    parser.add_argument("--sample_rate", type=int, help="Target sample rate (e.g., 22050)")
    parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (e.g., 1 for mono, 2 for stereo)")
    parser.add_argument("--codec", type=str, help="Method for encoding (e.g., libopus)")
    parser.add_argument("--samples", type=int, default=0, help="Number of samples to convert (0 for all)")
    args = parser.parse_args()

    # Set target format
    output_kwargs = {
        "f": args.format
    }

    # Set target bitratet
    if args.bitrate:
        target_bitrate = f"{args.bitrate}k"
        output_kwargs["audio_bitrate"] = target_bitrate

    # Set target sample rate
    if args.sample_rate:
        output_kwargs["ar"] = args.sample_rate

    # Set target audio channels
    if args.channels:
        output_kwargs["ac"] = args.channels

    # Set method for encoding
    if args.codec:
        output_kwargs["acodec"] = args.codec
    
    main(output_kwargs, args.samples)
