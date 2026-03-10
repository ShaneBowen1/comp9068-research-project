# COMP9068 Research Project
A Generative AI Approach for Audio Restoration due to Compression for Speech Enhancement

## Usage

1. Open up the terminal, go to the root directory (comp9068-research-project)

2. Build the Docker image:
    ```
    docker build -t test . -f Dockerfile
    ```

3. Run scripts:
    - transcode_audio.py
        - `--volume` Mount the */data_source* folder to read input files and save 
        transcoded files
        - `--format` Specify audio format to transcode (e.g., mp3, opus) *Required
        - `--bitrate` Specify bitrate to transcode (e.g., 128) **Default=\[16\]
        - `--sample_rate` Specify sample rate to transcode (e.g., 22050) *Optional
        - `--channels` Specify number of audio channels  (e.g., 1 for mono, 2 for stereo) **Default=1
        - `--codec` Specify method for encoding (e.g., libopus) **Default=libopus
        - `--application` Application type for encoding (e.g., voip, audio, lowdelay) **Default=audio
        - `--samples` Specify number of samples (e.g., 100) **Default=all
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source test python transcode_audio.py --format opus
        ```

    - audio_analysis.py
        - `--volume` Mount the */data_source* folder to read input files
        - `--format` Specify format to analyse (e.g., wav, opus) *Required
        - `--metric` Specify metric to calculate (e.g., PESQ, STOI) *Required
        - `--bitrate` Specify bitrate (e.g., 128) *Optional **Default=16
        - `--codec` Specify method for encoding (e.g., libopus) **Default=libopus
        - `--application` Application type for encoding (e.g., voip, audio, lowdelay) **Default=audio
        - `--samples` Specify number of samples (e.g., 100) **Default=all
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source test python audio_analysis.py --format opus --metric PESQ
        ```
