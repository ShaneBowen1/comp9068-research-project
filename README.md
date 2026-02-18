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
        - `--volume` Mount the */data_source* folder to read input files and save transcoded files
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source test python transcode_audio.py
        ```

    - audio_analysis.py
        - `--volume` Mount the */data_source* folder to read input files
        - `--target_format` Specify format to analyse (e.g., wav, opus) *Required
        - `--target_bitrate` Specify bitrate (e.g., 128) *Optional **Default=16
        - `--num_of_samples` Specify number of samples (e.g., 100) *Optional **Default=all
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source test python audio_analysis.py --target_format opus
        ```
