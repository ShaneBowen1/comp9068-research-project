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
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source test python audio_analysis.py
        ```
