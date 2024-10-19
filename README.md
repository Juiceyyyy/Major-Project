# Major Project
 LSTM based human tracking

## Requirements
The code should be run with python . We use the aws cli to help download the data, please first run
`python -m pip install -r requirements.txt` to install it.

The code also requires ffmpeg to process some videos, please install it for your respective system 
and make sure ffmpeg and ffprobe are available in the command line path. Use your package manager 
or find other download options here: https://ffmpeg.org/download.html

## Data 
Data can be downloaded using the download.py script in this folder. Simply run:
`python download.py`
It will automatically download the dataset videos and annotations and extract them under
REPO_ROOT/dataset/raw_data REPO_ROOT/dataset/annotations respectively.