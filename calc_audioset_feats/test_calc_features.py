'''
# GET STARTED: this is heavily based on the official VGGish documentation at https://github.com/tensorflow/models/tree/master/research/audioset/vggish
# It covers everything you need to install dependencies etc and start extracting audio features from your data

# Create a new python 3 conda environment
conda create --name test-vggish-env python=3.6

# Activate your new environment
source activate test-vggish-env
# (if this doesn't work, try "activate test-vggish-env" or "coda activate test-vggish-env" )

# Install the required packages
conda install -c numba numba
pip install numpy resampy tensorflow==1.15 tf_slim six soundfile numba==0.43.0 llvmlite==0.32.1
conda install -c conda-forge librosa

# Installation ready, let's test it.
python vggish_smoke_test.py

# If we see "Looks Good To Me", then we're all setup correctly.
# Use the below script as a template to calculate features easily and at different timescales etc.
'''

from AudiosetAnalysis import AudiosetAnalysis
import os

# Get all mp3 or wav files in our audio directory
audio_dir = 'audio_dir'
all_fs = os.listdir(audio_dir)
audio_fs = [f for f in all_fs if '.wav' in f.lower() or '.mp3' in f.lower()]

# Setup the audioset analysis
an = AudiosetAnalysis()
an.setup()

# Analyse each audio file in turn, and print the shape of the results
for f in audio_fs:
    path = os.path.join(audio_dir, f)
    results = an.analyse_audio(path)

    # Results are stored in a dictionary - keys tell you the feature time resolutions in milliseconds
    for res in results.items():
        print('{}: {}'.format(res[0],res[1].shape))
