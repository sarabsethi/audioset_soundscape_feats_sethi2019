# Calculating acoustic features of your audio data. 

This is heavily based on the official VGGish documentation at https://github.com/tensorflow/models/tree/master/research/audioset/vggish. We will cover everything you need to install dependencies etc and start extracting audio features from your data. Please ensure you have [Anaconda](https://www.anaconda.com/) installed before starting

### 1. Create a new python 3 conda environment

`conda create --name test-vggish-env python=3.6`

### 2. Activate your new environment

`source activate test-vggish-env`

(if this doesn't work, try "activate test-vggish-env" or "conda activate test-vggish-env" )

### 3. Install the required packages

`conda install -c numba numba`

`pip install numpy resampy tensorflow==1.15 tf_slim six soundfile numba==0.43.0 llvmlite==0.32.1`

`conda install -c conda-forge librosa`

### 4. Test everything is working correctly

`python vggish_smoke_test.py`

If you see "Looks Good To Me", then you're all setup correctly

### 5. Use `test_calc_features.py` as a template to calculate audio features from your own data
