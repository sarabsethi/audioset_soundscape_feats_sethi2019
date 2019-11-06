# audioset_soundscape_feats_sethi2019
Reproducability code for Sethi et. al. 2019 (in prep.)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3530203.svg)](https://doi.org/10.5281/zenodo.3530203)

## Setup
Data files are stored on Zenodo at:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3530206.svg)](https://doi.org/10.5281/zenodo.3530206)

Download the `.zip` file and extract all folders into this directory. We provide calculated features for all the audio used in our analyses in `.pickle` format.

Code is tested using Python 3.6 on an Anaconda installation

## Usage
Python scripts included are:
 
* `figure_scatter_lowdim.py` generates 2D scatters of audio features at various spatial and temporal scales. UMAP is used as a dimensionality reduction technique

* `figure_multiclass_predictions.py` performs a series of multiclass classification tasks, using a random forest classifier based on the acoustic features. F1 score for each class is plotted and compared for different feature sets

* `figure_anomalies.py` creates two panels of a larger figure. (1) A 2D representation of a Gaussian Mixture Model (GMM) fit to 5 full days of audio from a logged tropical rainforest site in Sabah, Malaysia. (2) Averaged anomaly scores of various playback sounds played at different distances from recorders at ten sites across a tropical rainforest study site in Sabah, Malaysia.

* `figure_anomaly_spectro_scores.py` generates spectrograms for playback experiments of chainsaw sounds played from a speaker at a variety of distances from a recorder in a logged tropical rainforest site in Sabah, Malaysia. Overlaid in red is the anomaly score assigned to each 0.96 s of the audio clip.

* `figure_anomaly_individual_playbacks.py` same as `figure_anomalies` panel (2) but per site results without averaging 

* `figure_feats_dendrogram.py` debdrogram showing similarity between different ecosystems in acoustic feature space

* `analysis_libs.py` and `plot_libs.py` contain auxiliary functions
