# audioset_soundscape_feats_sethi2019
Reproducability code for Sethi et. al. 2019 (in prep.)

## Setup

* `git clone https://github.com/sarabsethi/audioset_soundscape_feats_sethi2019.git`
* Install dependencies (working on this list)
* Reproduce figures by running python scripts: e.g. `python figure_scatter_lowdim.py`

## Figures

Code is provided to reproduce all results figures in our publication. Once this repository is cloned and all dependencies are installed, you should be able to reproduce our exact plots by running the below files

These files use functions contained within `analysis_libs.py` and `plot_libs.py`

* `figure_scatter_lowdim.py`: Compare different acoustic feature sets and different dimensionality reduction techniques

* `figure_clusterings.py`: Compare clusters of sites obtained from different feature sets

* `figure_regression.py`: Regress acoustic features against field data. See how number of dimensions of embedded acoustic features used affects regression performance

* `figure_predictions.py`: Predict field data at each site using a regression model trained on all sites except the point in question. Plot true values against predicted values

## Data

Data is stored in the `data` directory in pickle files. 

### Field data

Field data is provided for each of our sampling sites

* `agb_public.pickle`: AGB data, taken as an average within 1km of each site (data from [Pfeifer et. al. 2015](http://iopscience.iop.org/article/10.1088/1748-9326/10/4/044019/meta))
* `field_data_public.pickle`: Species community data at each site

### Audio feature data

AudioSet features are a raw 128 feature embeddding from the trained VGGish network from Google's [AudioSet project](https://github.com/tensorflow/models/tree/master/research/audioset). This produces 1 feature per second, which we average over 20 minute chunks

Soundscape compound features are made up of: ADI, ACI, H, Spectral Entropy and Temporal Entropy (last three taken from Suer 2008). These are computed using the [Acoustic Indices package](https://github.com/patriceguyot/Acoustic_Indices). Again we average over 20 minute chunks of audio.

* `mean_raw_audioset_feats.pickle`: Raw AudioSet features 
* `mean_raw_audioset_feats_nowater.pickle`: Raw AudioSet features, excluding sites near water sources
* `soundscape_vec.pickle`: Soundscape compound indices
* `soundscape_vec_nowater.pickle`: Soundscape compound indices, excluding sites near water sources
