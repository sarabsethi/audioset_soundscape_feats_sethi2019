from plot_libs import plot_pdist_clusts, get_feats_nice_name
from analysis_libs import get_embedded_data, get_audio_mean_data, calc_var_info, data_variance_explained
import matplotlib.pyplot as plt
import argparse
from matplotlib import rcParams
import numpy as np
import pickle
import os
import pandas as pd

'''
Compare clusters of sites obtained from different feature sets

Reproduces Fig. X from publication
'''

# Feature sets to test
feats = ['soundscape_vec','mean_raw_audioset_feats']

# Argument to change dimensionality reduction technique - UMAP works best for clustering
parser = argparse.ArgumentParser()
parser.add_argument('-dimred', default='umap_clust', type=str, help='Method of dimensionality reduction to use')
args = parser.parse_args()

# Figure setup
fig_s = 4.5
n_subplots_x = 1 + len(feats)
n_subplots_y = 1
subplt_idx = 1
fig = plt.figure(figsize=(fig_s*(1.6*n_subplots_x),fig_s*n_subplots_y))

field_df = pd.read_pickle(os.path.join('data','field_data_public.pickle'))

for f in feats:
    # Load data from pickle file
    with open(os.path.join('data','{}.pickle'.format(f)), 'rb') as savef:
        audio_feats_data, labels, classes, mins_per_feat = pickle.load(savef)

    # Embed data and calculate mean feature vectors per site
    dims = audio_feats_data.shape[1]
    data_red, data_red_labels = get_embedded_data(data=audio_feats_data,labels=labels,classes=classes,dims=dims,dimred=args.dimred)
    audio_mean_data, audio_mean_labels = get_audio_mean_data(data_red, labels=data_red_labels, classes=classes)

    # Get clusters and cluster similarity to species community data
    aud_cl, aud_pdist, aud_labs, field_cl, field_pdist, field_labs, vi_pval = calc_var_info(audio_mean_data, audio_mean_labels, field_df)

    # Plot clusters from audio feature data
    fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
    subplt_idx+=1
    plot_pdist_clusts(aud_cl, aud_pdist, aud_labs, '{} (p = {})'.format(get_feats_nice_name[f],vi_pval))

# Plot clusters from species community data ("ground truth" of sorts)
fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
subplt_idx+=1
plot_pdist_clusts(field_cl, field_pdist, field_labs, 'Species community clustering')

plt.subplots_adjust(top=0.85)
plt.suptitle('{} clustering'.format(args.dimred))

plt.show()
