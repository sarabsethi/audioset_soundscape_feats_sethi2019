from plot_libs import plot_low_dim_space, get_feats_nice_name
from analysis_libs import get_embedded_data
import matplotlib.pyplot as plt
import argparse
from matplotlib import rcParams
import pickle
import os
import pandas as pd

'''
Compare different acoustic feature sets and different dimensionality reduction techniques

Reproduces Fig. X from publication
'''

# Feature sets and dimensionality reduction techniques to test
feats = ['soundscape_vec','mean_raw_audioset_feats']
dimreds = ['pca','umap_default']
dataset = 'PC_recordings'

# Figure setup
fig_s = 6
n_subplots_x = len(feats)
n_subplots_y = len(dimreds)
subplt_idx = 1
fig = plt.figure(figsize=(fig_s*(n_subplots_x),fig_s*n_subplots_y))

agb_df = pd.read_pickle(os.path.join('data','agb_public.pickle'))

for f in feats:
    for dimred in dimreds:
        # Load data from pickle files
        with open(os.path.join('data','{}_{}.pickle'.format(dataset,f)), 'rb') as savef:
            audio_feats_data, labels, classes, mins_per_feat = pickle.load(savef)

        # Embed data using UMAP or PCA
        data_red, data_red_labels = get_embedded_data(data=audio_feats_data,labels=labels,classes=classes,dimred=dimred)

        # Plot embedded data
        fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
        subplt_idx+=1
        title = '{}'.format(get_feats_nice_name[f])
        plot_low_dim_space(data_red, data_red_labels, classes=classes,
            plt_title=title, mins_per_feat=mins_per_feat, agb_df=agb_df, dimred=dimred)

        plt.legend().remove()

plt.show()
