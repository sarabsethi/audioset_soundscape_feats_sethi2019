from analysis_libs import get_embedded_data, get_audio_mean_data, reg_df_col_with_features
from plot_libs import get_site_colour, get_feats_nice_name, plot_field_data_preds
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import os
import pickle

'''
Predict field data at each site using a regression model trained on all sites
except the point in question. Plot true values against predicted values

Reproduces Fig. X from publication
'''

field_df = pd.read_pickle(os.path.join('data','field_data_public.pickle'))
agb_df = pd.read_pickle(os.path.join('data','agb_public.pickle'))

# Audio feature sets to use
feats = ['soundscape_vec','mean_raw_audioset_feats']
dataset = 'PC_recordings'

dimred = 'pca'
dims = 3

# Figure setup
fig_s = 6
n_subplots_y = 1
n_subplots_x = len(feats)
subplt_idx = 1

fig = plt.figure(figsize=(fig_s*n_subplots_x*1.5,fig_s*n_subplots_y))

for f in feats:
    with open(os.path.join('data','{}_{}_nowater.pickle'.format(dataset,f)), 'rb') as savef:
        audio_feats_data, labels, classes, mins_per_feat = pickle.load(savef)

    n_feats = audio_feats_data.shape[1]

    # Get mean embedded feature vectors for each site
    data_red, data_red_labels = get_embedded_data(data=audio_feats_data,labels=labels,classes=classes,dims=dims,dimred=dimred)
    audio_mean_data, audio_mean_labels = get_audio_mean_data(data_red, labels=data_red_labels, classes=classes)

    # Plot predictions
    fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
    subplt_idx += 1
    plot_field_data_preds(audio_mean_data, audio_mean_labels, agb_df, 'Mean AGB')
    plt.title('{}'.format(get_feats_nice_name[f]))

plt.show()
