from plot_libs import get_feats_nice_name, plot_regr_dims
from analysis_libs import get_embedded_data, get_audio_mean_data, reg_df_col_with_features, data_variance_explained, calc_added_dims_regs
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pandas as pd
import pickle

'''
Regress acoustic features against field data. See how number of dimensions of
embedded acoustic features used affects regression performance

Reproduces Fig. X from publication
'''

# Feature sets to test
feats = ['soundscape_vec', 'mean_raw_audioset_feats']

field_df = pd.read_pickle(os.path.join('data','field_data_public.pickle'))
agb_df = pd.read_pickle(os.path.join('data','agb_public.pickle'))

# Argument to change dimensionality reduction technique - PCA and UMAP similar for
# regressions so default to simplest option: PCA
parser = argparse.ArgumentParser()
parser.add_argument('-dimred', default='pca', type=str, help='Method of dimensionality reduction to use')
args = parser.parse_args()

# Figure setup
fig_s = 6
n_subplots_x = len(feats)
n_subplots_y = 1
subplt_idx = 1
fig = plt.figure(figsize=(fig_s*n_subplots_x*1.5,fig_s*n_subplots_y))

for f in feats:
    with open(os.path.join('data','{}_nowater.pickle'.format(f)), 'rb') as savef:
        audio_feats_data, labels, classes, mins_per_feat = pickle.load(savef)

    n_feats = audio_feats_data.shape[1]

    # Calculated mean embedded feature data
    data_red, data_red_labels = get_embedded_data(data=audio_feats_data,labels=labels,classes=classes,dims=n_feats,dimred=args.dimred)
    audio_mean_data, audio_mean_classes = get_audio_mean_data(data_red, labels=data_red_labels, classes=classes)

    # Calculate regression scores for each added dimension
    agb_scores, agb_pvals, agb_dims_x = calc_added_dims_regs(audio_mean_data, audio_mean_classes, agb_df, 'Mean AGB')
    alpha_scores, alpha_pvals, alpha_dims_x = calc_added_dims_regs(audio_mean_data, audio_mean_classes, field_df, 'Alpha diversity (tot)')

    fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
    subplt_idx += 1

    # Plot variance explained for acoustic feature data dimensions
    var_expl_audio = data_variance_explained(audio_feats_data)
    plt.plot(range(1,len(var_expl_audio)+1),var_expl_audio,c='darkgreen',ls=':',label='Acoustic $\sigma^2$ expl.')

    plot_regr_dims(agb_scores, agb_pvals, agb_dims_x, args.dimred, colour='red', legend_txt='AGB')
    plot_regr_dims(alpha_scores, alpha_pvals, alpha_dims_x, args.dimred, colour='blue', legend_txt='AGB')

    plt.title('{}'.format(get_feats_nice_name[f]))

plt.show()
