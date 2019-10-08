from plot_libs import plot_low_dim_space
from analysis_libs import get_embedded_data, change_lab_type, get_audio_mean_data, least_common_values
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import numpy as np
import matplotlib
from imblearn.under_sampling import RandomUnderSampler

'''
Unsupervised dimensionality reduction of eco-acoustic features

Reproduces Fig. 2
'''

plot_scale = 1.3

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend',fontsize='smaller')

# Feature sets and dimensionality reduction techniques to test
feats = ['raw_audioset_feats_300s']
all_plots = [{'title': 'Global', 'dts_name':'cornell_sorted_balanced_data+audio_moths_sorted_june2019+cornell_seasonal_mic+PC_recordings+sulawesi_sorted_data+wrege_africa_data+cornell_nz_data_sorted', 'label_type':'dataset', 'dimred':'umap_vis'},
            {'title': 'Borneo land-use', 'dts_name':'audio_moths_sorted_june2019', 'label_type':'land-use', 'dimred':'umap_vis_landuse'},
            {'title': 'New York seasonal', 'dts_name':'strictbal_cornell_seasonal_mic', 'label_type':'month', 'dimred':'umap_vis'},
            {'title': 'Borneo diurnal', 'dts_name':'strictbal__specAM-VJR-1audio_moths_sorted_june2019', 'label_type':'hour', 'dimred':'umap_vis'}
            ]

# Figure setup
fig_s = 6
n_subplots_x = np.min([len(all_plots),2])
n_subplots_y = np.ceil(len(all_plots) / n_subplots_x)
subplt_idx = 1
fig = plt.figure(figsize=((fig_s+1)*(n_subplots_x),fig_s*n_subplots_y))

for pl in all_plots:

    for f in feats:
        # Load data from pickle files
        with open(os.path.join('data','{}_{}.pickle'.format(pl['dts_name'],f)), 'rb') as savef:
            audio_feats_data, labels, datetimes, recorders, unique_ids, classes, mins_per_feat = pickle.load(savef)
            labels, classes = change_lab_type(labels,datetimes,recorders,classes,unique_ids,type=pl['label_type'])

        if pl['label_type'] == 'dataset':
            print(least_common_values(classes[labels]))
            print('Orig shape {}'.format(audio_feats_data.shape))
            rus = RandomUnderSampler(random_state=42)
            audio_feats_data, labels = rus.fit_sample(audio_feats_data, labels)
            print('Bal shape {}'.format(audio_feats_data.shape))

        # Embed data using UMAP or PCA
        data_red, data_red_labels = get_embedded_data(data=audio_feats_data,labels=labels,dimred=pl['dimred'])

        # Plot embedded data
        fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
        subplt_idx+=1
        plot_low_dim_space(data_red, data_red_labels, classes=classes,
                           dimred=pl['dimred'], plot_scale=plot_scale,
                           label_type=pl['label_type'])
        plt.title(pl['title'])

plt.tight_layout()
fig_savefile = os.path.join('figs','scatter.pdf')
plt.savefig(fig_savefile, format="pdf")
plt.show()
