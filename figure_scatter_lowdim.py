from plot_libs import plot_low_dim_space
from analysis_libs import get_embedded_data, change_lab_type, least_common_values
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import numpy as np
import matplotlib

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
supp_figure = False

if not supp_figure:
    all_plots = [{'title': 'Global', 'leg_title': 'Location', 'dts_name':'cornell_sorted_balanced_data+audio_moths_sorted_june2019+cornell_seasonal_mic+PC_recordings+sulawesi_sorted_data+wrege_africa_data+cornell_nz_data_sorted', 'label_type':'dataset', 'dimred':'umap_vis'},
                {'title': 'Sabah, MY: habitat quality', 'leg_title': 'AGB', 'dts_name':'audio_moths_sorted_june2019', 'label_type':'land-use', 'dimred':'umap_vis_landuse'},
                {'title': 'Ithaca, USA: seasonal', 'leg_title': 'Month', 'dts_name':'strictbal_cornell_seasonal_mic', 'label_type':'month', 'dimred':'umap_vis'},
                {'title': 'Sabah, MY: diurnal', 'leg_title': 'Hour', 'dts_name':'strictbal__specAM-VJR-1audio_moths_sorted_june2019', 'label_type':'hour', 'dimred':'umap_vis'}
                ]
    fig_savef = 'scatter'
else:
    all_plots = [{'title': 'Sulawesi, ID: diurnal', 'leg_title': 'Hour', 'dts_name':'strictbal__specT5sulawesi_sorted_data', 'label_type':'hour', 'dimred':'umap_vis'},
                 {'title': 'Abel Tasman, NZ: diurnal', 'leg_title': 'Hour', 'dts_name':'strictbal__specS01cornell_nz_data_sorted', 'label_type':'hour', 'dimred':'umap_vis'},
                 {'title': 'Nouabalé-Ndoki, COG: diurnal', 'leg_title': 'Hour', 'dts_name':'strictbal__specnn06f_mono_wetwrege_africa_data', 'label_type':'hour', 'dimred':'umap_vis'},
                 {'title': 'Ithaca, USA: diurnal', 'leg_title': 'Hour', 'dts_name':'strictbal__specS10cornell_sorted_balanced_data', 'label_type':'hour', 'dimred':'umap_vis'}
                 ]
    fig_savef = 'supp_scatter'

panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']

# Figure setup
fig_s = 6
n_subplots_x = np.min([len(all_plots),2])
n_subplots_y = np.ceil(len(all_plots) / n_subplots_x)
subplt_idx = 1
fig = plt.figure(figsize=((fig_s+1)*(n_subplots_x),fig_s*n_subplots_y),dpi=300)

for pl in all_plots:

    for f in feats:
        # Load data from pickle files
        with open(os.path.join('multiscale_data','{}_{}.pickle'.format(pl['dts_name'],f)), 'rb') as savef:
            audio_feats_data, labels, datetimes, recorders, unique_ids, classes, mins_per_feat = pickle.load(savef)
            labels, classes = change_lab_type(labels,datetimes,recorders,classes,unique_ids,type=pl['label_type'])

        # Embed data using UMAP or PCA
        if 'dataset' in pl['label_type']: balance_before = True
        else: balance_before = False

        data_red, data_red_labels = get_embedded_data(data=audio_feats_data,labels=labels,dimred=pl['dimred'],balance_before=balance_before)

        # Plot embedded data
        fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
        subplt_idx+=1

        ax = plt.gca()
        ax.text(-0.02, 1.05, panel_labels[subplt_idx-2], transform=ax.transAxes,
          fontsize=28, fontweight='bold', va='top', ha='right')

        plot_low_dim_space(data_red, data_red_labels, classes=classes,
                           dimred=pl['dimred'], plot_scale=plot_scale,
                           label_type=pl['label_type'], leg_title=pl['leg_title'])
        plt.title(pl['title'])

plt.tight_layout()
fig_savefile = os.path.join('figs','{}.svg'.format(fig_savef))
plt.savefig(fig_savefile, format="svg")
fig_savefile_pdf = os.path.join('figs','{}.pdf'.format(fig_savef))
plt.savefig(fig_savefile_pdf, format="pdf")
plt.show()
