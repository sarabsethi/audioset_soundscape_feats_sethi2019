from analysis_libs import multi_class_classification, change_lab_type
from plot_libs import plot_multi_class_recalls
import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
import os
import pickle

'''
Multiclass classification problems using eco-acoustic features
'''

matplotlib.rcParams.update({'font.size': 24})
plt.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('font', family='serif')

feats = ['raw_audioset_feats_300s','v3_comp_sscape_idx_299s']
all_plots = [{'title':'Ithaca, USA: biodiversity', 'dts':'cornell_sorted_balanced_data', 'label_type':'land-use-ny'},
             {'title':'Sabah, MY: habitat quality', 'dts':'audio_moths_sorted_june2019', 'label_type':'land-use'},
             {'title':'Ithaca, USA: monthly', 'dts':'strictbal_cornell_seasonal_mic', 'label_type':'monthly'},
             {'title':'Sabah, MY: hourly', 'dts':'strictbal__specAM-VJR-1audio_moths_sorted_june2019', 'label_type':'hourly'}
             ]
panel_labels = ['(a)','(b)','(c)','(d)']

# How many training test splits - recommend 5
k_folds = 5

# Figure setup
n_subplots_x = 2
n_subplots_y = 2
subplt_idx = 1

fig = plt.figure(figsize=(18,10))

for plot in all_plots:
    # Plot predictions
    fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
    subplt_idx += 1

    ax = plt.gca()
    ax.text(-0.1, 1.15, panel_labels[subplt_idx-2], transform=ax.transAxes,
      fontsize=28, fontweight='bold', va='top', ha='right')

    for f in feats:
        # Load data from pickle files
        with open(os.path.join('multiscale_data','{}_{}.pickle'.format(plot['dts'],f)), 'rb') as savef:
            audio_feats_data, labels, datetimes, recorders, unique_ids, classes, mins_per_feat = pickle.load(savef)

        new_label_ixs, new_classes = change_lab_type(labels,datetimes,recorders,classes,unique_ids,type=plot['label_type'])
        new_labels = new_classes[new_label_ixs]

        cm, cm_labs, average_acc, accuracies = multi_class_classification(audio_feats_data, new_labels, k_fold=k_folds)

        plot_multi_class_recalls(accuracies, cm_labs, average_acc, plot['label_type'], f)
        ax.set_title(plot['title'])

    if subplt_idx == 2 or subplt_idx == 4:
        ax.set_ylabel('F1 score ($\%$)')

    if plot['label_type'] == 'land-use-ny': ax.set_xlabel('Avian richness (species per hour)')
    if plot['label_type'] == 'land-use': ax.set_xlabel('Above ground biomass ($log_{10}(t.ha^{-1})$)')
    if plot['label_type'] == 'monthly': ax.set_xlabel('Month')
    if plot['label_type'] == 'hourly': ax.set_xlabel('Time of day (hour)')

plt.tight_layout()
fig_savefile = os.path.join('figs','multiclass.svg')
plt.savefig(fig_savefile, format="svg")

plt.show()
