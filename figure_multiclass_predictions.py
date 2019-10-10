from analysis_libs import multi_class_classification, get_special_labels
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
plt.rc('font', family='serif')

feats = ['raw_audioset_feats_300s','v3_comp_sscape_idx_299s']
all_plots = [{'title':'Ithaca, USA: biodiversity', 'dts':'cornell_sorted_balanced_data', 'label_type':'land-use-ny'},
             {'title':'Sabah, MY: AGB', 'dts':'audio_moths_sorted_june2019', 'label_type':'land-use'},
             {'title':'Ithaca, USA: monthly', 'dts':'strictbal_cornell_seasonal_mic', 'label_type':'monthly'},
             {'title':'Sabah, MY: hourly', 'dts':'strictbal__specAM-VJR-1audio_moths_sorted_june2019', 'label_type':'hourly'}
             ]
# How many training test splits - recommend 5
k_folds = 1

# Figure setup
n_subplots_x = 2
n_subplots_y = 2
subplt_idx = 1

fig = plt.figure(figsize=(18,10))

for plot in all_plots:
    # Plot predictions
    fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
    subplt_idx += 1

    for f in feats:
        # Load data from pickle files
        with open(os.path.join('data','{}_{}.pickle'.format(plot['dts'],f)), 'rb') as savef:
            audio_feats_data, labels, datetimes, recorders, unique_ids, classes, mins_per_feat = pickle.load(savef)

        new_labels = get_special_labels(datetimes, recorders, unique_ids, type=plot['label_type'])
        cm, cm_labs, acc, recalls = multi_class_classification(audio_feats_data, new_labels, k_fold=k_folds)

        plot_multi_class_recalls(recalls, cm_labs, acc, plot['label_type'], f)
        plt.title(plot['title'])

    if subplt_idx == 2 or subplt_idx == 4:
        plt.ylabel('Recall (percentage)')

plt.tight_layout()
fig_savefile = os.path.join('figs','multiclass.pdf')
plt.savefig(fig_savefile, format="pdf")

plt.show()
