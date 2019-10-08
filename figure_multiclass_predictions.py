from analysis_libs import reg_df_col_with_features, change_lab_type, multi_class_classification, get_special_labels
from plot_libs import get_feats_nice_name, plot_field_data_preds, get_label_nice_name, plot_confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
import pandas as pd
import os
import pickle

'''
Multiclass classification problems using eco-acoustic features
'''

matplotlib.rcParams.update({'font.size': 24})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

feats = ['raw_audioset_feats_300s','v3_comp_sscape_idx_299s']
all_plots = [{'title':'New York: biodiversity', 'dts':'cornell_sorted_balanced_data', 'class_type':'land-use-ny'},
             {'title':'Borneo: AGB', 'dts':'audio_moths_sorted_june2019', 'class_type':'land-use'},
             {'title':'New York: monthly', 'dts':'strictbal_cornell_seasonal_mic', 'class_type':'monthly'},
             {'title':'Borneo: hourly', 'dts':'strictbal__specAM-VJR-1audio_moths_sorted_june2019', 'class_type':'hourly'}
             ]
k_folds = 1

# Figure setup
fig_s = 6
n_subplots_x = 2
n_subplots_y = 2
subplt_idx = 1

#fig = plt.figure(figsize=(fig_s*n_subplots_x*1.5,fig_s*n_subplots_y))
fig = plt.figure(figsize=(20,10))

for plot in all_plots:
    # Plot predictions
    fig.add_subplot(n_subplots_y,n_subplots_x,subplt_idx)
    subplt_idx += 1

    for f in feats:
        # Load data from pickle files
        with open(os.path.join('data','{}_{}.pickle'.format(plot['dts'],f)), 'rb') as savef:
            audio_feats_data, labels, datetimes, recorders, unique_ids, classes, mins_per_feat = pickle.load(savef)

        new_labels = get_special_labels(datetimes, recorders, unique_ids, type=plot['class_type'])
        cm, cm_labs, acc, recalls = multi_class_classification(audio_feats_data, new_labels, k_fold=k_folds)

        if 'audioset' in f: c = 'r'
        elif 'sscape' in f: c = 'b'

        plt.plot(recalls,c=c)
        plt.gca().axhline(y=acc,linestyle='--',c=c)
        plt.gca().xaxis.set_ticks(range(len(cm_labs)))
        plt.gca().xaxis.set_ticklabels(cm_labs)

plt.tight_layout()
fig_savefile = os.path.join('figs','multiclass.pdf')
plt.savefig(fig_savefile, format="pdf")

plt.show()
