import os
import pickle
from analysis_libs import change_lab_type, get_mean_feats_dendrogram, get_embedded_data
from scipy.cluster.hierarchy import dendrogram
import matplotlib
import matplotlib.pyplot as plt
from plot_libs import get_label_nice_name

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure(figsize=(12,6))

dts = 'cornell_sorted_balanced_data+audio_moths_sorted_june2019+cornell_seasonal_mic+PC_recordings+sulawesi_sorted_data+wrege_africa_data+cornell_nz_data_sorted'
f = 'raw_audioset_feats_300s'
label_type = 'dataset'
dimred = 'umap_vis'

with open(os.path.join('multiscale_data','{}_{}.pickle'.format(dts,f)), 'rb') as savef:
    audio_feats_data, labels, datetimes, recorders, unique_ids, classes, mins_per_feat = pickle.load(savef)
    labels, classes = change_lab_type(labels,datetimes,recorders,classes,unique_ids,type=label_type)

data_red, data_red_labels = get_embedded_data(data=audio_feats_data,labels=labels,dimred=dimred,balance_before=True)

linked = get_mean_feats_dendrogram(data_red, labels, classes)

classes = [get_label_nice_name(c,label_type) for c in classes]

dendrogram(linked, orientation='top',labels=classes,distance_sort='descending',color_threshold=100000)

plt.yticks([])
plt.ylabel('Distance between centroids (arbitrary units)')

fig_savef = 'figure_feats_dendrogram'
fig_savefile = os.path.join('figs','{}.svg'.format(fig_savef))
plt.savefig(fig_savefile, format="svg")

plt.show()
