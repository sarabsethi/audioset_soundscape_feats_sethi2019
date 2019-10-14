import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
from plot_libs import plot_playback_exp
from analysis_libs import get_land_use_type

sites = ['B10','C_Matrix','B1','D_Matrix','D100','E1','E100','Riparian_2','VJR1','VJR2','Cornell']
#sites = ['VJR2','Cornell']

anom_exp_dir = 'anomaly_playback_exps'
all_playback_files = np.asarray(os.listdir(anom_exp_dir))
feat = 'raw_audioset_feats_1s'
gmm_file_ext_safe = '_gmm-5.0days-10comps-diag_audio_moths_sorted_june2019-{}.pickle'.format(feat)
gmm_file_ext_cornell = '_gmm-5.0days-10comps-diag_cornell_playback_training_sorted-{}.pickle'.format(feat)

fig = plt.figure()

n_splots_x = np.min([4,len(sites)])
n_splots_y = np.ceil(len(sites)/4)
splot_idx = 1

for site_ix, site in enumerate(sites):
    site_playback_files = np.asarray([f for f in all_playback_files if f.startswith(site.replace('_',' '))])

    if 'cornell' in site.lower():
        gmm_file_ext = gmm_file_ext_cornell
        title = 'Ithaca, USA'.format(site)
    else:
        gmm_file_ext = gmm_file_ext_safe
        lu = get_land_use_type(site)
        if 'river' not in lu.lower(): lu = lu + ' AGB'
        title = '{} (Sabah, MY)'.format(lu)

    with open(os.path.join('anomaly_gmms', '{}{}'.format(site,gmm_file_ext)),'rb') as gmm_savef:
        gmm_model, af_data, labs, dts, uq_ids, clss, mins_per_feat, mins_per_rec_hr = pickle.load(gmm_savef)

    fig.add_subplot(n_splots_y, n_splots_x, splot_idx)
    splot_idx += 1
    plot_playback_exp(gmm_model, feat, af_data, labs, site, site_playback_files, anom_exp_dir)
    plt.title(title)

plt.legend(bbox_to_anchor=(1.5, 1))
plt.tight_layout()
plt.show()
