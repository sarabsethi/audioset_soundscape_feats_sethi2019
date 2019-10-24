import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib
from plot_libs import plot_cov_ellipse, plot_2d_anom_schematic
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

n_subplots_x = 3
n_subplots_y = 1

fig = plt.figure(figsize=(28,9))
matplotlib.rcParams.update({'font.size': 24})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# First do the 2D plot of the GMM at one of the sites
fig.add_subplot(n_subplots_y,n_subplots_x,1)
ax = plt.gca()
ax.text(-0.02, 1.05, '(a)', transform=ax.transAxes, fontsize=28, fontweight='bold', va='top', ha='right')

feat = 'raw_audioset_feats_1s'
anom_exp_dir = 'anomaly_playback_exps'
all_playback_files = np.asarray(os.listdir(anom_exp_dir))

site = 'B10'
gmm_file_ext = '_gmm-5.0days-10comps-diag_audio_moths_sorted_june2019-{}.pickle'.format(feat)

num_anoms = 50

with open(os.path.join('anomaly_gmms', '{}{}'.format(site,gmm_file_ext)),'rb') as gmm_savef:
    gmm_model, af_data, labs, dts, uq_ids, clss, mins_per_feat, mins_per_rec_hr = pickle.load(gmm_savef)

plot_2d_anom_schematic(gmm_model, af_data, labs, dts, uq_ids, num_anoms)

plt.show()
exit()

# Make the frame for the middle plot (manually edited in from figure_anomaly_spectros_scores.py in manuscript)
fig.add_subplot(n_subplots_y,n_subplots_x,2)
ax = plt.gca()
ax.text(-0.02, 1.05, '(b)', transform=ax.transAxes, fontsize=28, fontweight='bold', va='top', ha='right')


# Then plot the averaged playback experiment results across all sites
fig.add_subplot(n_subplots_y,n_subplots_x,3)
ax = plt.gca()
ax.text(-0.02, 1.05, '(c)', transform=ax.transAxes, fontsize=28, fontweight='bold', va='top', ha='right')

datasets = [{'sites': ['B10','C_Matrix','B1','D_Matrix','D100','E1','E100','Riparian_2','VJR1','VJR2'],
            'gmm_file_ext': '_gmm-5.0days-10comps-diag_audio_moths_sorted_june2019-{}.pickle'.format(feat)}
            ]

for dts in datasets:
    sites = dts['sites']
    if 'Cornell' in sites: xticks = [15,25,50,100]
    else: xticks = [1,10,25,50,100]

    gmm_file_ext = dts['gmm_file_ext']

    all_gmm_models = dict()
    all_in_samp_anoms = dict()
    for site in sites:
        print('Loading site GMM for {}'.format(site))
        with open(os.path.join('anomaly_gmms', '{}{}'.format(site,gmm_file_ext)),'rb') as gmm_savef:
            gmm_model, af_data, labs, dts, uq_ids, clss, mins_per_feat, mins_per_rec_hr = pickle.load(gmm_savef)
        all_gmm_models[site] = gmm_model

        in_samp_anoms = -1 * all_gmm_models[site].score_samples(af_data)
        in_samp_anoms = sorted(in_samp_anoms,reverse=True)
        all_in_samp_anoms[site] = in_samp_anoms

    print(all_gmm_models.keys())

    categories = [pb_f.split('-')[2].split('.')[0] for pb_f in all_playback_files]
    unq_cats, unq_categories_idxs = np.unique(categories, return_inverse=True)
    line_lab_x = np.inf

    for c_ix, unq_cat in enumerate(unq_cats):
        cat_idxs = np.where((unq_categories_idxs == c_ix))[0]
        cat_playback_files = all_playback_files[cat_idxs]

        cat_dists = [int(pb_f.split('-')[1].split('m')[0]) for pb_f in cat_playback_files]
        unq_cat_dists, unq_cat_dists_idxs = np.unique(cat_dists, return_inverse=True)

        c_scores = []
        plt_xs = []
        plt_ys = []

        for cd_ix, unq_cat_dist in enumerate(unq_cat_dists):
            if unq_cat_dist < line_lab_x: line_lab_x = unq_cat_dist

            cat_dist_idxs = np.where((unq_cat_dists_idxs == cd_ix))[0]
            cat_dist_playback_files = cat_playback_files[cat_dist_idxs]

            cd_scores = []
            for cd_pb_f in cat_dist_playback_files:
                site = cd_pb_f.split('-')[0].replace(' ','_')
                if site not in list(all_gmm_models.keys()): continue

                with open(os.path.join(anom_exp_dir,cd_pb_f), 'rb') as savef:
                    anom_exp_results = pickle.load(savef,encoding='latin1')

                feat_data = anom_exp_results[feat]
                feat_data = feat_data.reshape((feat_data.shape[1],feat_data.shape[2]))
                anom_scores = -1 * all_gmm_models[site].score_samples(feat_data)

                cd_scores.append(np.max(anom_scores))

            if len(cd_scores) > 0:
                plt_xs.append(unq_cat_dist)
                plt_ys.append(np.mean(cd_scores))

        plt_xs, plt_ys = zip(*sorted(zip(plt_xs, plt_ys)))
        plt.plot(plt_xs,plt_ys,label=unq_cat.capitalize())

    top_zone = []
    top_zzone = []
    for site in sites:
        site_anoms = all_in_samp_anoms[site]
        tot_n = len(site_anoms)
        top_zzone.append(site_anoms[int(tot_n / 10000)])
        top_zone.append(site_anoms[int(tot_n / 1000)])
    plt.axhline(y=np.mean(top_zone), c='gray',ls='--', alpha=0.6, zorder=0)
    plt.text(line_lab_x,np.mean(top_zone),'Top 0.1\%',verticalalignment='baseline')
    plt.axhline(y=np.mean(top_zzone), c='gray',ls='--', alpha=0.6, zorder=0)
    plt.text(line_lab_x,np.mean(top_zzone),'Top 0.01\%')

    plt.xlabel('Distance (m)')
    plt.ylabel('Anomaly score')
    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.xticks(xticks)
    plt.gca().set_xticklabels(xticks)

plt.title('Anomaly sensitivity')

plt.tight_layout()
plt.savefig('figs/anomalies.svg')

plt.show()
