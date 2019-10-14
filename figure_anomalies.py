import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib
from tqdm import tqdm
from plot_libs import plot_pdist_clusts, plot_cov_ellipse
from analysis_libs import get_clusters, get_embedded_data
from scipy.stats import multivariate_normal
from tqdm import tqdm
from sklearn.decomposition import PCA

n_subplots_x = 2
n_subplots_y = 2

fig = plt.figure(figsize=(20,19))
matplotlib.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# First do the 2D plot of the GMM at one of the sites
fig.add_subplot(n_subplots_y,n_subplots_x,1)

feat = 'raw_audioset_feats_1s'
anom_exp_dir = 'anomaly_playback_exps'
all_playback_files = np.asarray(os.listdir(anom_exp_dir))

site = 'B10'
gmm_file_ext = '_gmm-5.0days-10comps-diag_audio_moths_sorted_june2019-{}.pickle'.format(feat)

num_anoms = 50

with open(os.path.join('anomaly_gmms', '{}{}'.format(site,gmm_file_ext)),'rb') as gmm_savef:
    gmm_model, af_data, labs, dts, uq_ids, clss, mins_per_feat, mins_per_rec_hr = pickle.load(gmm_savef)

in_samp_anoms = -1 * gmm_model.score_samples(af_data)

pca = PCA(n_components=2)
af_data_red = pca.fit_transform(af_data)
#plt.scatter(af_data_red[:,0],af_data_red[:,1], c='k', s=1)

eigenvector = np.array([pca.components_]).T
eigenvector = np.reshape(eigenvector, (eigenvector.shape[:2]))

for cov, mean, weight in zip(gmm_model.covariances_,gmm_model.means_, gmm_model.weights_):
    mean_red = pca.transform(mean.reshape(1, -1))[0]
    mean_raw_tranform = np.dot(eigenvector.T, mean)
    diffs = mean_red - mean_raw_tranform

    cov_raw_tranform = np.dot(np.dot(eigenvector.T, np.diag(cov)),eigenvector)
    cov_red = pca.transform(cov.reshape(1, -1))[0]
    plot_cov_ellipse(cov_raw_tranform,mean_red,nstd=2,alpha=weight*1.5)

means_red = pca.transform(gmm_model.means_)

plt.scatter(means_red[:,0],means_red[:,1], c='blue', s=200*gmm_model.weights_, label='GMM centre')

sort_idx = np.flip(np.argsort(in_samp_anoms))
in_samp_anoms_sorted = in_samp_anoms[sort_idx]
uq_ids_sorted = uq_ids[sort_idx]
af_data_sorted = af_data[sort_idx,:]
dts_sorted = dts[sort_idx]

top_data = af_data[sort_idx[:num_anoms],:]
top_labs = uq_ids[sort_idx[:num_anoms]]

clusts, ord_pdist, ord_pdist_labels = get_clusters(top_data, top_labs)

chosen_idxs = []

for cl in clusts:
    cl_idxs = np.asarray([idx for idx, el in enumerate(uq_ids) if el in cl])
    cl_anoms = in_samp_anoms[cl_idxs]
    ci = cl_idxs[np.argmax(cl_anoms)]
    chosen_idxs.append(ci)

#norm_sounds_idxs = [76629, 295990, 189844, 297798, 411905, 129980, 279533, 237847, 74379, 316080]

print(chosen_idxs)

anoms_red = af_data_red[chosen_idxs,:]
plt.scatter(anoms_red[:,0],anoms_red[:,1],c='r',marker='*',s=100,label='Anomaly')

frame1 = plt.gca()
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.xlabel('PCA: Dim 1')
plt.ylabel('PCA: Dim 2')
plt.title('Acoustic feature space')
plt.axis('equal')

lgnd = plt.legend()
for i in range(len(lgnd.legendHandles)):
    lgnd.legendHandles[i]._sizes = [100]

# Then plot the averaged playback experiment results across all sites

fig.add_subplot(n_subplots_y,n_subplots_x,3)

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
    for site in tqdm(sites):
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
        if 'sinusoid' in unq_cat: continue

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

            #plt.scatter([unq_cat_dist] * len(cd_scores),cd_scores, c='k', alpha=0.2, s=3)
            if len(cd_scores) > 0:
                plt_xs.append(unq_cat_dist)
                plt_ys.append(np.mean(cd_scores))

        plt_xs, plt_ys = zip(*sorted(zip(plt_xs, plt_ys)))
        plt.plot(plt_xs,plt_ys,label=unq_cat.capitalize())

    top_one = []
    top_zone = []
    top_zzone = []
    for site in sites:
        site_anoms = all_in_samp_anoms[site]
        tot_n = len(site_anoms)
        top_zzone.append(site_anoms[int(tot_n / 10000)])
        top_zone.append(site_anoms[int(tot_n / 1000)])
        top_one.append(site_anoms[int(tot_n / 100)])
    #plt.axhline(y=np.mean(top_one), c='gray',ls='--', alpha=0.6, zorder=0)
    #plt.text(line_lab_x,np.mean(top_one),'Top 1%',verticalalignment='baseline')
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
plt.savefig('figs/anomalies.pdf')

plt.show()
