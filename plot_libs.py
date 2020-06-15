import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from analysis_libs import get_land_use_type, uniqueify_list, get_clusters, get_embedded_data
from sklearn.decomposition import PCA
import matplotlib
from datetime import datetime
import os
import pickle
import calendar


'''
This module provides functions to assist with plotting our data
'''


def plot_low_dim_space(embedded_data,labels,classes,dimred,label_type,plot_scale=1.0,leg_title=''):
    '''
    Plot a low dimensional embedding of some acoustic feature data

    Inputs:
        embedded_data (ndarray): Embedded data to plot (dimensions are assumed to be ordered)
        labels (ndarray): corresponding integer labels
        classes (ndarray): names of label numbers
        dimred (str): dimensionality reduction technique used
        label_type (str): type of label used (e.g. 'dataset', 'land-use', 'hour', 'month' etc.)
        plot_scale (float): scaling to use for scatter points
    '''

    # Get colours for data points
    if label_type == 'dataset':
        cmap = matplotlib.cm.get_cmap('tab20')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(labels))
        pt_colors = cmap(norm(labels))
        pt_colors = [lighten_color(c,1.2) for c in pt_colors]
    else:
        pt_colors = []
        for lab in labels:
            pt_colors.append(get_label_colour(classes[lab],label_type))

    dimred_title = get_dimred_nice_name(dimred)

    # Loop through classes getting nice human readable names
    nice_classes = [get_label_nice_name(c, label_type) for c in classes]
    nice_classes = uniqueify_list(nice_classes)
    print(nice_classes)

    # Default styles for scatter points
    pt_alpha = 0.2
    pt_sz = 1
    mean_sz = 100
    mrkr = 'o'

    # Plot data corresponding to each class
    for i,unique_rec in enumerate(classes):
        rec_indices = np.where(labels == i)[0] # Rows of data belonging to this class

        x_data = embedded_data[rec_indices,0]
        y_data = embedded_data[rec_indices,1]
        x_mean = np.mean(x_data)
        y_mean = np.mean(y_data)

        # Different datasets have different amount of points, so got to tweak alpha to make them look nice
        if 'month' in label_type:
            pt_alpha = 0.1

        lab_name = get_label_nice_name(classes[i],label_type)
        plt.scatter(x_data,y_data,color=pt_colors[rec_indices[0]],s=pt_sz*plot_scale,alpha=pt_alpha,marker=mrkr,zorder=1,label=lab_name,rasterized=True)
        plt.scatter(x_mean,y_mean,color=pt_colors[rec_indices[0]],alpha=1,marker=mrkr,s=mean_sz*plot_scale,edgecolors='white',zorder=3)

        plt.xlabel('{}: Dim 1'.format(dimred_title))
        plt.ylabel('{}: Dim 2'.format(dimred_title))

    # Order legend in sensible order, and legend styling
    lg_handles, lg_labels = plt.gca().get_legend_handles_labels()
    order = get_label_order(lg_labels,label_type)
    every_n = 1
    if 'hour' in label_type: every_n = 2
    m_loc = 'upper right'
    lgnd = plt.legend([lg_handles[idx] for _i, idx in enumerate(order) if _i % every_n == 0],[lg_labels[idx] for _i, idx in enumerate(order) if _i % every_n == 0],
                      loc=m_loc,ncol=1,columnspacing=0,borderaxespad=0,handletextpad=0,markerfirst=True,title=leg_title,title_fontsize=14)
    frame = lgnd.get_frame()
    frame.set_edgecolor('k')
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [30]
        lgnd.legendHandles[i].set_alpha(1)

    # Sort out axis labels
    ax = plt.gca()

    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    padding = 0.15
    ax.set_xlim(ax.get_xlim()[0],ax.get_xlim()[1] + (ax.get_xlim()[1]-ax.get_xlim()[0])*padding)
    y_offs = (ax.get_ylim()[1]-ax.get_ylim()[0])*padding
    ax.set_ylim(ax.get_ylim()[0] - (y_offs/2),ax.get_ylim()[1] + (y_offs/2))


def plot_multi_class_recalls(recalls, labels, average_accuracy, label_type, feat):
    '''
    Plot recall for each class as a result of a multiclass classification task

    Inputs:
        recalls (ndarray): vector of recalls for each class
        labels (ndarray): label corresponding to each class
        average_accuracy (float): balanced average recall across all classes
        label_type (str): type of label used (e.g. 'dataset', 'land-use', 'hour', 'month' etc.)
        feat (str): acoustic feature set used
    '''

    # Convert decimals to percentages
    recalls = recalls * 100
    average_accuracy = average_accuracy * 100

    # Get sensible order for labels
    order = get_label_order(labels,label_type)
    recalls = np.asarray(recalls)
    recalls = recalls[order]
    labels = labels[order]

    # Plotting recalls and colours
    if 'audioset' in feat:
        c = 'r'
        linestyle = '-'
    elif 'sscape' in feat:
        c = 'b'
        linestyle = '-.'
    else: c = 'k'
    plt.plot(recalls,c=c,linewidth=3,linestyle=linestyle)

    # Dotted line for average recall
    ax = plt.gca()
    ax.axhline(y=average_accuracy,linestyle='--',c=c)
    plt.text(0.1,average_accuracy+1,'{}%'.format(np.round(average_accuracy,1)),color=c)

    # Sort out axes
    ax.xaxis.set_ticks(range(len(labels)))
    ax.xaxis.set_ticklabels(labels)
    if 'hour' in label_type:
        ax.xaxis.set_ticks(ax.get_xticks()[::3])
        ax.xaxis.set_ticklabels(labels[::3])
    elif 'month' in label_type:
        ax.xaxis.set_ticks(ax.get_xticks()[::2])
        ax.xaxis.set_ticklabels(labels[::2])

    ax.set_ylim(0,100)


def plot_2d_anom_schematic(gmm_model, af_data, labs, dts, uq_ids, num_anoms):
    '''
    Plot a 2D schematic representation of a high dimensional GMM

    The top num_anoms sounds are first clustered, then the most anomalous from
    each cluster is plotted with a red star in the resulting schematic

    Inputs:
        gmm_model (GaussianMixture): GaussianMixture model fit to data
        af_data (ndarray): High dimensional data used to fit the GMM
        labs (ndarray): labels associated with the data
        dts (ndarray): datetimes associated with the data
        uq_ids (ndarray): unique ids associated with the data
        num_anoms (int): number of anomalies to identify (prior to clustering)
    '''

    print('Plotting 2D schematic of anomaly detection GMM')

    # Find anomaly scores for all data points
    in_samp_anoms = -1 * gmm_model.score_samples(af_data)

    # Find 2D embedding of data points
    pca = PCA(n_components=2)
    af_data_red = pca.fit_transform(af_data)

    # Get 2D embedding of covariances of each of the GMM components
    eigenvector = np.array([pca.components_]).T
    eigenvector = np.reshape(eigenvector, (eigenvector.shape[:2]))
    for cov, mean, weight in zip(gmm_model.covariances_,gmm_model.means_, gmm_model.weights_):
        mean_red = pca.transform(mean.reshape(1, -1))[0]
        mean_raw_tranform = np.dot(eigenvector.T, mean)
        # There's a consistent diff here so some offset is off in the maths but it doesn't matter for schematic depiction purposes
        diffs = mean_red - mean_raw_tranform

        cov_raw_tranform = np.dot(np.dot(eigenvector.T, np.diag(cov)),eigenvector)
        cov_red = pca.transform(cov.reshape(1, -1))[0]
        plot_cov_ellipse(cov_raw_tranform,mean_red,nstd=2,alpha=weight*1.5)

    means_red = pca.transform(gmm_model.means_)
    plt.scatter(means_red[:,0],means_red[:,1], c='blue', s=200*gmm_model.weights_, label='GMM centre')

    # Find top num_anoms anomalies
    sort_idx = np.flip(np.argsort(in_samp_anoms))
    top_data = af_data[sort_idx[:num_anoms],:]
    top_labs = uq_ids[sort_idx[:num_anoms]]

    # Cluster those anomalies and pick representatives from each cluster
    clusts, ord_pdist, ord_pdist_labels = get_clusters(top_data, top_labs)
    chosen_idxs = []
    for cl in clusts:
        cl_idxs = np.asarray([idx for idx, el in enumerate(uq_ids) if el in cl])
        cl_anoms = in_samp_anoms[cl_idxs]
        ci = cl_idxs[np.argmax(cl_anoms)]
        chosen_idxs.append(ci)

    # Plot the chosen anomalous points
    anoms_red = af_data_red[chosen_idxs,:]
    plt.scatter(anoms_red[:,0],anoms_red[:,1],c='r',marker='*',s=100,label='Anomaly')

    # Sort out axes and legend
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


def plot_playback_exp(gmm_model, feat, af_data, labs, site_playback_files, anom_exp_dir):
    '''
    Plot results from a single anomaly playback experiment

    Inputs:
        gmm_model (GaussianMixture): GaussianMixture model fit to data for site of interest
        feat (str): acoustic features used
        af_data (ndarray): audio feature data used to fit the GMM (5 full days of audio)
        labs (ndarray): labels corresponding to data
        site_playback_files (ndarray): filenames storing audio features from playback experiments
        anom_exp_dir (str): directory in which the site_playback_files are stored
    '''

    # Calculate anomaly scores for data used to train GMM
    training_scores = -1 * gmm_model.score_samples(af_data)
    training_scores_sorted = sorted(training_scores, reverse=True)

    # Loop through individual categories of anomalous sound (chainsaws, gunshots etc.)
    categories = [pb_f.split('-')[2].split('.')[0] for pb_f in site_playback_files]
    unq_cats, unq_categories_idxs = np.unique(categories, return_inverse=True)
    for ix, unq_cat in enumerate(unq_cats):
        cat_idxs = np.where((unq_categories_idxs == ix))[0]
        xs = []
        ys = []

        # Loop through different distances of playbacks (1m, 25m, 50m etc.)
        for pb_f in site_playback_files[cat_idxs]:
            dist = int(pb_f.split('-')[1].split('m')[0])
            anom_label = pb_f.split('-')[2].split('.')[0]

            with open(os.path.join(anom_exp_dir,pb_f), 'rb') as savef:
                anom_exp_results = pickle.load(savef,encoding='latin1')

            feat_data = anom_exp_results[feat]
            feat_data = feat_data.reshape((feat_data.shape[1],feat_data.shape[2]))

            # Calculate anomaly score for each feature corresponding to audio from
            # the playback experiment (typically on 0.96s resolution)
            anom_scores = -1 * gmm_model.score_samples(feat_data)
            xs.append(dist)
            ys.append(np.max(anom_scores))

        xs, ys = zip(*sorted(zip(xs, ys)))
        plt.plot(xs,ys,label='{}'.format(unq_cat))

    # Draw lines at top 0.01%, 0.1% of all anomaly scores used in training data
    zone_perc_idx = int(len(training_scores_sorted) / 1000)
    zzone_perc_idx = int(len(training_scores_sorted) / 10000)
    plt.axhline(y=training_scores_sorted[zone_perc_idx], c='gray',ls='--', alpha=0.75, zorder=0)
    plt.axhline(y=training_scores_sorted[zzone_perc_idx], c='gray',ls='--', alpha=0.75, zorder=0)

    # Sort out axes
    plt.xlabel('Distance (m)')
    plt.ylabel('Anomaly score')
    plt.yscale('symlog')
    plt.xscale('log')

    plt.xticks([],[])
    plt.xticks([1,20,100],['1','20','100'])


def get_dimred_nice_name(raw_name):
    '''
    Get human readable dimensionality reduction names
    '''

    lookup_dict = {'umap': 'UMAP (slow clustering)',
                            'umap_clust': 'UMAP',
                            'umap_default': 'UMAP',
                            'umap_vis': 'UMAP',
                            'umap_vis_landuse': 'UMAP',
                            'pca': 'PCA'
                            }

    if raw_name in lookup_dict: return lookup_dict[raw_name]

    return raw_name

def get_label_nice_name(label, label_type):
    '''
    Get human readable label names
    '''

    if label_type == 'dataset':
        nice_names = {'audio_moths_sorted_june2019': 'Sabah, MY\n(Audiomoth)',
                      'PC_recordings': 'Sabah, MY\n(Tascam)',
                      'PC_recordings_nowater': 'Sabah, MY (Tascam) (no water sites)',
                      'dena_sabah_sorted_data': 'Sabah, MY various sites',
                      'cornell_sorted_balanced_data': 'Ithaca, USA\n(Swift)',
                      'cornell_winter_sorted_balanced_data': 'Ithaca, USA\nWinter',
                      'cornell_seasonal_mic': 'Ithaca, USA\n(Custom mic)',
                      'cornell_nz_data_sorted': 'Abel Tasman,\nNZ (Swift)',
                      'sulawesi_sorted_data': 'Sulawesi, ID\n(Swift)',
                      'wrege_africa_data': 'Nouabalé-\nNdoki, COG\n(Swift)'
                      }
        if label in nice_names: return nice_names[label]

    if label_type == 'site':
        if label.startswith('AM '): label = label.split('AM ')[1]

        land_use_type = get_land_use_type(label)
        return land_use_type

    if label_type == 'hour':
        return label

    return label


def get_label_colour(label, label_type='site'):
    '''
    Get unique colour for labels
    '''

    if type(label) is not str:
        label = str(label)
    label = label.lower()

    if 'month' in label_type:
        month = datetime.strptime(label.split(' ')[0],'%b').strftime('%m')
        month_norm = (int(month)-1)/11
        return lighten_color(plt.cm.hsv(month_norm),1.2)

    if 'hour' in label_type:
        hour = label.split(':')[0]
        hour_norm = (int(hour))/23
        return lighten_color(plt.cm.hsv(hour_norm),1.2)

    if 'land-use' in label_type:
        if label == '$\\leq 2.45$': return 'sienna'
        elif label == '2.45 - 2.6': return 'darkgoldenrod'
        elif label == '2.6 - 2.75': return 'g'
        elif label == '$\\geq 2.75$': return 'k'
        elif label == 'river' or label == 'water': return 'blue'

    return 'k'


def get_label_order(labels, lab_type):
    '''
    Get sensible order for labels
    '''

    reord = []
    if 'month' in lab_type:
        reord = [list(calendar.month_abbr).index(c) for c in labels]

    elif lab_type == 'land-use-ny':
        reord = np.ones(len(labels))*-1
        for ix, lb in enumerate(labels):
            if lb ==  '$\\leq 1.4$': reord[ix] = 0
            if lb == '1.4 - 1.7': reord[ix] = 1
            if lb == '1.7 - 2.0': reord[ix] = 2
            if lb == '2.0 - 2.3': reord[ix] = 3
            if lb ==  '$\\geq 2.3$': reord[ix] = 4
        for ix, r in enumerate(reord):
            if r == -1: reord[ix] = np.max(reord) + 1

    elif lab_type == 'land-use' :
        reord = np.ones(len(labels))*-1
        for ix, lb in enumerate(labels):
            if lb ==  '$\\leq 2.45$': reord[ix] = 0
            if lb == '2.45 - 2.6': reord[ix] = 1
            if lb == '2.6 - 2.75': reord[ix] = 2
            if lb ==  '$\\geq 2.75$': reord[ix] = 3
        for ix, r in enumerate(reord):
            if r == -1: reord[ix] = np.max(reord) + 1

    elif 'dataset' in lab_type:
        reord = np.ones(len(labels))*-1
        for ix, lb in enumerate(labels):
            if 'borneo' in lb.lower() or 'congo' in lb.lower() or 'sulawesi' in lb.lower():
                reord[ix] = np.max(reord) + 1

        for ix, r in enumerate(reord):
            if r == -1: reord[ix] = np.max(reord) + 1

    if len(reord) >= 1:
        reord = [int(i) for i in reord]
        return np.argsort(reord)
    else: return range(len(labels))


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    '''
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    '''

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def lighten_color(color, amount=0.5):
    '''
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    '''

    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
