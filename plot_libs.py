import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from analysis_libs import get_land_use_type, uniqueify_list, get_label_order
from sklearn.decomposition import PCA
import matplotlib
from datetime import datetime
import os
import pickle


'''
This module provides functions to assist with plotting our data
'''


def plot_low_dim_space(embedded_data,labels,classes,dimred,label_type,plot_scale=1.0):
    dims = embedded_data.shape[1]

    colour_indexes_ndarray = labels
    print('colour_indexes_ndarray: {}'.format(colour_indexes_ndarray.shape))

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

    # Loop through classes

    nice_classes = [get_label_nice_name(c, label_type) for c in classes]
    nice_classes = uniqueify_list(nice_classes)
    print(nice_classes)

    pt_alpha = 1
    pt_sz = 1
    mean_sz = 100
    mrkr = 'o'

    for i,unique_rec in enumerate(classes):
        rec_indices = np.where(labels == i)[0] # Rows of data belonging to this class

        x_data = embedded_data[rec_indices,0]
        y_data = embedded_data[rec_indices,1]
        x_mean = np.mean(x_data)
        y_mean = np.mean(y_data)

        # Different datasets have different amount of points, so got to tweak alpha to make them look nice
        pt_alpha = 0.2
        if 'month' in label_type:
            pt_alpha = 0.1
        #if 'dataset' in label_type:
            #pt_alpha = 0.5
            #pt_sz = 5

        lab_name = get_label_nice_name(classes[i],label_type)
        plt.scatter(x_data,y_data,color=pt_colors[rec_indices[0]],s=pt_sz*plot_scale,alpha=pt_alpha,marker=mrkr,zorder=1,label=lab_name)
        plt.scatter(x_mean,y_mean,color=pt_colors[rec_indices[0]],alpha=1,marker=mrkr,s=mean_sz*plot_scale,edgecolors='white',zorder=3)

        plt.xlabel('{}: Dim 1'.format(dimred_title))
        plt.ylabel('{}: Dim 2'.format(dimred_title))

    # Order legend and only show 1 of every_n labels
    lg_handles, lg_labels = plt.gca().get_legend_handles_labels()
    order = get_label_order(lg_labels,label_type)

    every_n = 1
    ncol = 1
    if 'month' in label_type:
        ncol = 2
    elif 'hour' in label_type:
        every_n = 3

    m_loc = 'upper right'
    lgnd = plt.legend([lg_handles[idx] for _i, idx in enumerate(order) if _i % every_n == 0],[lg_labels[idx] for _i, idx in enumerate(order) if _i % every_n == 0],
                      loc=m_loc,ncol=ncol,columnspacing=0,borderaxespad=0,handletextpad=0)
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [30]
        lgnd.legendHandles[i].set_alpha(1)

    # Sort out axis labels
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.xticks([])
    plt.yticks([])

def plot_multi_class_recalls(recalls, labels, average_accuracy, label_type, feat):
    if 'audioset' in feat: c = 'r'
    elif 'sscape' in feat: c = 'b'
    else: c = 'k'

    # Convert decimals to percentages
    recalls = recalls * 100
    average_accuracy = average_accuracy * 100

    order = get_label_order(labels,label_type)
    recalls = np.asarray(recalls)
    recalls = recalls[order]
    labels = labels[order]

    plt.plot(recalls,c=c,linewidth=3)
    ax = plt.gca()
    ax.axhline(y=average_accuracy,linestyle='--',c=c)
    plt.text(0.1,average_accuracy+1,'{}%'.format(np.round(average_accuracy,1)),color=c)
    ax.xaxis.set_ticks(range(len(labels)))
    ax.xaxis.set_ticklabels(labels)

    if 'hour' in label_type:
        ax.xaxis.set_ticks(ax.get_xticks()[::3])
        ax.xaxis.set_ticklabels(labels[::3])
    elif 'month' in label_type:
        ax.xaxis.set_ticks(ax.get_xticks()[::2])
        ax.xaxis.set_ticklabels(labels[::2])

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# Get human readable dimensionality reduction names
def get_dimred_nice_name(raw_name):
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

    if label_type == 'dataset':
        nice_names = {'audio_moths_sorted_june2019': 'Sabah (Audiomoth)',
                      'PC_recordings': 'Sabah, MY (Tascam)',
                      'PC_recordings_nowater': 'Sabah, MY (Tascam) (no water sites)',
                      'dena_sabah_sorted_data': 'Sabah, MY various sites',
                      'cornell_sorted_balanced_data': 'Ithaca, USA (summer)',
                      'cornell_winter_sorted_balanced_data': 'Ithaca, USA (winter)',
                      'cornell_seasonal_mic': 'Ithaca, USA (all seasons)',
                      'cornell_nz_data_sorted': 'South Island, NZ',
                      'sulawesi_sorted_data': 'Sulawesi, ID',
                      'wrege_africa_data': 'Republic of the Congo'
                      }

        if label in nice_names: return nice_names[label]

    if label_type == 'site':
        if label.startswith('AM '): label = label.split('AM ')[1]

        land_use_type = get_land_use_type(label)
        return land_use_type

    if label_type == 'hour':
        return label

    return label


# Colours are based on species community clusters
def get_label_colour(label, label_type='site'):
    '''
    Get colour for label
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
        if label == 'low': return 'sienna'
        elif label == 'low-mid': return 'darkgoldenrod'
        elif label == 'mid': return 'g'
        elif label == 'mid-high': return 'darkslategray'
        elif label == 'high': return 'k'
        elif label == 'river' or label == 'water': return 'blue'

    return 'k'


def gmm_plot_ellipses(gmm, ellipse_scales = [1], alpha=0.5, color = 'k'):
    for e_sc in ellipse_scales:
        for n, mean in enumerate(gmm.means_):
            if gmm.covariance_type == 'full':
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == 'tied':
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == 'diag':
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == 'spherical':
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

            covariances = covariances * e_sc

            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = Ellipse(gmm.means_[n, :2], v[0], v[1],
                                      180 + angle,
                                      color=color)
            ell.set_clip_box(plt.gca().bbox)
            ell.set_alpha(alpha)
            plt.gca().add_artist(ell)
            plt.gca().set_aspect('equal', 'datalim')


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
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
    """
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

def plot_playback_exp(gmm_model, feat, af_data, labs, site, site_playback_files, anom_exp_dir):
    training_scores = -1 * gmm_model.score_samples(af_data)
    training_scores_sorted = sorted(training_scores, reverse=True)

    categories = [pb_f.split('-')[2].split('.')[0] for pb_f in site_playback_files]
    unq_cats, unq_categories_idxs = np.unique(categories, return_inverse=True)

    for ix, unq_cat in enumerate(unq_cats):
        cat_idxs = np.where((unq_categories_idxs == ix))[0]

        xs = []
        ys = []
        print('site {}, sounds {}'.format(site, unq_cat))
        for pb_f in site_playback_files[cat_idxs]:
            dist = int(pb_f.split('-')[1].split('m')[0])
            anom_label = pb_f.split('-')[2].split('.')[0]
            #print('site {}, dist {}, sounds {}'.format(site, dist, anom_label))

            with open(os.path.join(anom_exp_dir,pb_f), 'rb') as savef:
                anom_exp_results = pickle.load(savef,encoding='latin1')

            feat_data = anom_exp_results[feat]
            feat_data = feat_data.reshape((feat_data.shape[1],feat_data.shape[2]))
            #print(feat_data.shape)

            anom_scores = -1 * gmm_model.score_samples(feat_data)
            xs.append(dist)
            ys.append(np.max(anom_scores))

        xs, ys = zip(*sorted(zip(xs, ys)))
        plt.plot(xs,ys,label='{}'.format(unq_cat))

    # Draw lines at top 0.01%, 0.1%, 1%
    one_perc_idx = int(len(training_scores_sorted) / 100)
    zone_perc_idx = int(len(training_scores_sorted) / 1000)
    zzone_perc_idx = int(len(training_scores_sorted) / 10000)

    #plt.axhline(y=training_scores_sorted[one_perc_idx], c='gray',ls='--', alpha=0.75, zorder=0)
    plt.axhline(y=training_scores_sorted[zone_perc_idx], c='gray',ls='--', alpha=0.75, zorder=0)
    plt.axhline(y=training_scores_sorted[zzone_perc_idx], c='gray',ls='--', alpha=0.75, zorder=0)

    plt.xlabel('Distance (m)')
    plt.ylabel('Anomaly score')
    plt.yscale('symlog')
    plt.xscale('log')

    plt.xticks([],[])
    plt.xticks([1,20,100],['1','20','100'])
