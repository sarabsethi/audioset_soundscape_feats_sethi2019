import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import preprocessing
from matplotlib.ticker import MaxNLocator
from analysis_libs import reg_df_col_with_features, get_land_use_type, uniqueify_list, get_label_order
from sklearn.decomposition import PCA
import matplotlib
from matplotlib.patches import Ellipse
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import os
import pickle
import pandas as pd
from scipy import stats
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import calendar

'''
This module provides functions to assist with plotting our data

Dictionaries:
    get_feats_nice_name

Functions:
    get_label_colour
    get_site_pt_size_and_style
    get_dimred_nice_name
    smooth_density_curve
    plot_low_dim_space
    plot_pdist_clusts
    plot_regr_dims
'''

# Plots an image at each x and y location.
def plot_image(xData, yData, im, scale=0.5):
    for x, y in zip(xData, yData):
        bb = Bbox.from_bounds(x,y,scale,scale)
        bb2 = TransformedBbox(bb,plt.gca().transData)
        bbox_image = BboxImage(bb2,
                            norm = None,
                            origin=None,
                            clip_on=False)

        bbox_image.set_data(im)
        plt.gca().add_artist(bbox_image)


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


# Get human readable audio feature names
get_feats_nice_name = {'soundscape_vec': 'Soundscape compound index',
                       'mean_processed_audioset_feats': 'AudioSet features (post-processed)',
                       'mean_raw_audioset_feats': 'AudioSet features',
                       'comp_sscape_idx_60s': 'Soundscape compound index (60s)',
                       'comp_sscape_idx_300s': 'Soundscape compound index (300s)',
                       'comp_sscape_idx_299s': 'Soundscape compound index (299s)',
                       'raw_audioset_feats_60s': 'AudioSet features (60s)',
                       'raw_audioset_feats_300s': 'AudioSet features (300s)'
                       }

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
        nice_names = {'audio_moths_sorted_june2019': 'Borneo (Audiomoth)',
                      'PC_recordings': 'Borneo (Tascam)',
                      'PC_recordings_nowater': 'Borneo (Tascam) (no water sites)',
                      'dena_sabah_sorted_data': 'Borneo, various sites',
                      'cornell_sorted_balanced_data': 'New York (summer)',
                      'cornell_winter_sorted_balanced_data': 'New York (winter)',
                      'cornell_seasonal_mic': 'New York',
                      'cornell_nz_data_sorted': 'New Zealand',
                      'sulawesi_sorted_data': 'Sulawesi',
                      'wrege_africa_data': 'Congo'
                      }

        if label in nice_names: return nice_names[label]

    if label_type == 'site':
        if label.startswith('AM '): label = label.split('AM ')[1]

        land_use_type = get_land_use_type(label)
        return land_use_type

    if label_type == 'hour':
        label_int = int(label)
        return '{}:00'.format(str(label_int).zfill(2))

    return label


# Colours are based on species community clusters
def get_label_colour(label, label_type='site'):
    '''
    Get colour for label

    CAUTION: this is hardcoded, based on the clusters obtained from species communities
    that we see in figure_clustering.py. Any changes to the data will likely require
    this to be changed
    '''
    if type(label) is not str:
        label = str(label)
    label = label.lower()

    if 'month' in label_type:
        month = datetime.strptime(label.split(' ')[0],'%b').strftime('%m')
        month_norm = (int(month)-1)/11
        return lighten_color(plt.cm.hsv(month_norm),1.2)

    if 'hour' in label_type:
        hour = label
        hour_norm = (int(hour))/23
        return lighten_color(plt.cm.hsv(hour_norm),1.2)

    if label_type == 'site':
        if label.startswith('s') and len(label.split(' '))==1: return 'k'

        land_use_type = get_land_use_type(label).lower()
        if 'low agb' in land_use_type:
            return 'sienna'
        elif 'high agb' in land_use_type:
            #return 'darkslategray'
            return 'k'
        elif 'water' in land_use_type:
            return 'blue'
        else:
            return 'g'

    if 'land-use' in label_type:
        if label == 'low': return 'sienna'
        elif label == 'low-mid': return 'darkgoldenrod'
        elif label == 'mid': return 'g'
        elif label == 'mid-high': return 'darkslategray'
        elif label == 'high': return 'k'
        elif label == 'river' or label == 'water': return 'blue'

    return 'k'


def get_site_pt_size_and_style(site, agb_df):
    """
    Get styling for scatter of a site point - size scaled by AGB at given site

    Args:
        site (str): site name
        agb_df (pandas.DataFrame): agb dataframe

    Returns:
        pt_sz (float): size of scatter point
        pt_marker (str): type of marker to use

    Raises:
        Exception: description

    """
    if site.startswith('AM '):
        site = site.split('AM ')[1]

    if agb_df is None:
        pt_sz = 30
        pt_marker = 'x'
        return pt_sz, pt_marker

    agb_site = agb_df[agb_df['Recorder site'] == site]
    if agb_site.shape[0] == 0:
        pt_sz = 30
        pt_marker = 'x'
    else:
        agb_data = np.asarray(agb_df['Mean AGB'].tolist()).reshape(-1, 1)
        # Scale between sensible sizes for plotting
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(20, 90))
        agb_minmax = min_max_scaler.fit_transform(agb_data)
        idx = agb_df.index.get_loc(agb_site.iloc[0].name)

        pt_sz = int(agb_minmax[idx])
        pt_marker = 'o'

    return pt_sz, pt_marker


def smooth_density_curve(data, interp_pts=1000):
    """
    Create a smoothed density curve from a vector of observations of 1D features

    Args:
        data (ndarray): vector of data
        interp_pts (int): number of points to interpolate between

    Returns:
        x (ndarray): x axis of smoothed distribution
        smoothed_curve (ndarray): y axis of smoothed distribution
    """

    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(),data.max(), interp_pts)
    smoothed_curve = kde(x)

    return x, smoothed_curve

def plot_low_dim_species(field_df, agb_df=None):

    # Get field data and labels from dataframe object
    spec_communities = np.array(field_df['Bag of words'].values.tolist())
    spec_communities_sites = np.asarray(field_df['Recorder site'].tolist())

    dimred_title = 'PCA'
    pca_model = PCA(n_components=2)
    spec_communities_pca = pca_model.fit_transform(spec_communities)

    for site in list(set(spec_communities_sites)):
        site_idxs = np.where((spec_communities_sites == site))
        pt_sz, pt_marker = get_site_pt_size_and_style(site,agb_df)
        plt.scatter(spec_communities_pca[site_idxs,0],spec_communities_pca[site_idxs,1],label=site,c=get_label_colour(site),s=pt_sz,marker=pt_marker)
        plt.text(spec_communities_pca[site_idxs,0],spec_communities_pca[site_idxs,1],site)

    plt.title('Species communities')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('{}: Dim 1'.format(dimred_title))
    plt.ylabel('{}: Dim 2'.format(dimred_title))


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


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

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


def plot_low_dim_space(embedded_data,labels,classes,plt_title,dimred,agb_df=None,unique_ids=[],label_type=[],mins_per_feat=20,plot_scale=1.0,show_ellipses=False,use_legend=False):
    """
    Plot low-dimensional representation of embedded data

    Once high dimensional data has been embedded into some sensible embedding where
    dimensions are ordered, here we plot the first 2 dimensions. If data only has
    1 dimension a smoothed density distribution is plotted

    Args:
        embedded_data (ndarray): embedded feature data - rows are observations
        labels (ndarray): labels of embedded_data rows
        classes (ndarray): names of classes corresponding to labels
        plt_title (str): title of plot
        dimred (str): Dimensionality reduction technique used (for axis titles)
        agb_df (pandas.DataFrame): agb dataframe
        unique_ids (ndarray): unique IDs of each point for hover text
        label_type (str): label type of data
        mins_per_feat (float): minutes of audio per observation in embedded_data
    """

    dims = embedded_data.shape[1]

    if not unique_ids:
        unique_ids = np.asarray([''] * embedded_data.shape[0])

    colour_indexes_ndarray = labels
    print('colour_indexes_ndarray: {}'.format(colour_indexes_ndarray.shape))

    # Use a circular colourmap for when we are colouring points by time
    if label_type and label_type.strip() != '' and label_type != 'dataset':
        pt_colors = []
        for lab in labels:
            pt_colors.append(get_label_colour(classes[lab],label_type))
    else:
        cmap = matplotlib.cm.get_cmap('tab20')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(labels))
        pt_colors = cmap(norm(labels))
        pt_colors = [lighten_color(c,1.2) for c in pt_colors]

        #normalised = (colour_indexes_ndarray-np.min(colour_indexes_ndarray))/(np.max(colour_indexes_ndarray)-np.min(colour_indexes_ndarray))
        #pt_colors = plt.cm.brg(normalised)

    # Get legend entries which include sampling effort per class
    class_labels = []
    for i, c in enumerate(classes):
        #rec_indices = np.where(labels == i)[0]
        #class_labels.append('{} (N={} hrs)'.format(classes[i],round(len(rec_indices) * (mins_per_feat/60), 1)))
        class_labels.append('{}'.format(classes[i]))

    dimred_title = get_dimred_nice_name(dimred)

    # Loop through classes
    nice_classes = [get_label_nice_name(c, label_type) for c in classes]
    nice_classes = uniqueify_list(nice_classes)
    print(nice_classes)


    for i,unique_rec in enumerate(classes):
        rec_indices = np.where(labels == i)[0] # Rows of data belonging to this class
        if len(rec_indices) == 0:
            print('No indices found for {}'.format(unique_rec))
            continue
        # If high dimensional, scatter first two dimensions. For 1D data plot a
        # smoothed density distribution
        if dims >= 2: # High dimensional data
            if label_type:
                sz = 100
                c = nice_classes[i].lower()
                if False and (label_type == 'dataset' and ('borneo' in c or 'congo' in c or 'sulawesi' in c)):
                    mrkr = '^'
                elif label_type == 'land-use' and ('river' in c or 'water' in c):
                    mrkr = '^'
                else:
                    mrkr = 'o'
            else:
                sz, mrkr = get_site_pt_size_and_style(classes[i], agb_df)

            x_data = embedded_data[rec_indices,0]
            y_data = embedded_data[rec_indices,1]
            x_mean = np.mean(x_data)
            y_mean = np.mean(y_data)

            pt_alpha = 1
            if not use_legend:
                plt.text(x_mean+0.1,y_mean,nice_classes[i],color=pt_colors[rec_indices[0]])
                pt_alpha = 0.06
                if len(embedded_data) > 10000: pt_alpha = 0.02

            pt_sz = 1

            lab_name = get_label_nice_name(class_labels[i],label_type)
            if 'land-use' in label_type and 'river' not in lab_name.lower() and 'water' not in lab_name.lower(): lab_name = '{} AGB'.format(lab_name)

            show_means = False
            if 'month' in label_type or 'hour' in label_type or 'dataset' in label_type or not use_legend:
                pt_alpha = 0.3
                show_means = True
            if 'month' in label_type:
                pt_alpha = 0.1
            if 'dataset' in label_type:
                pt_alpha = 0.5
                pt_sz = 5

            plt.scatter(x_data,y_data,color=pt_colors[rec_indices[0]],s=pt_sz*plot_scale,alpha=pt_alpha,marker=mrkr,zorder=1,gid=unique_ids[rec_indices],label=lab_name)
            if show_means:
                plt.scatter(x_mean,y_mean,color=pt_colors[rec_indices[0]],alpha=1,marker=mrkr,s=sz*plot_scale,edgecolors='white',zorder=3)

            if len(rec_indices) > 1 and show_ellipses:
                points = embedded_data[rec_indices,:]
                plot_point_cov(points, nstd=1.4, alpha=0.2, color=pt_colors[rec_indices[0]])

            plt.xlabel('{}: Dim 1'.format(dimred_title))
            plt.ylabel('{}: Dim 2'.format(dimred_title))

            #plt.errorbar(x_mean, y_mean,color=pt_colors[rec_indices[0]],xerr=np.std(x_data),yerr=np.std(y_data),zorder=2)

        elif dims == 1: # 1D data
            embedded_data = embedded_data.flatten()
            x, p = smooth_density_curve(embedded_data[rec_indices,0])
            plt.plot(x,p,color=pt_colors[rec_indices[0]],label=class_labels[i])

    lg_handles, lg_labels = plt.gca().get_legend_handles_labels()
    order = get_label_order([l.split(' ')[0] for l in lg_labels],label_type)

    every_n = 1
    if 'month' in label_type:
        every_n = 2
    elif 'hour' in label_type:
        every_n = 3

    m_loc = 'upper left'

    lgnd = plt.legend([lg_handles[idx] for _i, idx in enumerate(order) if _i % every_n == 0],[lg_labels[idx] for _i, idx in enumerate(order) if _i % every_n == 0], loc=m_loc)
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [30]
        lgnd.legendHandles[i].set_alpha(1)

    plt.title(plt_title)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.xticks([])
    plt.yticks([])

def plot_safe_field_agb_comparison(field_df, agb_df, text_annot=True):
    print(field_df)
    merged_df = pd.merge(field_df, agb_df, left_on=['Recorder site'], right_on = ['Recorder site'])
    alpha_col = 'Alpha diversity (tot)'
    agb_col = 'Mean AGB'
    beta_col = 'Beta diversity (tot)'
    merged_df.plot(x=agb_col, y=[alpha_col,beta_col], style='o', ax=plt.gca())
    rho_alpha, pval_alpha = stats.pearsonr(merged_df[agb_col].tolist(),merged_df[alpha_col].tolist())
    legend_alpha = 'Alpha: rho = {}, p = {}'.format(np.round(rho_alpha,2),np.round(pval_alpha,4))

    rho_beta, pval_beta = stats.pearsonr(merged_df[agb_col].tolist(),merged_df[beta_col].tolist())
    legend_beta = 'Beta: rho = {}, p = {}'.format(np.round(rho_beta,2),np.round(pval_beta,4))

    plt.legend([legend_alpha,legend_beta])
    if text_annot:
        for idx, row in merged_df.iterrows():
            plt.text(row[agb_col], row[beta_col], row['Recorder site'])
            plt.text(row[agb_col], row[alpha_col], row['Recorder site'])

    plt.xlabel('Mean AGB')
    plt.ylabel('Species diversity')

def plot_pdist_clusts(clusters, ord_dist_mat, ord_labs, plt_title='', cmap=plt.cm.Blues, label_type=[]):
    """
    Plot a clustered pairwise distance matrix with boxes bounding the clusters

    Args:
        clusters (list of lists): clusters of data labels
        ord_dist_mat (ndarray): ordered pairwise distance matrix
        ord_labs (ndarray): ordered labels for ord_dist_mat
        plt_title (str): title of plot
        cmap (plt.cm): colormap
    """

    # Show the pairwise distance matrix as a heatmap
    plt.imshow(ord_dist_mat, cmap=cmap, interpolation='nearest',aspect='auto')
    tick_marks = np.arange(len(ord_labs))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ord_labs = [get_label_nice_name(l,label_type) for l in ord_labs]
    plt.yticks(tick_marks, ord_labs)
    plt.title(plt_title)

    # Draw rectangles around clusters
    clust_offst = -0.5
    for clust in clusters:
        clust_sz = len(clust)

        rect = patches.Rectangle((clust_offst,clust_offst),clust_sz,clust_sz,linewidth=5,edgecolor='r',facecolor='none')
        plt.gca().add_patch(rect)
        clust_offst = clust_offst + clust_sz

    # Colour labels by species community clusters
    for lab in plt.gca().get_yticklabels():
        lab.set_color(get_label_colour(lab.get_text(),label_type))


def plot_regr_dims(scores, pvals, dims_x, dimred, legend_txt='', colour='red'):
    """
    Plot how regression scores vary when different number of dimensions of the
    embedded feature data is used

    Args:
        scores (list): list of regression scores for each dimension
        pvals (list): list of p values for each regression
        dims_x (list): number of dimensions used for each regression task
        dimred (str): dimensionality reduction technique used
        legend_txt (str): text to use in the legend
        colour (str): colour of line in plot
    """

    # Plot regression scores and p values
    plt.plot(dims_x,scores,c=colour,label='{} regression score'.format(legend_txt))
    plt.plot(dims_x,pvals,c=colour,ls='--',label='{} regression $p$'.format(legend_txt))

    plt.xlabel('Number of {} dimensions'.format(get_dimred_nice_name(dimred)))
    plt.axhline(y=0.05,c='gray',ls='--')
    plt.text(1.5,0.06,'$p = 0.05$',color='gray')

    ax = plt.gca()
    ax.set_xlim([1, np.max(dims_x)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0.,top=1.05)

    #plt.legend(loc='upper left',ncol=1,prop={'size': 8})
    plt.legend(loc='upper left',ncol=1)


def plot_field_data_preds(audio_mean_data, audio_mean_labels, df, col):
    reg, mean_err, aud_reor, data_reor, labs_reor = reg_df_col_with_features(df, col, audio_mean_data, audio_mean_labels, just_model=True)

    true_vals = []
    pred_vals = []
    for i, pt in enumerate(aud_reor):
        excl_label = labs_reor[i]
        print('Excluding label: {}'.format(excl_label))
        reg_cv, mean_err_cv, aud_reor_cv, data_reor_cv, labs_reor_cv = reg_df_col_with_features(df, col, audio_mean_data, audio_mean_labels, exclude_labs=excl_label,just_model=True)

        true_val = data_reor[i]

        pt = pt.reshape(1, -1)
        pred = reg_cv.predict(pt)

        plt.scatter(true_val,pred,label=excl_label,c=get_label_colour(excl_label))
        plt.errorbar(true_val, pred, yerr=mean_err_cv, ecolor=get_label_colour(excl_label),elinewidth=0.5,capsize=2)

        #if not (pred - mean_err_cv <= true_val <= pred + mean_err_cv):
        #    plt.text(true_val,pred,'  ' + excl_label,color=get_label_colour(excl_label))

        true_vals.append(true_val)
        pred_vals.append(pred)

    mean_abs_err = mean_absolute_error(true_vals, pred_vals)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot a dashed line through y = x (predicted values = true values)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, c='gray',ls='--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel('True: {}'.format(col))
    plt.ylabel('Predicted: {}'.format(col))

    return mean_abs_err

def plot_confusion_matrix(cm, cm_labs, lab_type):
    ax = plt.gca()

    cm_labs = [get_label_nice_name(c, lab_type) for c in cm_labs]
    reord = get_label_order(cm_labs, lab_type)

    cm_labs = np.asarray(cm_labs)
    cm_labs = cm_labs[reord]

    cm = cm[reord,:]
    cm = cm[:,reord]

    cm_labs = np.asarray(cm_labs)
    cm_labs_y = np.flip(cm_labs)
    cm = np.flip(cm,0)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=[], yticklabels=cm_labs_y,
           ylabel='True label',
           xlabel='Predicted label')

    return im
