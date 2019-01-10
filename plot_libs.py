import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import preprocessing
from matplotlib.ticker import MaxNLocator
from analysis_libs import reg_df_col_with_features

'''
This module provides functions to assist with plotting our data

Functions:
    get_site_colour
    get_site_pt_size_and_style
    get_feats_nice_name
    get_dimred_nice_name
    smooth_density_curve
    plot_low_dim_space
    plot_pdist_clusts
    plot_regr_dims
'''


# Colours are based on species community clusters
def get_site_colour(site):
    '''
    Get colour for site

    CAUTION: this is hardcoded, based on the clusters obtained from species communities
    that we see in figure_clustering.py. Any changes to the data will likely require
    this to be changed
    '''

    if type(site) is not str:
        site = str(site)

    if 'OP' in site or 'Matrix' in site:
        return 'sienna'
    elif 'VJR 1' in site:
        return 'darkslategray'
    elif 'Riparian' in site or 'B1 602' in site:
        return 'blue'
    else:
        return 'g'


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

    agb_site = agb_df[agb_df['Recorder site'] == site]
    if agb_site.shape[0] == 0:
        pt_sz = 30
        pt_marker = 'x'
    else:
        agb_data = np.asarray(agb_df['Mean AGB'].tolist()).reshape(-1, 1)
        # Scale between sensible sizes for plotting
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(20, 60))
        agb_minmax = min_max_scaler.fit_transform(agb_data)
        idx = agb_df.index.get_loc(agb_site.iloc[0].name)

        pt_sz = agb_minmax[idx]
        pt_marker = 'o'

    return pt_sz, pt_marker


def get_feats_nice_name(feat_str):
    '''
    Get human readable audio feature names
    '''

    if 'soundscape_vec' in feat_str:
        return 'Soundscape compound index'

    if 'mean_processed_audioset_feats' in feat_str:
        return 'AudioSet features (post-processed)'

    if 'mean_raw_audioset_feats' in feat_str:
        return 'AudioSet features'


def get_dimred_nice_name(dimred):
    '''
    Get human readable dimensionality reduction names
    '''


    if 'umap' in dimred:
        return 'UMAP'

    if 'pca' in dimred:
        return 'PCA'

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


def plot_low_dim_space(embedded_data,labels,classes,plt_title,dimred,agb_df,unique_ids=[],label_time=[],mins_per_feat=20):
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
        label_time (str): whether to colour points by time - uses a cyclic colormap
        mins_per_feat (float): minutes of audio per observation in embedded_data
    """

    dims = embedded_data.shape[1]

    if not unique_ids:
        unique_ids = np.asarray([''] * embedded_data.shape[0])

    colour_indexes_ndarray = labels
    print('colour_indexes_ndarray: {}'.format(colour_indexes_ndarray.shape))

    # Use a circular colourmap for when we are colouring points by time
    if label_time == 'hrs':
        normalised = (colour_indexes_ndarray)/24
        pt_colors = plt.cm.hsv(normalised)
    else:
        pt_colors = []
        for l in labels:
            pt_colors.append(get_site_colour(classes[l]))
        pt_colors = np.asarray(pt_colors)

        #normalised = (colour_indexes_ndarray-np.min(colour_indexes_ndarray))/(np.max(colour_indexes_ndarray)-np.min(colour_indexes_ndarray))
        #pt_colors = plt.cm.brg(normalised)

    # Get legend entries which include sampling effort per class
    class_labels = []
    for i, c in enumerate(classes):
        rec_indices = np.where(labels == i)[0]
        class_labels.append('{} (N={} hrs)'.format(classes[i],round(len(rec_indices) * (mins_per_feat/60), 1)))

    dimred_title = get_dimred_nice_name(dimred)

    # Loop through classes
    for i,unique_rec in enumerate(classes):
        rec_indices = np.where(labels == i)[0] # Rows of data belonging to this class

        # If high dimensional, scatter first two dimensions. For 1D data plot a
        # smoothed density distribution
        if dims >= 2: # High dimensional data
            if label_time:
                sz = 30
                mrkr = 'o'
            else:
                sz, mrkr = get_site_pt_size_and_style(classes[i], agb_df)

            x_data = embedded_data[rec_indices,0]
            y_data = embedded_data[rec_indices,1]
            x_mean = np.mean(x_data)
            y_mean = np.mean(y_data)
            plt.scatter(x_data,y_data,color=pt_colors[rec_indices[0]],s=3,alpha=0.2,marker='o',zorder=1,gid=unique_ids[rec_indices])
            plt.scatter(x_mean,y_mean,color=pt_colors[rec_indices[0]],label=class_labels[i],marker=mrkr,s=sz,edgecolors='white',zorder=3)

            plt.text(x_mean,y_mean,'  ' + classes[i],color=pt_colors[rec_indices[0]])
            plt.xlabel('{}: Dim 1'.format(dimred_title))
            plt.ylabel('{}: Dim 2'.format(dimred_title))

            #plt.errorbar(x_mean, y_mean,color=pt_colors[rec_indices[0]],xerr=np.std(x_data),yerr=np.std(y_data),zorder=2)

        elif dims == 1: # 1D data
            embedded_data = embedded_data.flatten()
            x, p = smooth_density_curve(embedded_data[rec_indices,0])
            plt.plot(x,p,color=pt_colors[rec_indices[0]],label=class_labels[i])

    plt.legend(loc='best',ncol=1,prop={'size': 6})
    plt.title(plt_title)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.xticks([])
    plt.yticks([])


def plot_pdist_clusts(clusters, ord_dist_mat, ord_labs, plt_title, cmap=plt.cm.Blues):
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
    plt.imshow(ord_dist_mat, cmap=cmap, interpolation='nearest')
    tick_marks = np.arange(len(ord_labs))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
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
        lab.set_color(get_site_colour(lab.get_text()))


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

    for i, pt in enumerate(aud_reor):
        excl_label = labs_reor[i]
        print('Excluding label: {}'.format(excl_label))
        reg_cv, mean_err_cv, aud_reor_cv, data_reor_cv, labs_reor_cv = reg_df_col_with_features(df, col, audio_mean_data, audio_mean_labels, exclude_labs=excl_label,just_model=True)

        true_val = data_reor[i]

        pt = pt.reshape(1, -1)
        pred = reg_cv.predict(pt)

        plt.scatter(true_val,pred,label=excl_label,c=get_site_colour(excl_label))
        plt.errorbar(true_val, pred, yerr=mean_err_cv, ecolor=get_site_colour(excl_label),elinewidth=0.5,capsize=2)

        if not (pred - mean_err_cv <= true_val <= pred + mean_err_cv):
            plt.text(true_val,pred,'  ' + excl_label,color=get_site_colour(excl_label))


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
