import numpy as np
import umap
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
import calendar
import collections
import heapq
from operator import itemgetter
from imblearn.under_sampling import RandomUnderSampler


'''
This module provides functions to assist with analysis of our data
'''

def get_embedded_data(data,labels,dimred,dims=2,return_dimred=False,balance_before=False):
    '''
    Get a low dimensional embedding of audio feature data

    Audio features in this project are represented as high dimensional vectors.
    This function takes a matrix of many audio feature vectors and embeds them
    in a lower dimensional space using UMAP or PCA

    Inputs:
        data (ndarray): matrix of audio feature data - rows are observations
        labels (ndarray): labels of each of the observations - integer values
        dimdred (str): method of dimensionality reduction
                       options: umap, umap_clust, umap_default, umap_vis, umap_vis_landuse, pca

        dims (int): number of dimensions to embed data into
        return_dimred (bool): whether to return the fit dimensionality reduction object
        balance_before (bool): whether to balance classes previous to fitting low dimensional embedding

    Returns:
        X_r (ndarray): embedded data
        labels (ndarray): corresponding labels to observations

        Optional: dim_red_obj: fit dimensionality reduction object
    '''

    dims = int(np.min([dims,data.shape[1]]))

    if np.min(data.shape) == 1 or dimred == 'none':
        raise Exception('dimred is none, or np.min(data.shape) == 1')

    # Pick which dimensionality reduction technique to use
    dim_red_obj = None
    print(dimred)
    if dimred == 'umap':
        # These are parameters for best clustering - lots of epochs takes a while
        dim_red_obj = umap.UMAP(metric='euclidean', min_dist=0, n_neighbors=50, n_epochs=10000, random_state=42, n_components=dims)
    elif dimred == 'umap_clust':
        # Same as above but fewer epochs
        dim_red_obj = umap.UMAP(metric='euclidean', min_dist=0, n_neighbors=50, random_state=42, n_components=dims)
    elif dimred == 'umap_default':
        # Default UMAP parameters
        dim_red_obj = umap.UMAP(metric='euclidean', random_state=42, n_components=dims)
    elif dimred == 'umap_vis':
        # Parameters for better visualisation (tuned for hourly + seasonal cycles)
        dim_red_obj = umap.UMAP(metric='euclidean', n_neighbors=300, random_state=42, n_components=dims)
    elif dimred == 'umap_vis_landuse':
        # Parameters for better visualisation (tuned for land use)
        dim_red_obj = umap.UMAP(metric='euclidean', n_neighbors=600, min_dist=0.8, random_state=42, n_components=dims)

    elif dimred == 'pca':
        # Principal Component Analysis
        dim_red_obj = PCA(n_components=dims)
    else:
        raise Exception('Unrecognised dimensionality reduction dimred: {}'.format(dimred))

    # Normalise features to [0,1] range
    print('Normalising features to [0,1] before dimensionality reduction using {}'.format(dimred))
    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(data)

    print('Computing dimensionality reduction projection from data')
    if balance_before:
        # Balance data used to fit dimensionality reduction technique to ensure
        # the embedding isn't biased towards over-represented classes
        print('Rebalancing classes before dimred')
        print(data_minmax.shape)
        rus = RandomUnderSampler(random_state=42)
        data_minmax_bal, labels_bal = rus.fit_sample(data_minmax, labels)
        print('Balanced shape {}'.format(data_minmax_bal.shape))
        dim_red_obj.fit(data_minmax_bal)
    else:
        dim_red_obj.fit(data_minmax)

    # Embed data in new space
    X_r = dim_red_obj.transform(data_minmax)
    print('X_r: {}'.format(X_r.shape))

    if return_dimred:
        return X_r, labels, dim_red_obj
    else:
        return X_r, labels


def multi_class_classification(X, y, k_fold = 5):
    '''
    Do a multiclass classification task using a random forest classifier
    Accuracy is measured using f1 score

    Inputs:
        X (ndarray): feature data
        y (ndarray): labels associated with feature data
        k_fold (int): number of cross-fold validation runs to use

    Returns:
        (All of the below are averaged from cross-fold validation results)
        cm (ndarray): confusion matrix of results
        cm_labels (ndarray): labels for the confusion matrix
        average_accuracy (float): average accuracy across all classes
        accuracies (ndarray): individual accuracies for each class
    '''

    X = np.asarray(X)
    y = np.asarray(y)

    # dividing X, y into train and test data
    sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=0)

    # Do K fold cross validation
    all_cms = []
    all_accuracies = []
    print('Doing {} fold cross validation predictions. Classes: {}'.format(k_fold,np.unique(y)))
    for k, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # training a classifier
        clf = RandomForestClassifier(random_state=0, n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # model accuracy for X_test
        class_scores = f1_score(y_test,predictions,average=None)
        print('{}/{} folds mean accuracy: {}'.format(k+1,k_fold,np.mean(class_scores)))
        all_accuracies.append(class_scores)

        cm_labels = np.unique(y)
        k_cm = confusion_matrix(y_test, predictions, labels=cm_labels)
        all_cms.append(k_cm)

    # Get averages across K fold cross validation
    accuracies = np.mean(np.asarray(all_accuracies),axis=0)
    average_accuracy = np.mean(accuracies)
    print('Average accuracy = {}'.format(average_accuracy))

    cm = np.mean(np.asarray(all_cms),axis=0)

    return cm, cm_labels, average_accuracy, accuracies


def change_lab_type(labels,datetimes,recorders,classes,unique_ids,type='dataset'):
    '''
    Get new label type for data
    '''

    print(type)
    lab_list = []
    for uq_id_idx, uq_id in enumerate(unique_ids):
        new_l_list = []
        date = datetimes[uq_id_idx]
        if 'date' in type:
            new_l_list.append(date.strftime('%Y-%m-%d'))
        elif 'month' in type:
            new_l_list.append(date.strftime('%b'))
        elif 'year' in type:
            new_l_list.append(date.strftime('%Y'))
        elif 'site' in type:
            new_l_list.append(recorders[uq_id_idx])
        elif 'hour' in type:
            new_l_list.append(date.strftime('%H:00'))
        elif 'unq_id' in type:
            new_l_list.append(uq_id)
        elif 'dataset' in type:
            new_l_list.append(get_dataset_from_unq_id(uq_id))
        elif 'land-use-ny' in type:
            new_l_list.append(get_land_use_ny_type(recorders[uq_id_idx]))
        elif 'land-use' in type:
            new_l_list.append(get_land_use_type(recorders[uq_id_idx]))

        lab_list.append(' '.join(new_l_list))

    new_classes, new_labels = np.unique(lab_list, return_inverse=True)
    return new_labels, new_classes


def get_clusters(data, labels, return_ord_idx=False):
    '''
    Cluster data using AffinityPropagation clustering

    Args:
        data (ndarray): data to cluster
        labels (ndarray): labels corresponding to data

    Returns:
        clusts (list of lists): clusters of labels produced from data
        ordered_pdist (ndarray): reordered pairwise distance matrix using clusters
        ordered_labels (ndarray): reordered labels using clusters
    '''

    # Cluster data using affinity propogation clustering
    data_pdist = squareform(pdist(data, 'euclidean'))
    af = AffinityPropagation().fit(data)
    cluster_centers_indices = af.cluster_centers_indices_
    clust_labels = af.labels_
    n_clusters = len(cluster_centers_indices)

    # Create reordered matrices using clustering output and list of lists. Slightly
    # weird format as that is what variation of information algorithm expects
    clusts = []
    pdist_order = []
    for clust in range(n_clusters):
        clust_members = labels[clust_labels == clust]
        clusts.append(clust_members)

        label_idxs = [i for i, s in enumerate(labels) if s in clust_members]
        pdist_order = pdist_order + label_idxs

    # Reorder pairwise distance matrix and labels
    ordered_pdist = data_pdist[pdist_order,:]
    ordered_pdist = ordered_pdist[:,pdist_order]
    ordered_labels = labels[pdist_order]

    if return_ord_idx:
        return clusts, ordered_pdist, ordered_labels, pdist_order
    else:
        return clusts, ordered_pdist, ordered_labels


def get_mean_feats_dendrogram(data, labels, classes):

    mean_feats = []

    for i,unique_rec in enumerate(classes):
        rec_indices = np.where(labels == i)[0] # Rows of data belonging to this class

        mean_feats.append(np.mean(data[rec_indices,:],axis=0))

    mean_feats = np.asarray(mean_feats)
    print(mean_feats.shape)

    linked = linkage(mean_feats, 'single')
    return linked



def get_dataset_from_unq_id(unq_id):
    datasets = ['audio_moths_sorted_june2019','wrege_africa_data','cornell_nz_data_sorted','PC_recordings','sulawesi_sorted_data','cornell_seasonal_mic','cornell_sorted_balanced_data','cornell_winter_sorted_balanced_data','dena_sabah_sorted_data']

    for dt in datasets:
        if dt in unq_id: return dt

    return 'Unrecognised dataset'


def get_land_use_type(rec):
    '''
    Get land use type from sites in Sabah, Malaysia

    Based on ground truth above ground biomass values
    '''

    if 'AM ' in rec: rec = rec[3:]

    if rec in ['C Matrix', 'D Matrix','E1 651'] or rec in ['C_Matrix','D_Matrix']:
        return r'$\leq 2.45$'
    elif rec in ['E matrix 647','E1 648', 'D100 641'] or rec in ['E1', 'D100']:
        return '2.45 - 2.6'
    elif rec in ['E100 edge', 'C10 621', 'D1 634', 'D100 643'] or rec in ['E100']:
        return '2.6 - 2.75'
    elif rec in ['B10', 'VJR 1', 'VJR 2', 'LFE 703'] or rec in ['VJR1','VJR2','B10']:
        return r'$\geq 2.75$'
    elif 'Riparian' in rec or 'B1 602' in rec or rec in ['Riparian_2','B1']:
        return 'River'
    else:
        return rec

def get_land_use_ny_type(rec):
    '''
    Get land use type from sites in Ithaca, USA

    Based on ground truth avian biodiversity values
    '''

    rec_low = rec.lower()
    rec_num = int(rec_low[1:])

    low_alpha = [24,8,16,14,3,23]
    low_mid_alpha = [4,17,2,6,22]
    mid_alpha = [13,25,27,20,7,12]
    mid_high_alpha = [5,21,11,29,15,1]
    high_alpha = [10,28,9,26,19,30,18]

    if rec_num in low_alpha:
        return r'$\leq 1.4$'
    elif rec_num in low_mid_alpha:
        return '1.4 - 1.7'
    elif rec_num in mid_alpha :
        return '1.7 - 2.0'
    elif rec_num in mid_high_alpha:
        return '2.0 - 2.3'
    elif rec_num in high_alpha:
        return r'$\geq 2.3$'
    else:
        return rec


def least_common_values(array, to_find=None):
    '''
    Find least common values in an array, and associated number of instances
    '''

    counter = collections.Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))


def uniqueify_list(mylist):
    '''
    Turn non-unique list into unique list with numbers appended

    e.g. ['a', 'a', 'b', 'c', 'c', 'c']
     ->  ['a 1', 'a 2', 'b', 'c 1', 'c 2', 'c 3']
    '''

    dups = {}

    for i, val in enumerate(mylist):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]
        else:
            # Special case for first occurrence
            if dups[val][1] == 1:
                mylist[dups[val][0]] += ' {}'.format(str(dups[val][1]))

            # Increment occurrence value, index value doesn't matter anymore
            dups[val][1] += 1

            # Use stored occurrence value
            mylist[i] += ' {}'.format(str(dups[val][1]))

    return mylist
