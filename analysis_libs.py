import numpy as np
import umap
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
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
        print('dimred is none, or np.min(data.shape) == 1')
        return [], []

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
        print('Unrecognised dimensionality reduction dimred: {}'.format(dimred))
        return [], []

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

    Inputs:
        X (ndarray): feature data
        y (ndarray): labels associated with feature data
        k_fold (int): number of cross-fold validation runs to use

    Returns:
        (All of the below are averaged from cross-fold validation results)
        cm (ndarray): confusion matrix of results
        cm_labels (ndarray): labels for the confusion matrix
        accuracy (float): average recall across classes
        recalls (ndarray): individual recalls for each class
    '''

    X = np.asarray(X)
    y = np.asarray(y)

    # dividing X, y into train and test data
    sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=0)

    # Do K fold cross validation
    all_accuracies = []
    all_cms = []
    all_recalls = []
    print('Doing {} fold cross validation predictions. Classes: {}'.format(k_fold,np.unique(y)))
    for k, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # training a classifier
        clf = RandomForestClassifier(random_state=0, n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # model accuracy for X_test
        k_accuracy = balanced_accuracy_score(y_test, predictions)

        print('{}/{} folds accuracy: {}'.format(k+1,k_fold,k_accuracy))
        all_recalls.append(recall_score(y_test,predictions,average=None))
        all_accuracies.append(k_accuracy)

        cm_labels = np.unique(y)
        k_cm = confusion_matrix(y_test, predictions, labels=cm_labels)
        all_cms.append(k_cm)

    # Get averages across K fold cross validation
    accuracy = np.mean(all_accuracies)
    print('Average accuracy = {}'.format(accuracy))
    recalls = np.mean(np.asarray(all_recalls),axis=0)
    cm = np.mean(np.asarray(all_cms),axis=0)

    return cm, cm_labels, accuracy, recalls


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
    if rec in ['OP Belian', 'OP3 843', 'C Matrix', 'D Matrix', 'E1 651'] or rec in ['C_Matrix','D_Matrix']:
        return 'Low'
    elif rec in ['E10 654', 'E matrix 647', 'E100 658', 'E1 648'] or rec in ['E1']:
        return 'Low-mid'
    elif rec in ['D100 641', 'E100 edge', 'C10 621', 'D10 639'] or rec in ['E100', 'D100']:
        return 'Mid'
    elif rec in ['D1 634', 'D100 643', 'B10', 'B matrix 599'] or rec in ['B10']:
        return 'Mid-high'
    elif rec in ['VJR 1', 'VJR 2', 'LFE 703', 'LFE 705'] or rec in ['VJR1','VJR2']:
        return 'High'
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

    low_alpha = [26,27,13,19,29,16]
    low_mid_alpha = [20,12,14,25,30]
    mid_alpha = [2,3,7,28,22,1]
    mid_high_alpha = [23,9,21,5,24,10]
    high_alpha = [6,17,4,18,11,8,15]

    if rec_num in low_alpha:
        return 'Low'
    elif rec_num in low_mid_alpha:
        return 'Low-mid'
    elif rec_num in mid_alpha :
        return 'Mid'
    elif rec_num in mid_high_alpha:
        return 'Mid-high'
    elif rec_num in high_alpha:
        return 'High'
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
