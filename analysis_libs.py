import numpy as np
import umap
import pickle
from sklearn import preprocessing, linear_model
from sklearn.cluster import AffinityPropagation
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy import stats
from math import log
from statistics import mean
from fastcluster import linkage
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, recall_score
from tqdm import tqdm
import calendar
import collections
import heapq
from operator import itemgetter
from imblearn.under_sampling import RandomUnderSampler

'''
This module provides functions to assist with analysis of our data
'''

def least_common_values(array, to_find=None):
    counter = collections.Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))

def uniqueify_list(mylist):
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

def get_label_order(labels, lab_type):

    reord = []
    if 'month' in lab_type:
        print(labels)
        reord = [list(calendar.month_abbr).index(c) for c in labels]

    elif 'land-use' in lab_type:
        labels = [l.split(' ')[0] for l in labels]
        reord = np.ones(len(labels))*-1
        for ix, lb in enumerate(labels):
            if lb.lower() == 'low': reord[ix] = 0
            if lb.lower() == 'low-mid': reord[ix] = 1
            if lb.lower() == 'mid': reord[ix] = 2
            if lb.lower() == 'mid-high': reord[ix] = 3
            if lb.lower() == 'high': reord[ix] = 4
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

def change_lab_type(labels,datetimes,recorders,classes,unique_ids,type='dataset'):
    print(type)
    lab_list = []
    for uq_id_idx, uq_id in enumerate(unique_ids):
        new_l_list = []
        date = datetimes[uq_id_idx]
        #print(uq_id.split('_'))
        if 'date' in type:
            new_l_list.append(date.strftime('%Y-%m-%d'))
        elif 'month' in type:
            new_l_list.append(date.strftime('%b'))
        elif 'year' in type:
            new_l_list.append(date.strftime('%Y'))
        elif 'site' in type:
            new_l_list.append(recorders[uq_id_idx])
        elif 'hour' in type:
            new_l_list.append(date.strftime('%H'))
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

def get_embedded_data(data,labels,dimred,dims=2,return_dimred=False,balance_before=False):
    """
    Get a low dimensional embedding of audio feature data

    Audio features in this project are represented as high dimensional vectors.
    This function takes a matrix of many audio feature vectors and embeds them
    in a lower dimensional space using UMAP or PCA

    Args:
        data (ndarray): matrix of audio feature data - rows are observations
        labels (ndarray): labels of each of the observations - integer values
        dimdred (str): method of dimensionality reduction
                       options: umap_clust, umap, umap_default, pca
        dims (int): number of dimensions to embed data into

    Returns:
        X_r (ndarray): embedded data
        labels (ndarray): corresponding labels to observations
    """

    dims = int(np.min([dims,data.shape[1]]))

    if np.min(data.shape) == 1 or dimred == 'none':
        print('dimred is none, or np.min(data.shape) == 1: not doing any embedding')
        return data, labels

    else:
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
            print('Unrecognised dimensionality reduction dimred: {}. Returning raw data'.format(dimred))
            return data, labels

        print('Normalising features to [0,1] before dimensionality reduction using {}'.format(dimred))
        min_max_scaler = preprocessing.MinMaxScaler()
        data_minmax = min_max_scaler.fit_transform(data)

        print('Computing dimensionality reduction projection from data')

        if balance_before:
            print('Rebalancing classes before dimred')
            print(data_minmax.shape)
            rus = RandomUnderSampler(random_state=42)
            data_minmax_bal, labels_bal = rus.fit_sample(data_minmax, labels)
            print('Balanced shape {}'.format(data_minmax_bal.shape))
            dim_red_obj.fit(data_minmax_bal)
        else:
            dim_red_obj.fit(data_minmax)

        X_r = dim_red_obj.transform(data_minmax)
        print('X_r: {}'.format(X_r.shape))

        if return_dimred:
            return X_r, labels, dim_red_obj
        else:
            return X_r, labels

def get_audio_mean_data(data,labels,classes):
    """
    Get mean feature vectors for each class

    Args:
        data (ndarray): matrix of audio feature data - rows are observations
        labels (ndarray): labels of each of the observations - integer values
        classes (ndarray): name of each of the classes (maps labels to strings)

    Returns:
        mean_data (ndarray): mean feature for each class
        classes (ndarray): corresponding class names
    """

    mean_data = []
    for i,unique_class in enumerate(classes):
        rec_indices = np.where(labels == i)[0]
        rec_data = data[rec_indices,:]
        mean_data.append(np.mean(rec_data, axis=0))

    mean_data = np.asarray(mean_data)

    return mean_data, classes


def calc_var_info(audio_data, audio_labels, field_df):
    """
    Cluster sites by audio feature data compare to clusters arising using species
    community data using variation of information (VI)

    Sites are clustered using get_clusters and VI is calculated for null permutations
    of the data to assign a p-value for how similar the clusters arising from the
    audio data are to the clusters obtained using species community data

    Args:
        audio_data (ndarray): one mean audio feature vector per site
        audio_labels (ndarray): corresponding labels to the feature data
        field_df (pandas.DataFrame): species community data

    Returns:
        audio_clusts (list): list encoding how audio feature data clusters sites
        audio_pdist (ndarray): pairwise distance matrix of species communities
        audio_pdist_labels (ndarray): labels for audio_pdist rows / cols
        field_clusts (list): list encoding how species community data clusters sites
        field_pdist (ndarray): pairwise distance matrix of species communities
        field_pdist_labels (ndarray): labels for field_pdist rows / cols
        p_val (float): probability that the two clustering methods are uncorrelated
    """

    # Get field data and labels from dataframe object
    field_data = np.array(field_df['Bag of words'].values.tolist())
    field_labels = np.asarray(field_df['Recorder site'].tolist())

    # Do clustering of audio data
    audio_clusts, audio_pdist, audio_pdist_labels = get_clusters(audio_data, audio_labels)
    print('Doing audio clustering')
    #print(audio_clusts)

    # Do clustering of field data
    field_clusts, field_pdist, field_pdist_labels = get_clusters(field_data, field_labels)
    print('Doing field data clustering')
    #print(field_clusts)

    # Calculate variation of information between two clustering outputs
    real_vi = variation_of_information(audio_clusts,field_clusts)

    # Do null shufflings to get null distribution of VI
    n_nulls = 1000
    print('Calculating null VI distribution using {} iterations'.format(n_nulls))
    null_vis = []
    np.random.seed(42)
    audio_data_perm = audio_data.copy()
    field_data_perm = field_data.copy()
    for n in range(n_nulls):
        np.random.shuffle(audio_data_perm)
        np.random.shuffle(field_data_perm)
        null_audio_clusts,_,_ = get_clusters(audio_data_perm, audio_labels)
        null_field_clusts,_,_ = get_clusters(field_data_perm, field_labels)
        null_vi = variation_of_information(null_audio_clusts,null_field_clusts)
        null_vis.append(null_vi)

    # Calculate p-value of real VI using null distribution
    smaller_chance_vis = [i for i in null_vis if i <= real_vi]
    p_val = len(smaller_chance_vis) / len(null_vis)

    print('Real VI: {}, P value: {}'.format(real_vi,p_val))
    print('Max possible VI: {}'.format(np.log2(len(audio_labels))))

    return audio_clusts, audio_pdist, audio_pdist_labels, field_clusts, field_pdist, field_pdist_labels, p_val


def data_variance_explained(data):
    """
    Cumulative variance explained of data by each added dimension in a PCA embedding

    Args:
        data (ndarray): matrix of data - rows are observations

    Returns:
        var_expl: vector of cumulative variance explained by each added dimension
    """

    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(data)

    comps = np.min(data_minmax.shape)
    pca_model = PCA(n_components=comps)
    pca_model.fit_transform(data_minmax)

    return pca_model.explained_variance_ratio_.cumsum()

def get_clusters(data, labels, return_ord_idx=False):
    """
    Cluster data using AffinityPropagation clustering

    Args:
        data (ndarray): data to cluster
        labels (ndarray): labels corresponding to data

    Returns:
        clusts (list of lists): clusters of labels produced from data
        ordered_pdist (ndarray): reordered pairwise distance matrix using clusters
        ordered_labels (ndarray): reordered labels using clusters
    """

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
    '''

    rec = rec.lower()
    if 'op' in rec or 'matrix' in rec:
        return 'Low AGB'
    elif 'vjr' in rec or 'lfe' in rec:
        return 'High AGB'
    elif 'riparian' in rec or 'b1 602' in rec:
        return 'Water'
    else:
        return 'Mid AGB'
    '''

def get_land_use_ny_type(rec):
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

def get_special_labels(datetimes, recorders, unq_labels, type='hourly'):
    if type == 'hourly':
        labels = ['{}:00'.format(str(d.hour).zfill(2)) for d in datetimes]
    elif type == 'monthly':
        labels = [calendar.month_abbr[d.month] for d in datetimes]
    elif type == 'site':
        labels = recorders
    elif type == 'land-use':
        labels = [get_land_use_type(rec) for rec in recorders]
    elif type == 'land-use-ny':
        labels = [get_land_use_ny_type(rec) for rec in recorders]
    elif type == 'dataset':
        labels = [get_dataset_from_unq_id(unq_id) for unq_id in unq_labels]
    else:
        raise Exception('Don\'t understand type: {}'.format(type))

    return labels

def multi_class_classification(X, y, k_fold = 5):
    X = np.asarray(X)
    y = np.asarray(y)

    # dividing X, y into train and test data
    sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=0)
    #sss.get_n_splits(X, y)

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

        # print('Training classifier: X_train {}, X_test {}, y_train {}, y_test {}'.format(len(X_train), len(X_test), len(y_train), len(y_test)))
        predictions = clf.predict(X_test)

        # model accuracy for X_test
        k_accuracy = balanced_accuracy_score(y_test, predictions)

        print('{}/{} folds accuracy: {}'.format(k+1,k_fold,k_accuracy))
        all_recalls.append(recall_score(y_test,predictions,average=None))
        all_accuracies.append(k_accuracy)

        cm_labels = np.unique(y)
        k_cm = confusion_matrix(y_test, predictions, labels=cm_labels)
        all_cms.append(k_cm)

    accuracy = np.mean(all_accuracies)
    print('Average accuracy = {}'.format(accuracy))
    recalls = np.mean(np.asarray(all_recalls),axis=0)
    cm = np.mean(np.asarray(all_cms),axis=0)

    return cm, cm_labels, accuracy, recalls


def reg_df_col_with_features(df, col, audio_mean_data, audio_classes, exclude_labs='', just_model=False):
    """
    Calculate a regression between some field data and the mean audio feature vector
    per site using leave-one-out cross validation

    Fiddly work here is matching the sites, but then just a simple regression between
    field data and audio data. Regression model defined by get_reg_model().
    Leave-one-out cross validation is used and a p-value is returned by permuting
    the data and looking at the distribution of regression scores

    Args:
        df (pandas.DataFrame): field data dataframe
        col (str): column from the field data to regress against
        audio_mean_data (ndarray): mean audio feature vector for each class
        audio_classes (list): list of strings for classes of audio_mean_data
        exclude_labs (comma separated str): ignore these sites when fitting the regression model
        just_model (bool): if true no p-value is calculated

    Returns:
        mean_score (float): mean regression score
        combined_p (float): fishers combined p value from cross-val
        full_reg_model: fit regression model
        full_mean_err: mean error from full_reg_model fit

        reordered_audio_data (ndarray): reordered audio_mean_data
        reordered_df_data (ndarray): reordered column of data from input df
        reordered_df_labels (ndarray): reordered labels from input df
    """

    df_data = np.asarray(df[col].tolist())
    df_labels = np.asarray(df['Recorder site'].tolist())

    reorder_audio = []
    reorder_field = []
    # Match labels between audio and field data
    for fidx, fl in enumerate(df_labels):
        for aidx, al in enumerate(audio_classes):
            if al.startswith('AM '): al = al.split('AM ')[1]
            if fl == al and fl not in exclude_labs:
                reorder_audio.append(aidx)
                reorder_field.append(fidx)

    reordered_audio_classes = audio_classes[reorder_audio]
    reordered_audio_data = audio_mean_data[reorder_audio,:]
    reordered_df_labels = df_labels[reorder_field]
    reordered_df_data = df_data[reorder_field]
    # Make sure they really are matched!

    for df_idx, df_lab in enumerate(reordered_df_labels.tolist()):
        audio_lab = reordered_audio_classes.tolist()[df_idx]
        if audio_lab.startswith('AM '): audio_lab = audio_lab.split('AM ')[1]
        assert(df_lab == audio_lab)
    #print(reordered_df_labels.tolist())
    #print(reordered_audio_classes.tolist())
    #assert reordered_df_labels.tolist() == reordered_audio_classes.tolist()

    # Fit regression model to all data
    full_reg_model = get_reg_model().fit(reordered_audio_data, reordered_df_data)
    full_preds = full_reg_model.predict(reordered_audio_data)
    full_mean_sq_err = mean_squared_error(reordered_df_data, full_preds, multioutput='raw_values')
    full_mean_err = np.sqrt(full_mean_sq_err)

    if just_model:
        return full_reg_model, full_mean_err, reordered_audio_data, reordered_df_data, reordered_df_labels

    # Perform leave-one-out cross validation
    p_vals_list = []
    mean_errs_list = []
    reg_scores_list = []
    kf = KFold(n_splits=reordered_audio_data.shape[0])
    for train, test in kf.split(reordered_audio_data):
        train_audio_data = reordered_audio_data[train,:]
        test_audio_data = reordered_audio_data[test,:]
        train_df_data = reordered_df_data[train]
        test_df_data = reordered_df_data[test]

        # Fit to training data
        reg = get_reg_model().fit(train_audio_data, train_df_data)

        # Round to 5 decimal places because of weird rounding errors near R^2=1
        reg_score = round(reg.score(train_audio_data, train_df_data),5)

        # Find mean error of the test dataset (just one point for LOO CV)
        preds = reg.predict(test_audio_data)
        mean_sq_err = mean_squared_error(test_df_data, preds, multioutput='raw_values')
        mean_err = np.sqrt(mean_sq_err)

        if reg_score == 1:
            p_val = 1
        else:
            # Calculate p-values using permutations of training data and calculating
            # null distribution of regression scores
            n_nulls = 100
            null_scores = []
            ran_seed = np.sum(df_data)
            np.random.seed(2)
            reordered_audio_data_perm = train_audio_data.copy()
            for n in range(n_nulls):
                np.random.shuffle(reordered_audio_data_perm)
                rand_reg = get_reg_model().fit(reordered_audio_data_perm, train_df_data)
                null_scores.append(rand_reg.score(reordered_audio_data_perm, train_df_data))

            larger_chance_scores = [i for i in null_scores if i >= reg_score]
            p_val = len(larger_chance_scores) / len(null_scores)

        p_vals_list.append(p_val)
        mean_errs_list.append(mean_err)
        reg_scores_list.append(reg_score)

    # Combine p-values using Fisher's method
    fisher_stat,combined_p = stats.combine_pvalues(p_vals_list)
    if combined_p > 1: combined_p = 1

    tqdm.write('Leave one out CV: {} regression score: {}, p = {}'.format(col,round(mean(reg_scores_list),3),round(combined_p,3)))

    mean_score = mean(reg_scores_list)
    return mean_score, combined_p, full_reg_model, full_mean_err, reordered_audio_data, reordered_df_data, reordered_df_labels


def variation_of_information(X, Y):
    """
    SOURCE: https://gist.github.com/jwcarr/626cbc80e0006b526688

    Calculate variation of information between two clusterings of the same data
    points as described in below article

    Meila, M. (2007). Comparing clusterings-an information
    based distance. Journal of Multivariate Analysis, 98,
    873-895. doi:10.1016/j.jmva.2006.11.013

    Args:
        X (list of lists): one set of clusters
        Y (list of lists): second set of clusters

    Returns:
        variation of information between two clusterings of same data points
    """
    for ix, x in enumerate(X):
        for _ix, _x in enumerate(x):
            if _x.startswith('AM '): X[ix][_ix] = _x.split('AM ')[1]
    for iy, y in enumerate(Y):
        for _iy, _y in enumerate(y):
            if _y.startswith('AM '): Y[iy][_iy] = _y.split('AM ')[1]

    n = float(sum([len(x) for x in X]))
    sigma = 0.0
    for x in X:
        p = len(x) / n
        for y in Y:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (log(r / p, 2) + log(r / q, 2))
    return abs(sigma)

def get_reg_model():
    '''
    Simply allows us to not hard code which regression model we're using

    Returns:
        scipy regression model
    '''
    return RandomForestRegressor(random_state=42)
    #return linear_model.LinearRegression()
    #return linear_model.RANSACRegressor(random_state=12)

def calc_added_dims_regs(audio_mean_data, audio_classes, df, col, max_n_steps=9999):
    """
    Calculate regressions between mean audio feature data and field data for each
    added dimension of embedded feature data used

    Args:
        audio_mean_data (ndarray): mean audio feature vector for each class
        audio_classes (list): list of strings for classes of audio_mean_data
        df (pandas.DataFrame): field data dataframe
        col (str): column from the field data to regress against
        max_n_steps (int): maximum number of dimensions to step through
                           (helpful for very high dimensional data)

    Returns:
        r_scores (list): list of regression scores per added dimension
        pvals (list): list of p-values per added dimension
        dims_x (list): which dimensions were tested
    """

    n_feats = audio_mean_data.shape[1]

    r_scores = []
    pvals = []
    dims_x = []

    max_dims = np.min([n_feats+1,audio_mean_data.shape[0]-1])
    step = np.max([int((max_dims) / max_n_steps),1])
    flat_count = 0

    # Loop through the dimensions we want to test regressions at
    steps =  range(1,max_dims,step)
    for _dim in tqdm(steps):
        tqdm.write('Regression for {} dimensions'.format(_dim))
        audio_mean_data_reduced = audio_mean_data[:,:_dim]

        r_score, pval,_,_,_,_,_ = reg_df_col_with_features(df, col, audio_mean_data_reduced, audio_classes)

        r_scores.append(r_score)
        pvals.append(pval)
        dims_x.append(_dim)

        if pval == 1:
            flat_count += 1
            if flat_count == 2:
                break

    return r_scores, pvals, dims_x


def compute_serial_matrix(dist_mat,method="ward"):
    '''
        SOURCE: https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html

        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]

    return seriated_dist, res_order, res_linkage


def seriation(Z,N,cur_index):
    '''
        SOURCE: https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html

        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
