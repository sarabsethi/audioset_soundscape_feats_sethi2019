import os
import json
import imp
import numpy as np
import shutil
import itertools
import sys
import time

def downsample_feats(feat_mat, time_per_feat, target_time_per_feat):
    '''
    Downsample fine resolution audio features to average over longer timescales
    '''
    n_samples = int(target_time_per_feat / time_per_feat)
    if n_samples == 1:
        return feat_mat

    # Truncate observations to a round number of n_samples
    end =  n_samples * int(feat_mat.shape[0]/n_samples)

    if feat_mat.ndim == 2:
        # Higher dimensional features. Shape = (samples, features)
        feat_mat = feat_mat[:end,:]
        ptr = 0
        downsampled_feat = []
        while ptr + n_samples <= end:
            downsampled_feat.append(np.mean(feat_mat[ptr:ptr+n_samples,:],axis=0))
            ptr += n_samples

        downsampled_feat = np.asarray(downsampled_feat)
        return downsampled_feat
    elif feat_mat.ndim == 1:
        # 1 dimensional features
        feat_mat = feat_mat[:end]
        downsampled_feat = np.mean(feat_mat.reshape(-1, n_samples), 1)
        return downsampled_feat
    else:
        raise Exception('Can not downsample features with more than 2 dimensions')

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
