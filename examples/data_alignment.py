#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:02:17 2023

@author: dwzhou
"""

import numpy as np
import scipy as sp
import scipy.io as spio
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
from alphacsc import utils

#%%

def count_troughs(all_trough_data):
    n_troughs = 0
    for s in range(all_trough_data.size):
        troughs = all_trough_data[0, s]
        n_troughs += int(troughs.size / 4)
    return n_troughs

#% Assemble trough-aligned data

def assemble_trough_aligned_data(data, trough_data, data_len=200, detrend=True, normalize=True):
    n_troughs = count_troughs(trough_data)
    half_width = int(data_len / 2)
    
    X_all = np.empty([n_troughs, data_len], dtype='float64')
    YorN_all = np.empty([n_troughs, 1], dtype='float64')
    t_all = np.empty(n_troughs, dtype='bool')
    
    trough_i = 0
    for s in range(trough_data.size):
        for r in range(trough_data[0, s].shape[0]):
            tr = int(trough_data[0, s][r, 0]) - 1
            i = int(trough_data[0, s][r, 1])
            YorN = int(trough_data[0, s][r, 3])
            t = trough_data[0, s][r, 2]
            
            if i - half_width < 0 or i + half_width > 600:
                continue
            
            data_segment = data[s, tr, i - half_width:i + half_width]
            
            if detrend:
                data_segment = sp.signal.detrend(data_segment)
            if normalize:
                data_segment = data_segment / np.std(data_segment)

            X_all[trough_i, :] = data_segment
            YorN_all[trough_i, 0] = YorN
            t_all[trough_i] = t
            
            trough_i += 1
            
    X_all = np.delete(X_all, range(trough_i, n_troughs), axis=0)
    YorN_all = np.delete(YorN_all, range(trough_i, n_troughs), axis=0)
    t_all = np.delete(t_all, range(trough_i, n_troughs), axis=0)
    
    return X_all, YorN_all, t_all

def preprocess_data(data, offset, detrend, normalize=False):
    preprocessed_data = np.empty_like(data)

    for s in range(data.shape[0]):
        for r in range(data.shape[1]):
            data_segment = data[s, r, :]
            if offset:
                data_segment = data_segment - np.mean(data_segment)
            if detrend:
                data_segment = sp.signal.detrend(data_segment)
            if normalize:
                data_segment = data_segment / np.std(data_segment)

            preprocessed_data[s, r, :] = data_segment

    return preprocessed_data

def plot_individual_traces(X_all, YorN_all, data_len=200):
    for r in range(X_all.shape[0]):
        if YorN_all[r] == 1:
            plt.plot(range(data_len), X_all[r, :], alpha=0.1, color='r')
        else:
            plt.plot(range(data_len), X_all[r, :], alpha=0.1)
            
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()
    # fig.savefig('event_coverage_N.png', dpi=100)

# Example usage:
# data = spio.loadmat('data/full_trials_allsubjs_troughdata.mat')
# all_data = data['all_data']
# all_trough_data = data['all_trough_data']
# X_all, YorN_all, t_all = assemble_trough_aligned_data(all_data, all_trough_data, data_len=200)
# X_all = utils.check_univariate_signal(X_all)
# plot_individual_traces(X_all, YorN_all, 200)
