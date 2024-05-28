#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: david_zhou@brown.edu

To-do list:
    - Parameters for "modes": peak/rise/decay/trough, burst detection, 
    - Find optimal range of n_times_atom based on frequency band and mode
    - Group cycles based on temporal distribution of extrema
    - Limit atom timing in comparison (see above)
    - Put plot save options in all functions or take out
    - Compare with multiple atoms (plot_atom_comparison)
    - Add jitter to data alignment to randomize
    - Add clipping to data alignment
"""

# import os
# from os.path import dirname, join as pjoin
import itertools

import numpy as np
# import scipy as sp
# import scipy.io as spio
# import scipy.signal
# from scipy import stats
# import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd
# import statistics
# import math
import seaborn
colorblind10_palette = seaborn.color_palette("colorblind", n_colors=10)
from pprint import pprint

# from alphacsc import utils
# from alphacsc import BatchCDL

#%% Methods

def plot_sparse_code_and_reconstruction(cdl, X_all, trial_number):
    fig, axs = plt.subplots(2, 1, figsize=(9,6), sharex=True)
    
    sparse_code_trial = np.squeeze(cdl._z_hat[trial_number,:,:])
    
    # Plotting the sparse code
    im = axs[0].imshow(sparse_code_trial, aspect='auto', origin='lower', cmap='Greys', extent=[0, sparse_code_trial.shape[1], 0, sparse_code_trial.shape[0]])
    axs[0].set_title(f"Sparse code for trial {trial_number}")
    axs[0].set_ylabel("Atom #")
    axs[0].set_yticks(range(sparse_code_trial.shape[0]))

    # Add colorbar
    # fig.colorbar(im, ax=axs[0], orientation='vertical')
    
    # Original signal
    original_signal = X_all[trial_number, 0, :]

    # Reconstructed signal
    reconstructed_signals = cdl.transform_inverse(cdl._z_hat)
    reconstructed_signal = np.squeeze(reconstructed_signals[trial_number, :])

    # Plotting the original and reconstructed signals
    axs[1].plot(original_signal, label='Original')
    axs[1].plot(reconstructed_signal, label='Reconstructed', linestyle='dashed')
    axs[1].set_title(f"Original and reconstructed signals for trial {trial_number}")
    axs[1].set_xlabel("Time samples")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    plt.tight_layout()
    plt.show()



def plot_atoms(cdl, sfreq, labels_on=False):
    """
    Plots the temporal patterns of atoms for a given Convolutional Dictionary Learning (CDL) model.

    This function generates a plot of the temporal patterns of the atoms from a fitted CDL model. Each atom's 
    temporal pattern is displayed as a subplot within the figure. The x-axis represents time in seconds, and 
    the y-axis represents the amplitude of the atom's temporal pattern. The figure is displayed using `plt.show()`.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning model containing the atoms (`v_hat_`) and activations (`u_hat_`).
    sfreq : float
        The sampling frequency of the data used to fit the CDL model, in Hz.
    labels_on : bool, optional
        If True, classify and display labels for the atoms based on their cycle components. Default is False.

    Returns
    -------
    None

    Notes
    -----
    This function uses the `classify_cycle_components` function to label atoms if `labels_on` is True. The labels
    are determined based on a variance threshold.
    
    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> plot_atoms(cdl_model, sfreq=256)
    """
    
    v_hat = cdl.v_hat_
    u_hat = cdl.u_hat_
    
    if labels_on == True:
        labels = classify_cycle_components(cdl, variance_threshold=0.001)
    
    n_atoms, n_times_atom = v_hat.shape
    n_columns = min(6, n_atoms)
    n_rows = int(np.ceil(n_atoms / n_columns))
    figsize = (4 * n_columns, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, sharey=True)
    axes = axes.ravel()

    # Plot the temporal pattern of the atom
    for kk in range(n_atoms):
        ax = axes[kk]
        time = np.arange(n_times_atom) / sfreq
        ax.plot(time, v_hat[kk] * u_hat[kk], color='C%d' % kk)
        ax.set_xlim(0, n_times_atom / sfreq)
        # ax.set_ylim(min(v_hat[kk]), max(v_hat[kk]))
        ax.set(xlabel='Time (sec)')
        ax.grid(True)
        if labels_on == False:
            ax.title.set_text('atom #' + str(kk))
        else:
            ax.title.set_text('atom #' + str(kk) + ' : ' + labels[kk])

    fig.tight_layout()
    plt.show()
    


def plot_event_coverage(cdl):
    """
    Plots the event coverage of atoms for a given Convolutional Dictionary Learning (CDL) model.

    This function generates a plot displaying the event coverage of each atom from a fitted CDL model. 
    Each atom's event coverage is displayed as a subplot within the figure. The x-axis represents time bins, 
    and the y-axis represents the number of activations of the atom within each time bin. The figure is 
    displayed using `plt.show()`.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning model containing the atoms (`v_hat_`) and activations (`_z_hat`).

    Returns
    -------
    None

    Notes
    -----
    This function assumes a fixed data length of 200 time bins for plotting the activation bins. It iterates 
    over trials to count the activations within each time bin and plots these counts.

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> plot_event_coverage(cdl_model)
    """  
    n_atoms, n_times_atom = cdl.v_hat_.shape
    
    sparse_code = cdl._z_hat # n_trials x n_atoms x bins_i
    len_data = 200
    
    fig, axs = plt.subplots(nrows=int(np.ceil(cdl.n_atoms/6)), ncols=min(6,cdl.n_atoms))
    axs = axs.ravel()
    
    for i in range(cdl.n_atoms):
        
        which_atom = i # because they're sorted
        activation_bins = np.zeros(len_data, dtype=int)
        
        # loop over trials
        for r in range(sparse_code.shape[0]):
            
            for b in range(sparse_code.shape[2]):
                
                if sparse_code[r,which_atom,b] > 0:
                    activation_bins[b:b+n_times_atom] += 1
        
        axs[i].step(range(len_data),activation_bins)
        axs[i].title.set_text('atom #' + str(which_atom))
    
    fig = plt.gcf()
    fig.set_size_inches(18.5, cdl.n_atoms/2)
    # fig.savefig('output/event_coverage' + str(cdl.n_atoms) + '_len' + str(cdl.v_hat_.shape[1]) + '.png', dpi=100)
    plt.show()



def plot_atom_scatter(cdl, class_labels, sfreq,
                      choose_atoms=None, color_cycle=None, scale_factor=1):
    """
    Plots a scatter plot of atom activations across trials for a given Convolutional Dictionary Learning (CDL) model.

    This function generates a scatter plot showing the activations of each atom across trials. The color of each point 
    represents the atom, and the size of the point is proportional to the activation value. The scatter plot is sorted 
    by the provided class labels, and boundaries between different class labels are indicated with horizontal dotted lines.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning model containing the atoms (`v_hat_`) and activations (`_z_hat`).
    class_labels : numpy.ndarray
        A binary classification of trials, with 1 representing 'Y' trials and 0 representing 'N' trials.
    sfreq : float
        The sampling frequency of the data used to fit the CDL model, in Hz.
    choose_atoms : list, optional
        A list of atom indices to plot. If not provided, all atoms are plotted.
    color_cycle : list, optional
        A list of colors to be used for the different atoms. If not provided, the default color cycle is used.
    scale_factor : float, optional
        A scaling factor for the size of the scatter plot dots. Default is 1.

    Returns
    -------
    None

    Notes
    -----
    - This function uses `itertools.cycle` to cycle through the provided or default color cycle.
    - The activation times are converted to milliseconds for the x-axis.
    - The scatter plot is sorted based on the class labels to visually separate different trial classes.

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> class_labels = np.random.randint(0, 2, size=(cdl_model._z_hat.shape[0],))
    >>> plot_atom_scatter(cdl_model, class_labels, sfreq=256)
    """
    
    sparse_code = cdl._z_hat  # n_trials x n_atoms x bins_i
    n_atoms, n_times_atom = cdl.v_hat_.shape
    len_data = sparse_code.shape[2] + n_times_atom - 1

    if choose_atoms == None:
        choose_atoms = list(range(n_atoms))

    if color_cycle is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    color_cycle = itertools.cycle(color_cycle)

    # Flatten the class_labels array
    class_labels = class_labels.ravel()

    # Sort the sparse code by class_labels
    sorted_indices = np.argsort(class_labels)
    sparse_code = sparse_code[sorted_indices]

    # Find the unique class labels and their corresponding indices
    unique_labels, boundary_indices = np.unique(class_labels[sorted_indices], return_index=True)

    fig, ax = plt.subplots()

    for i in choose_atoms:
        which_atom = i  # because they're sorted

        time_values = []
        trial_indices = []
        dot_sizes = []

        # loop over trials
        for r in range(sparse_code.shape[0]):
            for b in range(sparse_code.shape[2]):
                if sparse_code[r, which_atom, b] > 0:
                    time_values.append((b + n_times_atom // 2) / sfreq * 1000)
                    trial_indices.append(r)
                    dot_sizes.append(sparse_code[r, which_atom, b] * scale_factor)

        ax.scatter(time_values, trial_indices, label='atom #' + str(which_atom), alpha=0.5, color=next(color_cycle),
                   s=dot_sizes)

    # Draw dotted horizontal lines indicating the boundaries between different class labels
    for boundary_index in boundary_indices:
        ax.axhline(y=boundary_index + 1, linestyle='--', color='black', linewidth=1)

    ax.legend()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial index')
    fig.set_size_inches(18.5, 10.25)
    # plt.savefig('output/atom_scatter' + str(cdl.n_atoms) + '_len' + str(cdl.v_hat_.shape[1]) + '.png', dpi=100)
    plt.show()
    
    
def plot_extrema_scatter(cdl, sfreq,
                         YorN=None, choose_atoms=None, scale_factor=1, variance_threshold=0.01):
    """
    Plots a scatter plot of peaks and troughs found using find_extrema with combine_atoms=True.

    This function generates a scatter plot showing the peaks and troughs of the activations across trials from a 
    fitted CDL model. The color of each point represents the type of extrema (peaks in red, troughs in blue), and 
    the size of the point is proportional to the activation value. The scatter plot is optionally sorted by a 
    binary classification of trials.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning model containing the atoms (`v_hat_`) and activations (`_z_hat`).
    sfreq : float
        The sampling frequency of the data used to fit the CDL model, in Hz.
    YorN : numpy.ndarray, optional
        A binary classification of trials, with 1 representing 'Y' trials and 0 representing 'N' trials. If provided, 
        the scatter plot will be sorted by these labels.
    choose_atoms : list of int, optional
        A list of atom indices to include in peak and trough detection. If not provided, all available atoms will be used.
    scale_factor : float, optional
        A scaling factor for the size of the scatter plot dots. Default is 1.
    variance_threshold : float, optional
        A threshold for the variance used in peak and trough detection. Default is 0.01.

    Returns
    -------
    None

    Notes
    -----
    - This function uses the `find_extrema` function with `combine_atoms=True` to detect peaks and troughs.
    - If `YorN` is provided, the scatter plot will be sorted by these labels, and a boundary line will be drawn between 
      different classes of trials.

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> YorN = np.random.randint(0, 2, size=(cdl_model._z_hat.shape[0],))
    >>> plot_extrema_scatter(cdl_model, sfreq=256, YorN=YorN)
    """
    peak_array, trough_array = find_extrema(cdl, choose_atoms=choose_atoms, combine_atoms=True, variance_threshold=variance_threshold)

    if YorN is not None:
        YorN = YorN.ravel()
        sorted_indices = np.argsort(YorN)
        peak_array = peak_array[sorted_indices]
        trough_array = trough_array[sorted_indices]

        boundary_index = np.where(np.diff(YorN[sorted_indices]))[0][0] + 1

    fig, ax = plt.subplots()

    peak_indices = []
    trough_indices = []

    for r in range(peak_array.shape[0]):
        for t in range(peak_array.shape[2]):
            if peak_array[r, 0, t]:
                peak_indices.append((t/sfreq*1000, r))
            if trough_array[r, 0, t]:
                trough_indices.append((t/sfreq*1000, r))

    peak_x, peak_y = zip(*peak_indices)
    trough_x, trough_y = zip(*trough_indices)

    ax.scatter(peak_x, peak_y, alpha=0.5, color='red', s=scale_factor, label='Peaks')
    ax.scatter(trough_x, trough_y, alpha=0.5, color='blue', s=scale_factor, label='Troughs')

    if YorN is not None:
        ax.axhline(y=boundary_index, linestyle='--', color='black', linewidth=1)
        ax.fill_between([0, peak_array.shape[2]/sfreq*1000], boundary_index, facecolor='gray', alpha=0.2)

    ax.legend()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial index')
    fig.set_size_inches(18.5, 10.25)
    # plt.savefig('output/extrema_scatter.png', dpi=100)
    plt.show()



def plot_extrema_cpdf(cdl, choose_atoms=None, n_bins=100):
    """
    Plots the conditional probability density function (CPDF) of peaks and troughs over time for a given
    Convolutional Dictionary Learning (CDL) model.

    This function generates a line plot displaying the CPDF of peaks and troughs from a fitted CDL model. 
    The CPDF is calculated separately for peaks and troughs, with the x-axis representing the time index 
    and the y-axis representing the probability density. The peaks and troughs are identified using the 
    `find_extrema` function with `combine_atoms=True`.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning model containing the atoms (`v_hat_`) and activations (`_z_hat`).
    choose_atoms : list, optional
        A list of atoms to be included when finding the extrema. If not provided, all atoms will be considered.
    n_bins : int, optional
        The number of bins to use for the histogram. Default is 100.

    Returns
    -------
    None

    Notes
    -----
    - This function uses the `find_extrema` function to identify peaks and troughs.
    - The `combine_atoms` parameter in `find_extrema` is set to True to consider combined extrema across all selected atoms.

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> plot_extrema_cpdf(cdl_model, n_bins=50)
    """
    peak_array, trough_array = find_extrema(cdl, choose_atoms=choose_atoms, combine_atoms=True)

    peak_indices = []
    trough_indices = []

    for r in range(peak_array.shape[0]):
        for t in range(peak_array.shape[2]):
            if peak_array[r, 0, t]:
                peak_indices.append((t, r))
            if trough_array[r, 0, t]:
                trough_indices.append((t, r))

    peak_x, _ = zip(*peak_indices)
    trough_x, _ = zip(*trough_indices)

    fig, ax = plt.subplots()

    hist_data, bin_edges = np.histogram(peak_x, bins=n_bins, range=(0, peak_array.shape[2]), density=True)
    ax.plot(bin_edges[:-1], hist_data, label='Peaks', color='red', alpha=0.7)

    hist_data, bin_edges = np.histogram(trough_x, bins=n_bins, range=(0, trough_array.shape[2]), density=True)
    ax.plot(bin_edges[:-1], hist_data, label='Troughs', color='blue', alpha=0.7)

    ax.legend()
    ax.set_xlabel('Time index')
    ax.set_ylabel('Probability Density')
    fig.set_size_inches(18.5, 10.25)
    # plt.savefig('output/extrema_cpdf.png', dpi=100)
    plt.show()

        
    
def get_atom_occurrences(sparse_code, X_all, YorN_all, trial, n_times_atom, which_atom, 
                         win_adjust=None):
    """
    Extracts segments from the data corresponding to the occurrences of a specified atom,
    adjusts the window around the atom occurrences, and pads the segments if necessary.

    This function identifies segments of the raw data where a specified atom is active, adjusts 
    the extraction window around each occurrence based on the `win_adjust` parameter, and pads the 
    segments if they exceed the data boundaries.

    Parameters
    ----------
    sparse_code : ndarray
        The sparse code representation of the data, with shape (n_trials, n_atoms, n_time_points).
    X_all : ndarray
        The raw data with shape (n_trials, n_channels, n_time_points).
    YorN_all : ndarray
        A binary array indicating the trial classification, with 1 representing 'Y' trials and 0 representing 'N' trials.
    trial : int
        The specific trial classification (0 or 1) to filter the trials for extraction.
    n_times_atom : int
        The number of time points in the atom.
    which_atom : int
        The index of the atom for which occurrences are being extracted.
    win_adjust : tuple of int, optional
        A tuple of two integers representing the number of time points to adjust the window
        around the atom occurrences (default is (0, 0)).

    Returns
    -------
    matches : ndarray
        The concatenated segments of the data corresponding to the atom occurrences, with shape
        (n_occurrences, n_times_atom + sum(np.abs(win_adjust))).
    trial_indices : list
        A list of trial indices corresponding to each occurrence.

    Example
    -------
    >>> sparse_code = np.random.rand(10, 5, 100)
    >>> X_all = np.random.rand(10, 1, 100)
    >>> YorN_all = np.random.randint(0, 2, (10, 1))
    >>> trial = 1
    >>> n_times_atom = 20
    >>> which_atom = 2
    >>> matches, trial_indices = get_atom_occurrences(sparse_code, X_all, YorN_all, trial, n_times_atom, which_atom, win_adjust=(5, 5))

    Notes
    -----
    - The function uses `np.clip` to ensure the indices are within the bounds of the data array.
    - The segments are padded with `np.nan` if the window adjustment exceeds the data boundaries.
    """
    
    if win_adjust == None:
        win_adjust = (0,0)
    
    matches = np.empty((0, n_times_atom + sum(np.abs(win_adjust))), dtype='float64')
    trial_indices = []
    
    # loop over trials
    for r in range(sparse_code.shape[0]):
        
        if YorN_all[r,0] != trial: continue
        
        # loop over activation-based time
        for b in range(sparse_code.shape[2]):
            
            if sparse_code[r, which_atom, b] > 0:
                start_idx = np.clip(b + win_adjust[0], 0, X_all.shape[2] - 1)
                end_idx = np.clip(b + n_times_atom + win_adjust[1], start_idx, X_all.shape[2])
                
                seg = X_all[r, 0, start_idx:end_idx][:, None].transpose()
                
                # Calculate the amount of padding needed for each side
                left_pad = max(0, -(b + win_adjust[0]))
                right_pad = max(0, b + n_times_atom + win_adjust[1] - X_all.shape[2])
                
                # Pad the segment if necessary, to match the desired output shape
                if left_pad > 0 or right_pad > 0:
                    seg = np.pad(seg, ((0, 0), (left_pad, right_pad)),
                                 mode='constant', constant_values=np.nan)
                
                matches = np.concatenate([matches, seg], axis=0)
                trial_indices.append(r)
                
    return matches, trial_indices

    
    
def plot_atom_comparison(cdl, X_all, YorN_all, sfreq, 
                         method='std', n_boot=1000, choose_atoms=None, win_adjust=(0,0), align_by='start'):
    """
    Plots a comparison of the average signal segments for each atom between 'Y' and 'N' trials.

    This function generates a plot with subplots for each atom, showing the average signal segment in 'Y' trials 
    (red) and 'N' trials (blue) with shaded areas representing the standard error of the mean or bootstrap confidence 
    intervals. The segments are aligned based on the specified method.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning model containing the atoms (`v_hat_`) and activations (`_z_hat`).
    X_all : numpy.ndarray
        A data array containing the signal segments for all trials, with shape (n_trials, n_channels, n_time_points).
    YorN_all : numpy.ndarray
        A binary classification of trials, with 1 representing 'Y' trials and 0 representing 'N' trials.
    sfreq : float
        The sampling frequency of the data in Hz.
    method : str, optional
        Method of computing the standard error or confidence intervals. Can be 'std' (standard error) or 'bootstrap'. 
        Default is 'std'.
    n_boot : int, optional
        Number of bootstraps for the bootstrap method. Default is 1000.
    choose_atoms : list of int, optional
        List of indices of atoms in the dictionary to plot. If not provided, all atoms are plotted.
    win_adjust : tuple of int, optional
        Number of samples to adjust the window start/end positions. Default is (0, 0).
    align_by : str or int, optional
        Position by which to align the matched segments. Can be 'start', 'peak', 'trough', 'match_average', 'data_average', 
        or an int for the index in the adjusted window. Default is 'start'.

    Returns
    -------
    None

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> X_all = np.random.rand(100, 1, 1000)
    >>> YorN_all = np.random.randint(0, 2, (100, 1))
    >>> plot_atom_comparison(cdl_model, X_all, YorN_all, sfreq=256)

    Notes
    -----
    - The function uses `get_atom_occurrences` to extract and align segments of the data.
    - Shaded areas represent the standard error or bootstrap confidence intervals.
    - The function assumes that the data array `X_all` has trials along the first dimension and time points along the last dimension.
    """
    
    sparse_code = cdl._z_hat # n_trials x n_atoms x bins_i
    # sparse_code_Y = sparse_code[(YorN_all == 1).ravel(),:,:]
    # sparse_code_N = sparse_code[(YorN_all == 0).ravel(),:,:]
    
    if choose_atoms is None:
        choose_atoms = list(range(cdl.n_atoms))
        
    n_atoms, n_times_atom = cdl.v_hat_.shape

    n_subplots = len(choose_atoms)
    fig, axs = plt.subplots(nrows=int(np.ceil(n_subplots/6)), ncols=min(6, n_subplots))
    axs = np.atleast_2d(axs).ravel()
    
    if align_by == 'start':
        if win_adjust[0] < 0: align_idx = -win_adjust[0]
        else: align_idx = 0
    elif isinstance(align_by, int):
        align_idx = align_by

    for i, atom_i in enumerate(choose_atoms):
        
        atom = cdl.v_hat_[atom_i]
        if align_by == 'peak': align_idx = np.argmax(atom) + -win_adjust[0]
        elif align_by == 'trough': align_idx = np.argmin(atom) + -win_adjust[0]
        
        matches_Y, trials_Y = get_atom_occurrences(sparse_code, X_all, YorN_all, trial=1, 
                                                   n_times_atom=n_times_atom, which_atom=atom_i, win_adjust=win_adjust)
        matches_N, trials_N = get_atom_occurrences(sparse_code, X_all, YorN_all, trial=0, 
                                                   n_times_atom=n_times_atom, which_atom=atom_i, win_adjust=win_adjust)
                
        if align_by == 'match_average':
            align_level_Y = np.nanmean(matches_Y, axis=None)
            align_level_N = np.nanmean(matches_N, axis=None)
        elif align_by == 'data_average':
            align_level_Y = np.nanmean(X_all[trials_Y, 0, :], axis=1).mean()
            align_level_N = np.nanmean(X_all[trials_N, 0, :], axis=1).mean()
        elif align_by == None:
            align_level_Y = 0
            align_level_N = 0
        else:
            align_level_Y = np.nanmean(matches_Y, axis=0)[align_idx]
            align_level_N = np.nanmean(matches_N, axis=0)[align_idx]
        
        matches_Y_mean = np.nanmean(matches_Y, axis=0) - align_level_Y
        matches_N_mean = np.nanmean(matches_N, axis=0) - align_level_N
        
        if method == 'std':
            matches_Y_std = np.nanstd(matches_Y, axis=0) / np.sqrt(matches_Y.shape[0])
            matches_N_std = np.nanstd(matches_N, axis=0) / np.sqrt(matches_N.shape[0])
        
            Y_low_bound, Y_high_bound = matches_Y_mean - 2 * matches_Y_std, matches_Y_mean + 2 * matches_Y_std
            N_low_bound, N_high_bound = matches_N_mean - 2 * matches_N_std, matches_N_mean + 2 * matches_N_std
        
        elif method == 'bootstrap':
            bootstrap_means_Y = np.zeros((n_boot, n_times_atom + sum(np.abs(win_adjust))))
            bootstrap_means_N = np.zeros((n_boot, n_times_atom + sum(np.abs(win_adjust))))
        
            for b in range(n_boot):
                bootstrap_sample_Y = matches_Y[np.random.choice(matches_Y.shape[0], size=matches_Y.shape[0], replace=True), :]
                bootstrap_means_Y[b, :] = np.nanmean(bootstrap_sample_Y, axis=0) - align_level_Y
        
                bootstrap_sample_N = matches_N[np.random.choice(matches_N.shape[0], size=matches_N.shape[0], replace=True), :]
                bootstrap_means_N[b, :] = np.nanmean(bootstrap_sample_N, axis=0) - align_level_N
        
            Y_low_bound, Y_high_bound = np.percentile(bootstrap_means_Y, [2.5, 97.5], axis=0)
            N_low_bound, N_high_bound = np.percentile(bootstrap_means_N, [2.5, 97.5], axis=0)
            
        axs[i].fill_between(np.arange(n_times_atom + sum(np.abs(win_adjust)))/sfreq*1000, 
                            Y_low_bound, Y_high_bound, alpha=0.5, color='r')
        axs[i].fill_between(np.arange(n_times_atom + sum(np.abs(win_adjust)))/sfreq*1000, 
                            N_low_bound, N_high_bound, alpha=0.5)
        
        axs[i].plot(np.arange(n_times_atom + sum(np.abs(win_adjust)))/sfreq*1000, matches_Y_mean, color='r')
        axs[i].plot(np.arange(n_times_atom + sum(np.abs(win_adjust)))/sfreq*1000, matches_N_mean)
        
        axs[i].set_title('atom #' + str(atom_i))
        axs[i].set_xlim(0, (n_times_atom + sum(np.abs(win_adjust))) / sfreq * 1000)
        axs[i].set_xlabel('Time (ms)')
        axs[i].grid(True)

    fig.set_size_inches(min(min(6, n_subplots) * 4, 19), int(np.ceil(n_subplots/6)) * 5)
    # fig.savefig('output/datacompare_len' + str(cdl.v_hat_.shape[1]) + '.png', dpi=100)
    plt.show()
    
    
def get_extrema_occurrences(peak_array, trough_array, X_all, sfreq, YorN_all, trial, 
                            extrema_type='trough', segment_win=(-50,50), win_filter=None):
    """
    Extracts segments from the data corresponding to the occurrences of extrema, either peaks or troughs.
    Adjusts the window around the extrema occurrences and pads the segments if necessary.

    Parameters
    ----------
    peak_array : ndarray
        Array containing the occurrences of peaks, with shape (n_trials, n_channels, n_time_points).
    trough_array : ndarray
        Array containing the occurrences of troughs, with shape (n_trials, n_channels, n_time_points).
    X_all : ndarray
        The raw data with shape (n_trials, n_channels, n_time_points).
    sfreq : float
        The sampling frequency of the data in Hz.
    YorN_all : ndarray
        A binary classification of trials, with 1 representing 'Y' trials and 0 representing 'N' trials.
    trial : int
        The specific trial classification (0 or 1) to filter the trials for extraction.
    extrema_type : str, optional
        The type of extrema to extract ('peak' or 'trough'). Default is 'trough'.
    segment_win : tuple of int, optional
        A tuple of two integers representing the window size around the extrema occurrences in time points. Default is (-50, 50).
    win_filter : tuple of float, optional
        A tuple of two floats representing the window filter range in milliseconds. If provided, only extrema within this window will be considered.

    Returns
    -------
    matches : ndarray
        The concatenated segments of the data corresponding to the extrema occurrences, with shape (n_occurrences, sum(np.abs(segment_win))).
    trial_indices : list
        A list of trial indices corresponding to each occurrence.

    Example
    -------
    >>> peak_array = np.random.randint(0, 2, (10, 1, 1000))
    >>> trough_array = np.random.randint(0, 2, (10, 1, 1000))
    >>> X_all = np.random.rand(10, 1, 1000)
    >>> sfreq = 256
    >>> YorN_all = np.random.randint(0, 2, 10)
    >>> matches, trial_indices = get_extrema_occurrences(peak_array, trough_array, X_all, sfreq, YorN_all, trial=1, extrema_type='peak', segment_win=(-100, 100))

    Notes
    -----
    - This function uses `np.clip` to ensure the indices are within the bounds of the data array.
    - The segments are padded with `np.nan` if the window adjustment exceeds the data boundaries.
    - If `win_filter` is provided, the function converts the window filter range from milliseconds to time index values.
    """
    
    matches = np.empty((0, sum(np.abs(segment_win))), dtype='float64')
    trial_indices = []
    
    # Choose the appropriate extrema array based on the extrema_type
    extrema_array = peak_array if extrema_type == 'peak' else trough_array
    
    # Convert win_filter into time index values in X_all
    if win_filter:
        win_filter_idx = (int(win_filter[0] * sfreq / 1000), int(win_filter[1] * sfreq / 1000))

    # Loop over trials
    for r in range(extrema_array.shape[0]):
        
        if YorN_all[r] != trial: continue
        
        # Loop over time
        for t in range(extrema_array.shape[2]):
            if extrema_array[r, 0, t]:
                # Check if the extrema occur within the specified window in X_all
                if win_filter:
                    if t < win_filter_idx[0] or t > win_filter_idx[1]:
                        continue

                start_idx = np.clip(t + segment_win[0], 0, X_all.shape[2] - 1)
                end_idx = np.clip(t + segment_win[1], start_idx, X_all.shape[2])
                
                seg = X_all[r, 0, start_idx:end_idx][:, None].transpose()
                
                # Calculate the amount of padding needed for each side
                left_pad = max(0, -(t + segment_win[0]))
                right_pad = max(0, t + segment_win[1] - X_all.shape[2])
                
                # Pad the segment if necessary, to match the desired output shape
                if left_pad > 0 or right_pad > 0:
                    seg = np.pad(seg, ((0, 0), (left_pad, right_pad)),
                                 mode='constant', constant_values=np.nan)
                
                matches = np.concatenate([matches, seg], axis=0)
                trial_indices.append(r)
                
    return matches, trial_indices



def align_by_extrema(cdl, X_all, class_labels, sfreq, method='std', n_boot=1000,
                     choose_atoms=None, labels=None, segment_win=(-50, 50), 
                     win_filter=None, alignx_by='trough', aligny_by='data_average', 
                     plot=False):
    """
    Aligns signal segments by extrema (peaks or troughs) and optionally plots the comparison between different trial classes.

    This function identifies extrema (peaks or troughs) in the signal segments using a fitted Convolutional Dictionary 
    Learning (CDL) model, aligns the segments based on these extrema, and computes the average signal segment for each 
    trial class. It can also plot the average signal segments with shaded areas representing the standard error or 
    bootstrap confidence intervals.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning model containing the atoms (`v_hat_`) and activations (`_z_hat`).
    X_all : numpy.ndarray
        A data array containing the signal segments for all trials, with shape (n_trials, n_channels, n_time_points).
    class_labels : numpy.ndarray
        A binary or categorical classification of trials.
    sfreq : float
        The sampling frequency of the data in Hz.
    method : str, optional
        Method of computing the standard error or confidence intervals. Can be 'std' (standard error) or 'bootstrap'. 
        Default is 'std'.
    n_boot : int, optional
        Number of bootstraps for the bootstrap method. Default is 1000.
    choose_atoms : list of int, optional
        List of indices of atoms in the dictionary to use for finding extrema. If not provided, all atoms are used.
    labels : list, optional
        Labels for the atoms. If not provided, no labels are used.
    segment_win : tuple of int, optional
        A tuple representing the window size around the extrema occurrences in time points. Default is (-50, 50).
    win_filter : tuple of float, optional
        A tuple representing the window filter range in milliseconds. If provided, only extrema within this window will be considered.
    alignx_by : str, optional
        The type of extrema to align by ('peak' or 'trough'). Default is 'trough'.
    aligny_by : str, optional
        Method to align the y-axis. Can be 'start', 'peak', 'trough', 'match_average', 'data_average', or None. Default is 'data_average'.
    plot : bool, optional
        If True, a plot of the average signal segments with confidence intervals is generated. Default is False.

    Returns
    -------
    results : dict
        A dictionary containing the results with the following keys:
        - 'time': Array of time points in milliseconds.
        - 'means': List of mean signal segments for each trial class.
        - 'lower_bounds': List of lower bounds of confidence intervals for each trial class.
        - 'upper_bounds': List of upper bounds of confidence intervals for each trial class.
        - 'labels': List of unique class labels.
        - 'matches': List of matches for each class.

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> X_all = np.random.rand(100, 1, 1000)
    >>> class_labels = np.random.randint(0, 2, 100)
    >>> results = align_by_extrema(cdl_model, X_all, class_labels, sfreq=256, plot=True)

    Notes
    -----
    - The function uses `find_extrema` to identify peaks and troughs in the signal segments.
    - Shaded areas represent the standard error or bootstrap confidence intervals.
    - The function assumes that the data array `X_all` has trials along the first dimension and time points along the last dimension.
    """

    peak_array, trough_array = find_extrema(cdl, labels=labels, combine_atoms=True)
    # print(np.any(trough_array))

    if alignx_by == 'peak':
        extrema_type = 'peak'
    elif alignx_by == 'trough':
        extrema_type = 'trough'
    else:
        extrema_type = 'trough'  # Default to 'peak' if align_by is not 'trough'

    unique_class_labels = np.unique(class_labels)
    if len(unique_class_labels) > 2:
        bounds = [2.5, 97.5]
    elif len(unique_class_labels) == 2:
        bounds = [5, 95]

    # Plotting
    if plot:
        fig, ax = plt.subplots()

    results = {
        'time': np.arange(sum(np.abs(segment_win))) / sfreq * 1000,
        'means': [],
        'lower_bounds': [],
        'upper_bounds': [],
        'labels': [],
        'matches': [],
    }

    for idx, label in enumerate(unique_class_labels):

        matches, trials = get_extrema_occurrences(peak_array, trough_array, X_all, sfreq, class_labels, trial=label,
                                                  extrema_type=extrema_type, segment_win=segment_win,
                                                  win_filter=win_filter)

        if aligny_by == 'match_average':
            align_level = np.nanmean(matches, axis=None)
        elif aligny_by == 'data_average':
            align_level = np.nanmean(X_all[trials, 0, :], axis=1).mean()
        elif aligny_by is None:
            align_level = 0
        elif aligny_by == 'start':
            align_level = np.nanmean(matches, axis=0)[segment_win[0]]
        else:
            align_level = np.nanmean(matches, axis=0)[-segment_win[0]]

        matches_mean = np.nanmean(matches, axis=0) - align_level

        if method == 'std':
            matches_std = np.nanstd(matches, axis=0) / np.sqrt(matches.shape[0])
            low_bound, high_bound = matches_mean - 2 * matches_std, matches_mean + 2 * matches_std
        elif method == 'bootstrap':
            bootstrap_means = np.zeros((n_boot, sum(np.abs(segment_win))))
            for b in range(n_boot):
                bootstrap_sample = matches[np.random.choice(matches.shape[0], size=matches.shape[0], replace=True), :]
                bootstrap_means[b, :] = np.nanmean(bootstrap_sample, axis=0) - align_level
            low_bound, high_bound = np.percentile(bootstrap_means, bounds, axis=0)

        results['means'].append(matches_mean)
        results['lower_bounds'].append(low_bound)
        results['upper_bounds'].append(high_bound)
        results['labels'].append(label)
        results['matches'].append(matches)

        if plot:
            ax.fill_between(results['time'], low_bound, high_bound, alpha=0.5, label=label,
                            color=colorblind10_palette[idx])
            ax.plot(results['time'], matches_mean, color=colorblind10_palette[idx])

    if plot:
        ax.set_xlim(0, results['time'][-1])
        ax.set_xlabel('Time (ms)')
        ax.legend()
        ax.grid(True)
        fig.set_size_inches(10, 5)
        plt.show()

    # Convert lists to arrays for easier indexing
    results['means'] = np.array(results['means'])
    results['lower_bounds'] = np.array(results['lower_bounds'])
    results['upper_bounds'] = np.array(results['upper_bounds'])
    results['matches'] = np.array(results['matches'])

    return results



def plot_aligned_extrema(cdl, X_all, class_labels, sfreq,
                         method='std', n_boot=1000, choose_atoms=None,
                         segment_win=(-50, 50), win_filter=None,
                         alignx_by='trough', aligny_by='data_average'):
    """
    Plots the average signal segments aligned by extrema (peaks or troughs) for different trial classes.

    This function identifies extrema (peaks or troughs) in the signal segments using a fitted Convolutional Dictionary 
    Learning (CDL) model, aligns the segments based on these extrema, and plots the average signal segment for each 
    trial class with shaded areas representing the standard error or bootstrap confidence intervals.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning model containing the atoms (`v_hat_`) and activations (`_z_hat`).
    X_all : numpy.ndarray
        A data array containing the signal segments for all trials, with shape (n_trials, n_channels, n_time_points).
    class_labels : numpy.ndarray
        A binary or categorical classification of trials.
    sfreq : float
        The sampling frequency of the data in Hz.
    method : str, optional
        Method of computing the standard error or confidence intervals. Can be 'std' (standard error) or 'bootstrap'. 
        Default is 'std'.
    n_boot : int, optional
        Number of bootstraps for the bootstrap method. Default is 1000.
    choose_atoms : list of int, optional
        List of indices of atoms in the dictionary to use for finding extrema. If not provided, all atoms are used.
    segment_win : tuple of int, optional
        A tuple representing the window size around the extrema occurrences in time points. Default is (-50, 50).
    win_filter : tuple of float, optional
        A tuple representing the window filter range in milliseconds. If provided, only extrema within this window will be considered.
    alignx_by : str, optional
        The type of extrema to align by ('peak' or 'trough'). Default is 'trough'.
    aligny_by : str, optional
        Method to align the y-axis. Can be 'start', 'peak', 'trough', 'match_average', 'data_average', or None. Default is 'data_average'.

    Returns
    -------
    None

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> X_all = np.random.rand(100, 1, 1000)
    >>> class_labels = np.random.randint(0, 2, 100)
    >>> plot_aligned_extrema(cdl_model, X_all, class_labels, sfreq=256)

    Notes
    -----
    - The function uses `find_extrema` to identify peaks and troughs in the signal segments.
    - Shaded areas represent the standard error or bootstrap confidence intervals.
    - The function assumes that the data array `X_all` has trials along the first dimension and time points along the last dimension.
    - This function is unused in the latest implementation.
    """
    

    peak_array, trough_array = find_extrema(cdl, combine_atoms=True)

    if alignx_by == 'peak':
        extrema_type = 'peak'
    elif alignx_by == 'trough':
        extrema_type = 'trough'
    else:
        extrema_type = 'trough' # Default to 'peak' if align_by is not 'trough'
        
    unique_class_labels = np.unique(class_labels)
    if len(unique_class_labels) > 2:
        bounds = [2.5, 97.5]
    elif len(unique_class_labels) == 2:
        bounds = [5, 95]

    # Plotting
    fig, ax = plt.subplots()

    for idx, label in enumerate(unique_class_labels):

        matches, trials = get_extrema_occurrences(peak_array, trough_array, X_all, sfreq, class_labels, trial=label,
                                                  extrema_type=extrema_type, segment_win=segment_win,
                                                  win_filter=win_filter)

        if aligny_by == 'match_average':
            align_level = np.nanmean(matches, axis=None)
        elif aligny_by == 'data_average':
            align_level = np.nanmean(X_all[trials, 0, :], axis=1).mean()
        elif aligny_by is None:
            align_level = 0
        elif aligny_by == 'start':
            align_level = np.nanmean(matches, axis=0)[segment_win[0]]
        else:
            align_level = np.nanmean(matches, axis=0)[-segment_win[0]]

        matches_mean = np.nanmean(matches, axis=0) - align_level

        if method == 'std':
            matches_std = np.nanstd(matches, axis=0) / np.sqrt(matches.shape[0])

            low_bound, high_bound = matches_mean - 2 * matches_std, matches_mean + 2 * matches_std

        elif method == 'bootstrap':
            bootstrap_means = np.zeros((n_boot, sum(np.abs(segment_win))))

            for b in range(n_boot):
                bootstrap_sample = matches[np.random.choice(matches.shape[0], size=matches.shape[0],
                                                             replace=True), :]
                bootstrap_means[b, :] = np.nanmean(bootstrap_sample, axis=0) - align_level

            low_bound, high_bound = np.percentile(bootstrap_means, bounds, axis=0)

        ax.fill_between(np.arange(sum(np.abs(segment_win))) / sfreq * 1000,
                        low_bound, high_bound, alpha=0.5, label=label,
                        color=colorblind10_palette[idx])
        ax.plot(np.arange(sum(np.abs(segment_win))) / sfreq * 1000, matches_mean,
                color=colorblind10_palette[idx])

    ax.set_xlim(0, (sum(np.abs(segment_win))) / sfreq * 1000)
    ax.set_xlabel('Time (ms)')
    ax.legend()
    ax.grid(True)

    fig.set_size_inches(10, 5)
    plt.show()



    
def classify_cycle_components(cdl, 
                              variance_threshold=0.01,
                              margin=2):
    """
    Classify the atoms in the given Convolutional Dictionary Learning (CDL) object into four categories:
    'peak', 'trough', 'rise', and 'decay', based on their temporal representations. Atoms with variance below
    the specified threshold will not be classified.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A fitted Convolutional Dictionary Learning (CDL) object containing the atoms to be classified.
    variance_threshold : float, optional
        The relative threshold (in range [0, 1]) for the variance of the atom's temporal representation
        below which an atom will not be classified. The default is 0.01.
    margin : int, optional
        The margin around the middle of the temporal representation where maxima or minima should not be considered
        as indicative of 'peak' or 'trough'. The default is 2.

    Returns
    -------
    labels : list of str
        A list of labels corresponding to each atom in the CDL object. The labels can be 'peak', 'trough',
        'rise', 'decay', or an empty string if the atom's variance is below the threshold.

    Notes
    -----
    - 'peak': Atoms with global maxima close to the middle of the temporal representation.
    - 'trough': Atoms with global minima close to the middle of the temporal representation.
    - 'rise': Atoms with maxima closest to the end and minima close to the beginning of the temporal representation.
    - 'decay': Atoms with maxima closest to the beginning and minima close to the end of the temporal representation.

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> labels = classify_cycle_components(cdl_model)
    >>> print(labels)
    """
    
    n_atoms, n_times_atom = cdl.v_hat_.shape
    labels = [""] * n_atoms
    variances = np.var(cdl.v_hat_, axis=1)
    min_var = np.min(variances)
    max_var = np.max(variances)
    # var_threshold = min_var + (max_var - min_var) * variance_threshold

    for i in range(n_atoms):
        atom = cdl.v_hat_[i] * cdl.u_hat_[i]
        min_idx = np.argmin(atom)
        max_idx = np.argmax(atom)
        mid_idx = n_times_atom // 2
    
        if variances[i] < variance_threshold:
            continue
    
        if margin < max_idx < n_times_atom - margin:
            labels[i] = "peak"
        elif margin < min_idx < n_times_atom - margin:
            labels[i] = "trough"
        else:
            if min_idx < max_idx and min_idx < mid_idx:
                labels[i] = "rise"
            elif max_idx < min_idx and mid_idx < min_idx:
                labels[i] = "decay"
    
    return labels


from collections import defaultdict

def find_extrema(cdl, labels=None, roll=None, choose_atoms=None, combine_atoms=False, variance_threshold=0.01, margin=2):
    """
    Given a Convolutional Dictionary Learning (CDL) object and a list of labels for each atom, find the indices of 
    peaks and troughs in the sparse code. A peak is appended if the label of an atom is 'peak', 'rise', or 'decay', 
    and a trough is appended if the label of an atom is 'trough', 'rise', or 'decay'.

    Parameters
    ----------
    cdl : alphacsc.ConvolutionalDictionaryLearning
        A Convolutional Dictionary Learning (CDL) object with learned atoms (`v_hat_`) and sparse code (`_z_hat`).
    labels : list of str, optional
        A list of labels for each atom in `cdl.v_hat_`. The labels should be one of 'peak', 'trough', 'rise', or 'decay'.
        If not provided, the labels are generated using `classify_cycle_components`.
    roll : int, optional
        The number of indices to consider when finding the "true" activation index by taking the max of any consecutive 
        non-zero activation weights. By default, it is set to the ceiling of one-fourth of the atom length.
    choose_atoms : list of int, optional
        A list of atom indices to be considered for peak and trough detection. If not provided, all available atoms are used.
    combine_atoms : bool, optional
        If True, combines the extrema across atoms by weighting the positions of activations. Default is False.
    variance_threshold : float, optional
        The relative threshold (in range [0, 1]) for the variance of the atom's temporal representation below which an 
        atom will not be classified. Default is 0.01.
    margin : int, optional
        The margin around the middle of the temporal representation where maxima or minima should not be considered as 
        indicative of 'peak' or 'trough'. Default is 2.

    Returns
    -------
    peak_array : ndarray
        A 3D boolean array of the same dimensions as the input data, where True indicates the presence of a peak. The first 
        dimension represents the trial index, the second dimension is typically 0, and the third dimension represents the time index.
    trough_array : ndarray
        A 3D boolean array of the same dimensions as the input data, where True indicates the presence of a trough. The first 
        dimension represents the trial index, the second dimension is typically 0, and the third dimension represents the time index.

    Example
    -------
    >>> from alphacsc import ConvolutionalDictionaryLearning
    >>> import numpy as np
    >>> # Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
    >>> peak_array, trough_array = find_extrema(cdl_model)

    Notes
    -----
    - This function uses `classify_cycle_components` to generate labels if they are not provided.
    - The `roll` parameter is used to handle consecutive non-zero activation weights by taking the max within a window.
    - The function assumes that the data array `X_all` has trials along the first dimension and time points along the last dimension.
    """
    if labels is None:
        labels = classify_cycle_components(cdl, variance_threshold=variance_threshold)
    
    if choose_atoms is None:
        choose_atoms = list(range(cdl.v_hat_.shape[0]))
    
    n_atoms, n_times_atom = cdl.v_hat_.shape
    sparse_code = cdl._z_hat
    n_trials = sparse_code.shape[0]
    n_channels = cdl.u_hat_.shape[1]
    n_times = sparse_code.shape[2] + n_times_atom - 1
    
    if roll is None:
        roll = int(np.ceil(n_times_atom / 2))

    peak_array = np.zeros((n_trials, n_channels, n_times), dtype=bool)
    trough_array = np.zeros((n_trials, n_channels, n_times), dtype=bool)

    extrema_candidates = defaultdict(list)
    
    for i in range(n_atoms):
        if i not in choose_atoms:
            continue
    
        atom = cdl.v_hat_[i] * cdl.u_hat_[i]
        min_idx = np.argmin(atom)
        max_idx = np.argmax(atom)
        
        for r in range(n_trials):
            for b in range(sparse_code.shape[2]):
                if sparse_code[r, i, b] > 0:
                    if max_idx > margin and max_idx < n_times_atom - margin:
                        extrema_candidates[r].append(("peak", b + max_idx, sparse_code[r, i, b]))
                    if min_idx > margin and min_idx < n_times_atom - margin:
                        extrema_candidates[r].append(("trough", b + min_idx, sparse_code[r, i, b]))

    if combine_atoms:
        for r in range(n_trials):
            sorted_candidates = sorted(extrema_candidates[r], key=lambda x: x[1])
    
            running = []
            for i, candidate in enumerate(sorted_candidates):
                if not running or running[0][0] == candidate[0]:
                    running.append(candidate)
                elif running[0][0] != candidate[0] or i == len(sorted_candidates) - 1:
                    if i == len(sorted_candidates) - 1:
                        running.append(candidate)

                    weighted_i = round(sum([t[1] * t[2] for t in running]) / sum([t[2] for t in running]))

                    if running[0][0] == "peak":
                        peak_array[r, 0, weighted_i] = True
                    else:
                        trough_array[r, 0, weighted_i] = True

                    if i < len(sorted_candidates) - 1:
                        running = [candidate]
    else:
        for r in range(n_trials):
            for candidate in extrema_candidates[r]:
                if candidate[0] == "peak":
                    peak_array[r, 0, candidate[1]] = True
                else:
                    trough_array[r, 0, candidate[1]] = True

    return peak_array, trough_array




def plot_event_extrema(X_all, trial_idx, peak_array, trough_array):
    """
    Plots a selected trial from the input data and marks peaks and troughs identified by the find_extrema function.

    Parameters
    ----------
    X_all : ndarray
        A three-dimensional array representing the input data. The first dimension represents the trial index,
        the second dimension represents the channel index, and the third dimension represents the time index.
    trial_idx : int
        The index of the trial to be plotted.
    peak_array : ndarray
        A three-dimensional boolean array of the same dimensions as the input data, where True indicates the presence
        of a peak. Generated by the find_extrema function.
    trough_array : ndarray
        A three-dimensional boolean array of the same dimensions as the input data, where True indicates the presence
        of a trough. Generated by the find_extrema function.

    Returns
    -------
    None

    Example
    -------
    >>> X_all = np.random.rand(10, 1, 1000)
    >>> peak_array = np.zeros_like(X_all, dtype=bool)
    >>> trough_array = np.zeros_like(X_all, dtype=bool)
    >>> # Assuming peak_array and trough_array have been populated
    >>> plot_event_extrema(X_all, trial_idx=0, peak_array=peak_array, trough_array=trough_array)

    Notes
    -----
    - The function assumes that the data array `X_all` has trials along the first dimension, channels along the second dimension,
      and time points along the third dimension.
    - The peaks and troughs are marked with yellow and blue circles, respectively.
    """
    trial_data = X_all[trial_idx, 0, :]
    time_indices = np.arange(trial_data.shape[0])
    
    peak_indices = np.where(peak_array[trial_idx, 0, :])[0]
    trough_indices = np.where(trough_array[trial_idx, 0, :])[0]

    plt.figure(figsize=(12, 6))
    plt.plot(time_indices, trial_data, label="Trial data")
    
    plt.scatter(peak_indices, trial_data[peak_indices], marker="o", color="y", label="Peaks")
    plt.scatter(trough_indices, trial_data[trough_indices], marker="o", color="b", label="Troughs")
    
    plt.xlabel("Time index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Trial {trial_idx} with Peaks and Troughs")
    plt.show()
    

#%% Main

if __name__ == "__main__":
    
    import scipy.io as spio
    import alphacsc.utils as utils
    from alphacsc import BatchCDL
    import data_alignment as da
    
    data = spio.loadmat('data/full_trials_allsubjs_troughdata.mat')
    all_data = data['all_data']
    all_trough_data = data['all_trough_data']
    
    n_troughs = da.count_troughs(all_trough_data)
    n_times = [350] # make sure is even

    X_all, YorN_all, t_all = da.assemble_trough_aligned_data(all_data, 
                                                             all_trough_data, 
                                                             n_times[0])

    # X_all = X_all32
    X_all = utils.check_univariate_signal(X_all)
    
    sfreq = 600.

    # Define the shape of the dictionary
    n_atoms = 6
    n_times_atom = 20 # 100 # 1000

    cdl = BatchCDL(
        # Shape of the dictionary
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        # Request a rank1 dictionary with unit norm temporal and spatial maps
        rank1=True, uv_constraint='separate',
        # Initialize the dictionary with random chunk from the data
        D_init= 'chunk', # random | chunk
        # rescale the regularization parameter to be 20% of lambda_max
        lmbd_max="scaled", reg=.2,
        # Number of iteration for the alternate minimization and cvg threshold
        n_iter=100, eps=1e-6,
        # solver for the z-step
        solver_z="lgcd", solver_z_kwargs={'tol': 1e-2, 'max_iter': 1000},
        # solver for the d-step
        solver_d='auto', solver_d_kwargs={'max_iter': 300},
        # sort
        sort_atoms=True,
        # Technical parameters
        verbose=1, random_state=0, n_jobs=4)

    cdl.fit(X_all)
    
    #%%
    
    labels = classify_cycle_components(cdl, variance_threshold=0.01)
    plot_atoms(cdl,sfreq, labels_on=True)
    # plot_event_coverage(cdl)
    # plot_atom_scatter(cdl, YorN_all, sfreq, scale_factor=20)
    plot_extrema_scatter(cdl, sfreq, YorN_all, scale_factor=20)
    plot_extrema_cpdf(cdl)
    # plot_atom_comparison(cdl, X_all, YorN_all, sfreq)
    plot_aligned_extrema(cdl, X_all, YorN_all, sfreq, method='bootstrap', n_boot=10000, 
                         segment_win=(-100, 100), win_filter=None, alignx_by='trough', aligny_by = 'trough')    

    peaks, troughs = find_extrema(cdl, combine_atoms=True)
    plot_event_extrema(X_all, 7, peaks, troughs)
    
    