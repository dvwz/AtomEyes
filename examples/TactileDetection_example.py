#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np

import AtomEyes as ae
import data_alignment as da

import alphacsc.utils as utils
from alphacsc import BatchCDL

import os

#%% Load data
"""
Load the dataset from a .mat file. This file contains the data for all subjects and troughs.
"""

data = spio.loadmat('../data/full_trials_allsubjs_troughdata.mat') 

all_data = data['all_data']
all_trough_data = data['all_trough_data']

#%% Assemble data
"""
Preprocess the data by counting troughs, normalizing, and detrending. Assemble the data for analysis.
"""

n_troughs = da.count_troughs(all_trough_data)
n_times = [200] # make sure is even

normalize = False

all_data = da.preprocess_data(all_data, 
                              offset=True, detrend=False, normalize=normalize)

X_all, YorN_all, t_all = da.assemble_trough_aligned_data(all_data, 
                                                         all_trough_data, 
                                                         n_times[0],
                                                         detrend=True,
                                                         normalize=normalize)

X_all = utils.check_univariate_signal(X_all)

#%% Plot individual traces
"""
Plot every time series with overlap and low alpha. Save the plots as PNG and PDF files.
"""

time = np.arange(X_all.shape[2]) * (1000.0 / 600)  # Convert sample indices to time (ms)

n_trials = X_all.shape[0]
# Plot every time series with overlap and low alpha
fig, ax = plt.subplots()
for trial in range(n_trials):
    ax.plot(time, X_all[trial, 0, :], alpha=0.3)

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
plt.show()    

save_dir = "figures/nb"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, "sample_events.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "sample_events.pdf"), dpi=300)

#%% Plot mean waveform and SEM
"""
Plot the mean and standard error of the mean (SEM) for detected and non-detected trials.
"""

def plot_mean_and_sem(X_all, YorN_all, sfreq):
    plt.figure()
    unique_classes = np.unique(YorN_all)

    class_mapping = {0: "Non-Detected", 1: "Detected"}  # mapping of class to string
    time = np.arange(X_all.shape[2]) * (1000.0 / sfreq)  # Convert sample indices to time (ms)

    for cls in unique_classes:
        class_indices = np.where(YorN_all == cls)[0]
        class_data = np.squeeze(X_all[class_indices, 0, :])

        mean_data = np.mean(class_data, axis=0)
        sem_data = np.std(class_data, axis=0) / np.sqrt(class_data.shape[0])

        # Use the class string in the label
        plt.plot(time, mean_data, label=f"{class_mapping[cls]} Mean")
        plt.fill_between(time, mean_data - sem_data, mean_data + sem_data, alpha=0.5)
        
    plt.xlabel("Time (ms)")
    plt.ylabel("Dipole (nAm)")

    # Improved title
    plt.title("Mean Beta Events for Detected and Non-Detected Trials")

    plt.legend()
    plt.show()


# Example usage
plot_mean_and_sem(X_all, YorN_all, 600)

plt.savefig(os.path.join(save_dir, "minaligned_mean.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "minaligned_mean.pdf"), dpi=300)

#%% Perform rank-1 convolutional dictionary learning
"""
Perform rank-1 convolutional dictionary learning with 4 atoms of 32 samples each.
"""
sfreq = 600.

# Define the shape of the dictionary
n_atoms = 4
n_times_atom = 32

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
    verbose=1, random_state=0, n_jobs=8)

cdl.fit(X_all)

#%% Plot sparse code and reconstruction
"""
Plot the sparse code and reconstruction of the CDL model.
"""

ae.plot_sparse_code_and_reconstruction(cdl, X_all, 9)

plt.savefig(os.path.join(save_dir, "reconstruction.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "reconstruction.pdf"), dpi=300)

#%% Plot atoms
"""
Plot the temporal patterns of the atoms learned by the CDL model.
"""

ae.plot_atoms(cdl,sfreq, labels_on=False)

plt.savefig(os.path.join(save_dir, "atoms.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "atoms.pdf"), dpi=300)

#%% Plot event coverage
"""
Plot the event coverage of each atom in the CDL model.
"""

ae.plot_event_coverage(cdl)

plt.savefig(os.path.join(save_dir, "temporal_coverage.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "temporal_coverage.pdf"), dpi=300)

#%% Plot atom scatter
"""
Plot a scatter plot of atom activations across trials.
"""

if normalize == True: scale_factor=20
elif normalize == False: scale_factor=1e9

ae.plot_atom_scatter(cdl, YorN_all, sfreq, scale_factor=scale_factor)

plt.savefig(os.path.join(save_dir, "atom_scatter.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "atom_scatter.pdf"), dpi=300)

#%% Find and plot extrema
"""
Find the peaks and troughs in the activations and plot the selected trial.
"""

peaks, troughs = ae.find_extrema(cdl, combine_atoms=True, choose_atoms=None)
ae.plot_event_extrema(X_all, 9, peaks, troughs)

plt.savefig(os.path.join(save_dir, "event_extrema.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "event_extrema.pdf"), dpi=300)

#%% Plot extrema scatter
"""
Plot a scatter plot of peaks and troughs found in the activations.
"""

ae.plot_extrema_scatter(cdl, sfreq, YorN_all, scale_factor=20)

plt.savefig(os.path.join(save_dir, "extrema_scatter.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "extrema_scatter.pdf"), dpi=300)

#%% Plot aligned extrema
"""
Align signal segments by extrema and plot the comparison between different trial classes.
"""

ae.plot_aligned_extrema(cdl, X_all, YorN_all, sfreq, 
                        method='bootstrap', n_boot=1000, 
                        choose_atoms=None,
                        segment_win=(-100, 100), win_filter=(150,175), alignx_by='trough', aligny_by = 'data_average') 

plt.savefig(os.path.join(save_dir, "troughaligned_mean.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "troughaligned_mean.pdf"), dpi=300)