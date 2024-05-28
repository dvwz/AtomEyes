#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:57:59 2023

@author: dwzhou
"""

import numpy as np
import AtomEyes as ae

def test_detect_overlapping_activations():
    # Define the atom length and create a sparse code with overlapping activations
    n_times_atom = 3
    sparse_code = np.array([
        # trial 1
        [
            [0, 1, 0, 0, 0, 0, 0],  # atom 1
            [0, 0, 1, 0, 0, 0, 0],  # atom 2
            [0, 0, 0, 0, 1, 0, 0]   # atom 3
        ],
        # trial 2
        [
            [0, 0, 1, 0, 0, 0, 0],  # atom 1
            [0, 1, 0, 0, 0, 0, 0],  # atom 2
            [0, 0, 0, 1, 0, 0, 0]   # atom 3
        ]
    ])

    # Call the function
    overlap_matrix = ae.detect_overlapping_activations(sparse_code, n_times_atom)

    # Ensure that the output has the right number of dimensions
    assert overlap_matrix.ndim == 3, f"Expected 3 dimensions, but got {overlap_matrix.ndim}"

    # Define the expected output
    expected_output = np.array([
        # trial 1
        [[False, True, True, False, True, False, False],
         [False, True, True, False, True, False, False],
         [False, True, True, False, True, False, False]],
        # trial 2
        [[False, True, True, True, False, False, False],
         [False, True, True, True, False, False, False],
         [False, True, True, True, False, False, False]]
    ])

    # Check if the output matches the expected result
    assert np.array_equal(overlap_matrix, expected_output), f"Expected {expected_output}, but got {overlap_matrix}"

test_detect_overlapping_activations()


test_detect_overlapping_activations()
