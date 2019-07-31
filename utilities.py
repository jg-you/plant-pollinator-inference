#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for manipulating plant--pollinator data matrices.

Author:
Jean-Gabriel Young <jgyou@umich.edu>
"""
import numpy as np


def contract(M):
    """Remove all empty rows and columns from a matrix."""
    mask_0 = ~np.all(M == 0, axis=0)
    M_prime = M[:, mask_0]
    mask_1 = ~np.all(M_prime == 0, axis=1)
    M_final = M_prime[mask_1, :]
    return M_final, mask_0, mask_1


def expand_1d_array(x, mask):
    """Re-expand a contracted 1D matrix."""
    y = np.zeros(mask.shape[0])
    ix = -1
    for i, nonemptyrow in enumerate(mask):
        if nonemptyrow:
            ix += 1
            y[i] = x[ix]
    return y


def expand_2d_array(x, mask_0, mask_1):
    """Re-expand a contracted 2D matrix."""
    y = np.zeros((mask_1.shape[0], mask_0.shape[0]))
    ix = -1
    for i, nonemptyrow in enumerate(mask_1):
        if nonemptyrow:
            ix += 1
            jx = -1
            for j, nonemptycol in enumerate(mask_0):
                if nonemptycol:
                    jx += 1
                    y[i, j] = x[ix, jx]
    return y


def sort(A, B, reverse_X=True, reverse_Y=True):
    """Sort array A based on the margins of identically sized array B."""
    margin0 = B.sum(axis=0)
    margin1 = B.sum(axis=1)
    if reverse_X:
        argsort0 = np.argsort(-margin0)
    else:
        argsort0 = np.argsort(margin0)
    if reverse_Y:
        argsort1 = np.argsort(margin1)
    else:
        argsort1 = np.argsort(-margin1)
    return A[argsort1, :][:, argsort0]
