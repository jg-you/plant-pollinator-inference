#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python functions to interact with the Stan model.

Author:
Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
Sabine Dritz <sjdritz@ucdavis.edu>
"""
import numpy as np
import pickle
import stan
import nest_asyncio
nest_asyncio.apply()
del nest_asyncio
import os
import io

abs_path = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Model functions
# =============================================================================
def compile_stan_model(M):
    """Compile Stan model."""
    source_path = os.path.join(abs_path, 'model.stan')

    # Read in source code
    with io.open(source_path, 'r', encoding='utf-8') as f:
            model_code = f.read()

    # Prepare the data dictionary
    data = dict()
    data = {"n_p": M.shape[0],
            "n_a": M.shape[1],
            "M": M}
    model = stan.build(model_code, data = data)
    return model

# =============================================================================
# Sampling functions
# =============================================================================
def generate_sample(model, num_chains=4, warmup=5000, num_samples=500):
    """Run sampling for data matrix M."""    
    
    samples = model.sample(num_chains = num_chains, 
                           num_samples = num_samples, 
                           num_warmup = warmup,
                           max_depth = 15)

    return samples


def save_samples(samples, fpath='samples.bin'):
    """Save samples as binaries, with pickle.

    Warning
    -------
    To retrieve this data, one has to load *the exact version of the model*
    used to generate the samples in memory. Hence, re-compiling the model will
    make the data inaccessible.
    """
    with open(fpath, 'wb') as f:
        pickle.dump(samples, f)


def load_samples(fpath='samples.bin'):
    """Load samples from binaries, with pickle.

    Warning
    -------
    Must have loaded *the same version of the model* in memory.
    """
    with open(fpath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Inference functions
# =============================================================================
def get_posterior_predictive_matrix(samples):
    """Calculate the posterior predictive matrix."""
    Q = samples['Q'][0]
    C = samples['C'][0]
    r = samples['r'][0]
    ones = np.ones((len(samples['lp__'][0]), Q.shape[1], Q.shape[2]))
    sigma_tau = np.einsum('ki,kj->kij', samples['sigma'][0], samples['tau'][0])
    accu = (1 - Q) * np.einsum('kij,k->kij', ones, C) * sigma_tau
    accu += Q * np.einsum('kij,k->kij', ones, C * (1 + r)) * sigma_tau
    return np.mean(accu, axis=0)


def estimate_network(samples):
    """Return the matrix of edge probabilities P(B_ij=1)."""
    return np.mean(samples['Q'], axis=-1)


def get_network_property_distribution(samples, property, num_net=10):
    """Return the average posterior value of an arbitrary network property.

    Input
    -----
    samples: StanFit object
        The posterior samples.
    property: function
        This function should take an incidence matrix as input and return a
        scalar.
    num_net: int
        Number of networks to generate for each parameter samples.
    """
    values = np.zeros(len(samples['lp__'][0]) * num_net)
    for i, Q in enumerate(samples['Q'][0]):
        for j in range(num_net):
            B = np.random.binomial(n=1, p=Q)
            values[i * num_net + j] = property(B)
    return values


