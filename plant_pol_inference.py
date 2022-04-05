#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python functions to interact with the Stan model.

Author:
Jean-Gabriel Young <jgyou@umich.edu>
"""
import random
import numpy as np
import pickle
import stan
import os
import io

abs_path = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Model functions
# =============================================================================
def compile_stan_model(M, force=False):
    """Autocompile Stan model."""
    source_path = os.path.join(abs_path, 'model.stan')

    print("[Compiling]")
           
    # Prepare the data dictionary
    data = dict()
    data = {"n_p": M.shape[0],
            "n_a": M.shape[1],
            "M": M}

    # Read in source code
    with io.open(source_path, 'rt', encoding='utf-8') as f:
            model_code = f.read()

    # Build the posterior model
    model = stan.build(model_code, data = data, random_seed = random.randrange(0, 100))
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


def test_samples(samples, tol=0.1, num_chains=4):
    """Verify that no chain has a markedly lower average log-probability."""

    log_probs = samples['lp__'][0]
    n = len(log_probs) // num_chains  # number of samples per chain
    log_probs = [log_probs[list(range(i, n - (num_chains - i), num_chains))] for i in range(num_chains)]
    log_probs_means = np.array([np.mean(lp) for lp in log_probs])
    print(log_probs_means)
    return np.alltrue(log_probs_means - (1 - tol) * max(log_probs_means) > 0)


# =============================================================================
# Inference functions
# =============================================================================
def get_posterior_predictive_matrix(samples):
    """Calculate the posterior predictive matrix."""
    Q = samples['Q'].transpose((2, 0, 1))
    C = samples['C'][0]
    r = samples['r'][0]
    ones = np.ones((Q.shape[0], Q.shape[1], Q.shape[2]))
    sigma_tau = np.einsum('ki,kj->kij', samples['sigma'].transpose(), samples['tau'].transpose())
    accu = (1 - Q) * np.einsum('kij,k->kij', ones, C) * sigma_tau
    accu += Q * np.einsum('kij,k->kij', ones, C * (1 + r)) * sigma_tau
    return np.mean(accu, axis=0)


def estimate_network(samples):
    """Return the matrix of edge probabilities P(B_ij=1)."""
    Q = samples['Q'].transpose((2, 0, 1))
    return np.mean(Q, axis = 0)


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
    Q = samples['Q'].transpose((2, 0, 1))
    values = np.zeros(Q.shape[0] * num_net)
    for i, Q in enumerate(Q):
        for j in range(num_net):
            B = np.random.binomial(n=1, p=Q)
            values[i * num_net + j] = property(B)
    return values

