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
    target_path = os.path.join(abs_path, 'model.bin')

    if os.path.exists(target_path):
        # Test whether the model has changed and only compile if it did
        with open(target_path, 'rb') as f:
            current_model = pickle.load(f)
        with open(source_path, 'r') as f:
            file_content = "".join([line for line in f])
        if file_content != current_model.model_code or force:
            print(target_path, "[Compiling]", ["", "[Forced]"][force])
            
            # Prepare the data dictionary
            data = dict()
            data = {"n_p": M.shape[0],
                    "n_a": M.shape[1],
                    "M": M}
            
            # Read in source code
            with io.open(source_path, 'rt', encoding='utf-8') as f:
                    model_code = f.read()
                    
            # Build the posterior model
            model = stan.build(model_code, data = data, random_seed = 1)
            with open(target_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            print(target_path, "[Skipping --- already compiled]")
    else:
        # If model binary does not exist, compile it
        print(target_path, "[Compiling]")
           
        # Prepare the data dictionary
        data = dict()
        data = {"n_p": M.shape[0],
                "n_a": M.shape[1],
                "M": M}
        
        # Read in source code
        with io.open(source_path, 'rt', encoding='utf-8') as f:
                model_code = f.read()
                
        # Build the posterior model
        model = stan.build(model_code, data = data, random_seed = 1)
        with open(target_path, 'wb') as f:
            pickle.dump(model, f)

def load_model(M):
    """Load the model to memory."""
    compile_stan_model(M)
    with open(os.path.join(abs_path, "model.bin"), 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Sampling functions
# =============================================================================
def generate_sample(model, num_chains=4, warmup=5000, num_samples=500):
    """Run sampling for data matrix M."""    
    
    fit = model.sample(num_chains = 4, num_samples = warmup + num_samples)

    return fit


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
    n = len(samples['lp__']) // num_chains  # number of samples per chain
    log_probs = [samples['lp__'][i * n:(i + 1) * n] for i in range(num_chains)]
    log_probs_means = np.array([np.mean(lp) for lp in log_probs])
    return np.alltrue(log_probs_means - (1 - tol) * max(log_probs_means) > 0)


# =============================================================================
# Inference functions
# =============================================================================
def get_posterior_predictive_matrix(samples):
    """Calculate the posterior predictive matrix."""
    Q = samples['Q']
    C = samples['C']
    r = samples['r']
    ones = np.ones((len(samples['lp__']), Q.shape[1], Q.shape[2]))
    sigma_tau = np.einsum('ki,kj->kij', samples['sigma'], samples['tau'])
    accu = (1 - Q) * np.einsum('kij,k->kij', ones, C) * sigma_tau
    accu += Q * np.einsum('kij,k->kij', ones, C * (1 + r)) * sigma_tau
    return np.mean(accu, axis=0)


def estimate_network(samples):
    """Return the matrix of edge probabilities P(B_ij=1)."""
    return np.mean(samples['Q'], axis=0)


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
    values = np.zeros(len(samples['lp__']) * num_net)
    for i, Q in enumerate(samples['Q']):
        for j in range(num_net):
            B = np.random.binomial(n=1, p=Q)
            values[i * num_net + j] = property(B)
    return values

