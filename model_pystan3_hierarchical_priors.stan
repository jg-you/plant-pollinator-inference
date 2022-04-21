data {
  // Dimensions of the data matrix, and matrix itself.
  int<lower=1> n_p;
  int<lower=1> n_a;
  array[n_p, n_a] int<lower=0> M;
}
transformed data {
  // Pre-compute the marginals of M to save computation in the model loop.
  array[n_p] int M_rows = rep_array(0, n_p);
  array[n_a] int M_cols = rep_array(0, n_a);
  int M_tot = 0;
  for (i in 1:n_p) {
    for (j in 1:n_a) {
      M_rows[i] += M[i, j];
      M_cols[j] += M[i, j];
      M_tot += M[i, j];
    }
  }
}
parameters {
  real<lower=0> C;
  real<lower=0> r;
  simplex[n_p] sigma;
  simplex[n_a] tau;
  real<lower=0, upper=1> rho;
}
model {
  // Prior
  r ~ exponential(0.01);

  // Global sums and parameters
  target += M_tot * log(C) - C;
  // Weighted marginals of the data matrix 
  for (i in 1:n_p) {
    target += M_rows[i] * log(sigma[i]);
  }
  for (j in 1:n_a) {
    target += M_cols[j] * log(tau[j]);
  }
  // Pairwise loop
  for (i in 1:n_p) {
    for (j in 1:n_a) {
      real nu_ij_0 = log(1 - rho);
      real nu_ij_1 = log(rho) + M[i,j] * log(1 + r) - C * r * sigma[i] * tau[j];
      if (nu_ij_0 > nu_ij_1)
        target += nu_ij_0 + log1p_exp(nu_ij_1 - nu_ij_0);
      else
        target += nu_ij_1 + log1p_exp(nu_ij_0 - nu_ij_1);
    }
  }
} 
generated quantities {
  // Posterior edge probability matrix
  array[n_p, n_a] real<lower=0> Q;
  for (i in 1:n_p) {
    for (j in 1:n_a) {
      real nu_ij_0 = log(1 - rho);
      real nu_ij_1 = log(rho) + M[i,j] * log(1 + r) - C * r * sigma[i] * tau[j];
      if (nu_ij_1 > 0) 
        Q[i, j] = 1 / (1+ exp(nu_ij_0 - nu_ij_1));
      else
        Q[i, j] = exp(nu_ij_1) / (exp(nu_ij_0) + exp(nu_ij_1));
    }
  }
}
