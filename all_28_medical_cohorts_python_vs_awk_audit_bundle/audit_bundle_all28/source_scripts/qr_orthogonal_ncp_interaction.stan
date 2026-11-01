// =============================================================================
// File: qr_orthogonal_ncp_interaction.stan
// Stan Reference Implementation of Householder QR-Orthogonal NCP + Pull-Back
// =============================================================================
data {
  int<lower=4> N;
  int<lower=1> M;
  array[N] int<lower=1, upper=M> group_id;
  vector[N] x1;
  vector[N] x2;
  vector[N] y;
  real<lower=0> prior_sd_fixed;
  real<lower=0> lkj_eta;
}
transformed data {
  real mu1 = mean(x1);
  real mu2 = mean(x2);
  real s1  = fmax(sd(x1) * sqrt((N - 1.0) / N), 1e-15);
  real s2  = fmax(sd(x2) * sqrt((N - 1.0) / N), 1e-15);
  vector[N] u1 = (x1 - mu1) / s1;
  vector[N] u2 = (x2 - mu2) / s2;
  vector[N] raw_u12 = u1 .* u2;
  matrix[N, 3] M_main = append_col(append_col(rep_vector(1.0, N), u1), u2);
  matrix[N, 3] Q_m = qr_thin_Q(M_main);
  matrix[3, 3] R_m = qr_thin_R(M_main);
  vector[3] pi_orth = mdivide_right_tri_low((Q_m' * raw_u12)', R_m')';
  vector[N] r_orth = raw_u12 - M_main * pi_orth;
  real s_orth = fmax(sqrt(dot_self(r_orth) / N), 1e-15);
  vector[N] i_orth = r_orth / s_orth;
  matrix[N, 4] Z_orth = append_col(M_main, i_orth);
  vector[4] scale_clin_marg = [1.0, 1.0 / s1, 1.0 / s2, 1.0 / (s1 * s2 * s_orth)]';
}
parameters {
  vector[4] gamma_pop;
  real<lower=1e-6> sigma_eps;
  vector<lower=1e-6>[2] sigma_rand_orth;
  cholesky_factor_corr[2] L_Omega_orth;
  matrix[2, M] z_rand_ncp;
}
transformed parameters {
  matrix[2, M] b_rand_orth = diag_pre_multiply(sigma_rand_orth, L_Omega_orth) * z_rand_ncp;
  vector[N] eta = Z_orth * gamma_pop;
  if (M > 1) {
    for (n in 1:N) eta[n] += b_rand_orth[1, group_id[n]] + b_rand_orth[2, group_id[n]] * u1[n];
  }
}
model {
  gamma_pop ~ normal(0.0, prior_sd_fixed);
  sigma_eps ~ exponential(1.0);
  sigma_rand_orth ~ normal(0.0, 1.5);
  L_Omega_orth ~ lkj_corr_cholesky(lkj_eta);
  to_vector(z_rand_ncp) ~ std_normal();
  y ~ normal(eta, sigma_eps);
}
generated quantities {
  vector[4] beta_clin_marg = scale_clin_marg .* gamma_pop;
}
