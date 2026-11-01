#!/usr/bin/awk -f
# ==============================================================================
# strict_orthogonal_medical_audit.awk
# Exact Rational (Q) & Gram-Schmidt Orthogonalization Audit Engine in Pure AWK
# ==============================================================================
BEGIN {
    CONVFMT = "%.15g"
    OFMT    = "%.10g"
    if (col_x1 == "") col_x1 = 1
    if (col_x2 == "") col_x2 = 2
    if (col_y  == "") col_y  = 3
    if (header == "") header = 1
    if (prior_sd == "") prior_sd = 2.5
    if (sigma_eps == "") sigma_eps = 1.0
    n_obs = 0

    if (ARGC == 1) {
        run_builtin_exact_rational_benchmarks()
        exit 0
    }
}

NR == 1 && header == 1 {
    name_x1 = $(col_x1); name_x2 = $(col_x2); name_y = $(col_y)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", name_x1)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", name_x2)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", name_y)
    next
}

NF >= 3 {
    v1 = $(col_x1); v2 = $(col_x2); vy = $(col_y)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", v1)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", v2)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", vy)
    if (v1 == "" || v2 == "" || vy == "" || v1 == "NA" || v2 == "NA" || vy == "NA") next
    n_obs++
    X1[n_obs] = v1 + 0.0; X2[n_obs] = v2 + 0.0; Y[n_obs] = vy + 0.0
}

END {
    if (ARGC > 1 && n_obs >= 4) {
        if (name_x1 == "") name_x1 = "X1"
        if (name_x2 == "") name_x2 = "X2"
        if (name_y  == "") name_y  = "Y"
        execute_full_clinical_audit("Input_CSV_Cohort", n_obs, X1, X2, Y, sigma_eps, prior_sd, out_csv)
    }
}

function abs_val(x) { return (x < 0) ? -x : x }
function max_val(a, b) { return (a > b) ? a : b }
function min_val(a, b) { return (a < b) ? a : b }

function gcd(a, b,    t) {
    a = int(abs_val(a) + 0.5); b = int(abs_val(b) + 0.5)
    while (b != 0) { t = b; b = a % b; a = t }
    return (a == 0) ? 1 : a
}

function rat_set(arr_n, arr_d, key, num, den,    g, sgn) {
    if (den == 0) { printf("[ERROR] Rational division by zero at key=%s\n", key) > "/dev/stderr"; exit 1 }
    if (num == 0) { arr_n[key] = 0; arr_d[key] = 1; return }
    sgn = ((num < 0 && den > 0) || (num > 0 && den < 0)) ? -1 : 1
    num = int(abs_val(num) + 0.5); den = int(abs_val(den) + 0.5)
    g = gcd(num, den)
    arr_n[key] = sgn * int(num / g + 0.5); arr_d[key] = int(den / g + 0.5)
}

function rat_sub(An, Ad, kA, Bn, Bd, kB, Cn, Cd, kC,    n1, d1, n2, d2, g1) {
    n1 = An[kA]; d1 = Ad[kA]; n2 = -Bn[kB]; d2 = Bd[kB]; g1 = gcd(d1, d2)
    rat_set(Cn, Cd, kC, n1 * int(d2 / g1 + 0.5) + n2 * int(d1 / g1 + 0.5), int(d1 / g1 + 0.5) * d2)
}

function rat_mul(An, Ad, kA, Bn, Bd, kB, Cn, Cd, kC,    n1, d1, n2, d2, g1, g2) {
    n1 = An[kA]; d1 = Ad[kA]; n2 = Bn[kB]; d2 = Bd[kB]
    g1 = gcd(abs_val(n1), d2); g2 = gcd(abs_val(n2), d1)
    rat_set(Cn, Cd, kC, int(n1 / g1) * int(n2 / g2), int(d1 / g2 + 0.5) * int(d2 / g1 + 0.5))
}

function rat_div(An, Ad, kA, Bn, Bd, kB, Cn, Cd, kC,    T_n, T_d) {
    rat_set(T_n, T_d, "tmp", Bd[kB], Bn[kB])
    rat_mul(An, Ad, kA, T_n, T_d, "tmp", Cn, Cd, kC)
}

function rat_str(An, Ad, k) {
    return (Ad[k] == 1) ? sprintf("%d", An[k]) : sprintf("%d/%d", An[k], Ad[k])
}

function rat_flt(An, Ad, k) { return (An[k] + 0.0) / (Ad[k] + 0.0) }

function float_to_rat(val, max_den, arr_n, arr_d, key,    sgn, x, p0, q0, p1, q1, a, p2, q2, iter) {
    if (abs_val(val) < 1e-15) { rat_set(arr_n, arr_d, key, 0, 1); return }
    sgn = (val < 0) ? -1 : 1; x = abs_val(val)
    p0 = 0; q0 = 1; p1 = 1; q1 = 0
    for (iter = 1; iter <= 40; iter++) {
        a = int(x); p2 = a * p1 + p0; q2 = a * q1 + q0
        if (q2 > max_den) break
        p0 = p1; q0 = q1; p1 = p2; q1 = q2
        if (abs_val(x - a) < 1e-12) break
        x = 1.0 / (x - a)
    }
    rat_set(arr_n, arr_d, key, sgn * p1, q1)
}

function solve_exact_q_system(p, Gn, Gd, bn, bd, xn, xd,    i, j, k, piv, mx_piv, tv, tn, td, fn, fd, pr_n, pr_d, An, Ad) {
    for (i = 1; i <= p; i++) {
        for (j = 1; j <= p; j++) rat_set(An, Ad, i "," j, Gn[i "," j], Gd[i "," j])
        rat_set(An, Ad, i "," (p + 1), bn[i], bd[i])
    }
    for (k = 1; k <= p; k++) {
        piv = k; mx_piv = abs_val(rat_flt(An, Ad, k "," k))
        for (i = k + 1; i <= p; i++) {
            tv = abs_val(rat_flt(An, Ad, i "," k))
            if (tv > mx_piv) { mx_piv = tv; piv = i }
        }
        if (piv != k) {
            for (j = k; j <= p + 1; j++) {
                tn = An[k "," j]; td = Ad[k "," j]
                An[k "," j] = An[piv "," j]; Ad[k "," j] = Ad[piv "," j]
                An[piv "," j] = tn; Ad[piv "," j] = td
            }
        }
        rat_set(fn, fd, "piv", An[k "," k], Ad[k "," k])
        for (j = k; j <= p + 1; j++) rat_div(An, Ad, k "," j, fn, fd, "piv", An, Ad, k "," j)
        for (i = 1; i <= p; i++) {
            if (i != k && An[i "," k] != 0) {
                rat_set(fn, fd, "fac", An[i "," k], Ad[i "," k])
                for (j = k; j <= p + 1; j++) {
                    rat_mul(fn, fd, "fac", An, Ad, k "," j, pr_n, pr_d, "prod")
                    rat_sub(An, Ad, i "," j, pr_n, pr_d, "prod", An, Ad, i "," j)
                }
            }
        }
    }
    for (i = 1; i <= p; i++) rat_set(xn, xd, i, An[i "," (p + 1)], Ad[i "," (p + 1)])
}

function solve_and_invert_float(p, G, b, sol, Ginv,    i, j, k, piv, mx_piv, tmp, fac, M) {
    for (i = 1; i <= p; i++) {
        for (j = 1; j <= p; j++) { M[i, j] = G[i, j]; M[i, p + j] = (i == j) ? 1.0 : 0.0 }
        M[i, 2 * p + 1] = b[i]
    }
    for (k = 1; k <= p; k++) {
        piv = k; mx_piv = abs_val(M[k, k])
        for (i = k + 1; i <= p; i++) if (abs_val(M[i, k]) > mx_piv) { mx_piv = abs_val(M[i, k]); piv = i }
        if (piv != k) for (j = 1; j <= 2 * p + 1; j++) { tmp = M[k, j]; M[k, j] = M[piv, j]; M[piv, j] = tmp }
        fac = M[k, k]; if (abs_val(fac) < 1e-30) fac = 1e-30
        for (j = 1; j <= 2 * p + 1; j++) M[k, j] /= fac
        for (i = 1; i <= p; i++) if (i != k) {
            fac = M[i, k]
            for (j = 1; j <= 2 * p + 1; j++) M[i, j] -= fac * M[k, j]
        }
    }
    for (i = 1; i <= p; i++) {
        sol[i] = M[i, 2 * p + 1]
        for (j = 1; j <= p; j++) Ginv[i, j] = M[i, p + j]
    }
}

function execute_full_clinical_audit(cohort_name, n, x1, x2, y, seps, tau0, csv_out_path,
    i, j, k, mu1, mu2, s1, s2, tau_m1, tau_m2, tau_s, u1, u2, w_raw,
    pi0, pi1, pi2, r_orth, s_orth, i_orth, x3_raw, mu3_raw, s3_raw, z3_post,
    Z_orth, X_raw, G_orth, rhs_orth, G_m, rhs_m, m_row, pi_vec, G_m_inv,
    G_orth_reg, gamma_orth, cov_orth, beta_raw, rss_orth, dof, sigma2_hat, pred_i,
    scale_clin, beta_prop, se_prop, t_prop, se_raw, t_raw, beta_1_poly, beta_2_poly, beta_3_poly,
    Gn_q, Gd_q, bn_q, bd_q, xn_q, xd_q, beta_q_clin, q_rel_err, y_mean, lam0) {

    mu1 = 0.0; mu2 = 0.0; y_mean = 0.0
    for (i = 1; i <= n; i++) { mu1 += x1[i]; mu2 += x2[i]; y_mean += y[i] }
    mu1 /= n; mu2 /= n; y_mean /= n
    s1 = 0.0; s2 = 0.0
    for (i = 1; i <= n; i++) { s1 += (x1[i] - mu1)^2; s2 += (x2[i] - mu2)^2 }
    s1 = sqrt(max_val(s1 / n, 1e-30)); s2 = sqrt(max_val(s2 / n, 1e-30))
    tau_m1 = abs_val(mu1 / s1); tau_m2 = abs_val(mu2 / s2); tau_s = max_val(s1 / s2, s2 / s1)

    for (i = 1; i <= n; i++) { u1[i] = (x1[i] - mu1) / s1; u2[i] = (x2[i] - mu2) / s2; w_raw[i] = u1[i] * u2[i] }
    for (j = 1; j <= 3; j++) { rhs_m[j] = 0.0; for (k = 1; k <= 3; k++) G_m[j, k] = 0.0 }
    for (i = 1; i <= n; i++) {
        m_row[1] = 1.0; m_row[2] = u1[i]; m_row[3] = u2[i]
        for (j = 1; j <= 3; j++) {
            rhs_m[j] += m_row[j] * w_raw[i]
            for (k = 1; k <= 3; k++) G_m[j, k] += m_row[j] * m_row[k]
        }
    }
    solve_and_invert_float(3, G_m, rhs_m, pi_vec, G_m_inv)
    pi0 = pi_vec[1]; pi1 = pi_vec[2]; pi2 = pi_vec[3]

    s_orth = 0.0
    for (i = 1; i <= n; i++) { r_orth[i] = w_raw[i] - (pi0 + pi1 * u1[i] + pi2 * u2[i]); s_orth += r_orth[i]^2 }
    s_orth = sqrt(max_val(s_orth / n, 1e-30))
    mu3_raw = 0.0
    for (i = 1; i <= n; i++) { i_orth[i] = r_orth[i] / s_orth; x3_raw[i] = x1[i] * x2[i]; mu3_raw += x3_raw[i] }
    mu3_raw /= n
    s3_raw = 0.0
    for (i = 1; i <= n; i++) s3_raw += (x3_raw[i] - mu3_raw)^2
    s3_raw = sqrt(max_val(s3_raw / n, 1e-30))

    for (i = 1; i <= n; i++) {
        z3_post[i] = (x3_raw[i] - mu3_raw) / s3_raw
        Z_orth[i, 1] = 1.0; Z_orth[i, 2] = u1[i]; Z_orth[i, 3] = u2[i]; Z_orth[i, 4] = i_orth[i]
    }
    lam0 = (seps > 0 && tau0 > 0) ? (seps / tau0)^2 : 0.0
    for (j = 1; j <= 4; j++) { rhs_orth[j] = 0.0; for (k = 1; k <= 4; k++) G_orth[j, k] = 0.0 }
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= 4; j++) {
            rhs_orth[j] += Z_orth[i, j] * y[i]
            for (k = 1; k <= 4; k++) G_orth[j, k] += Z_orth[i, j] * Z_orth[i, k]
        }
    }
    for (j = 1; j <= 4; j++) {
        float_to_rat(rhs_orth[j], 100000, bn_q, bd_q, j)
        for (k = 1; k <= 4; k++) float_to_rat(G_orth[j, k] + ((j == k) ? lam0 : 0.0), 100000, Gn_q, Gd_q, j "," k)
    }
    solve_exact_q_system(4, Gn_q, Gd_q, bn_q, bd_q, xn_q, xd_q)

    for (j = 1; j <= 4; j++) {
        for (k = 1; k <= 4; k++) G_orth_reg[j, k] = G_orth[j, k] + ((j == k) ? lam0 : 0.0)
    }
    solve_and_invert_float(4, G_orth_reg, rhs_orth, gamma_orth, cov_orth)

    scale_clin[1] = 1.0; scale_clin[2] = 1.0 / s1; scale_clin[3] = 1.0 / s2; scale_clin[4] = 1.0 / (s1 * s2 * s_orth)
    dof = max_val(n - 4, 1); rss_orth = 0.0
    for (i = 1; i <= n; i++) {
        pred_i = 0.0
        for (j = 1; j <= 4; j++) pred_i += Z_orth[i, j] * gamma_orth[j]
        rss_orth += (y[i] - pred_i)^2
    }
    sigma2_hat = rss_orth / dof
    for (j = 1; j <= 4; j++) {
        beta_prop[j]   = gamma_orth[j] * scale_clin[j]
        beta_q_clin[j] = rat_flt(xn_q, xd_q, j) * scale_clin[j]
        se_prop[j]     = sqrt(max_val(cov_orth[j, j] * sigma2_hat, 1e-30)) * scale_clin[j]
        t_prop[j]      = beta_prop[j] / se_prop[j]
    }
    q_rel_err = abs_val(beta_prop[2] - beta_q_clin[2]) / max_val(abs_val(beta_q_clin[2]), 1e-15)
    beta_1_poly = (gamma_orth[2] - pi1 * gamma_orth[4] / s_orth) / s1
    beta_3_poly = gamma_orth[4] * scale_clin[4]
    beta_raw[2] = beta_1_poly - mu2 * beta_3_poly
    se_raw[2]   = sqrt(se_prop[2]^2 + (mu2^2) * (se_prop[4]^2))
    t_raw[2]    = beta_raw[2] / max_val(se_raw[2], 1e-30)

    printf("* [%s (N=%d)] Exact Q B1=%+.6f | Prop B1=%+.6f (t=%+.2f, Q_RelErr=%.2e) | Raw B1=%+.6f (t=%+.2f)\n",
           cohort_name, n, beta_q_clin[2], beta_prop[2], t_prop[2], q_rel_err, beta_raw[2], t_raw[2])
    if (csv_out_path != "") {
        printf("u1_orth,u2_orth,i_orth,y,scale_b0,scale_b1,scale_b2,scale_b3\n") > csv_out_path
        for (i = 1; i <= n; i++) {
            printf("%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                   u1[i], u2[i], i_orth[i], y[i], scale_clin[1], scale_clin[2], scale_clin[3], scale_clin[4]) >> csv_out_path
        }
        close(csv_out_path)
    }
}

function run_builtin_exact_rational_benchmarks(    j, k, G4n, G4d, b4n, b4d, sol4_simple_n, sol4_simple_d, sol4_orth_n, sol4_orth_d,
    GBn, GBd, bBn, bBd, mu_ref_n, mu_ref_d) {
    print "[AWK Script 1] Proposition 3 (4-Point Seasonal Q) & Proposition 18.1 (Conjugate Bayesian Q)"
    for (j = 1; j <= 4; j++) for (k = 1; k <= 4; k++) rat_set(G4n, G4d, j "," k, 0, 1)
    rat_set(G4n, G4d, "1,1", 4, 1); rat_set(G4n, G4d, "1,4", 2, 1)
    rat_set(G4n, G4d, "2,2", 5, 1); rat_set(G4n, G4d, "2,3", 2, 1)
    rat_set(G4n, G4d, "3,2", 2, 1); rat_set(G4n, G4d, "3,3", 4, 1)
    rat_set(G4n, G4d, "4,1", 2, 1); rat_set(G4n, G4d, "4,4", 5, 1)
    rat_set(b4n, b4d, 1, 40, 1); rat_set(b4n, b4d, 2, 16, 1); rat_set(b4n, b4d, 3, 16, 1); rat_set(b4n, b4d, 4, 36, 1)
    solve_exact_q_system(4, G4n, G4d, b4n, b4d, sol4_simple_n, sol4_simple_d)

    rat_set(G4n, G4d, "1,4", 0, 1); rat_set(G4n, G4d, "4,1", 0, 1); rat_set(G4n, G4d, "4,4", 4, 1)
    rat_set(b4n, b4d, 4, 16, 1)
    solve_exact_q_system(4, G4n, G4d, b4n, b4d, sol4_orth_n, sol4_orth_d)
    printf("  Prop 3 Simple Centered Q : (%s, %s, %s, %s) vs Proposed Orth Q : (%s, %s, %s, %s)\n",
           rat_str(sol4_simple_n, sol4_simple_d, 1), rat_str(sol4_simple_n, sol4_simple_d, 2),
           rat_str(sol4_simple_n, sol4_simple_d, 3), rat_str(sol4_simple_n, sol4_simple_d, 4),
           rat_str(sol4_orth_n, sol4_orth_d, 1), rat_str(sol4_orth_n, sol4_orth_d, 2),
           rat_str(sol4_orth_n, sol4_orth_d, 3), rat_str(sol4_orth_n, sol4_orth_d, 4))

    for (j = 1; j <= 4; j++) for (k = 1; k <= 4; k++) rat_set(GBn, GBd, j "," k, 0, 1)
    rat_set(GBn, GBd, "1,1", 201, 25); rat_set(GBn, GBd, "2,2", 251, 25)
    rat_set(GBn, GBd, "3,3", 201, 25); rat_set(GBn, GBd, "4,4", 251, 25)
    rat_set(bBn, bBd, 1, 16, 1); rat_set(bBn, bBd, 2, 15, 1)
    rat_set(bBn, bBd, 3, -8, 1); rat_set(bBn, bBd, 4,  8, 1)
    solve_exact_q_system(4, GBn, GBd, bBn, bBd, mu_ref_n, mu_ref_d)
    printf("  Prop 18.1 Conjugate Q    : beta_0=%s (%+.8f), beta_1=%s (%+.8f), beta_2=%s (%+.8f), beta_3=%s (%+.8f)\n",
           rat_str(mu_ref_n, mu_ref_d, 1), rat_flt(mu_ref_n, mu_ref_d, 1),
           rat_str(mu_ref_n, mu_ref_d, 2), rat_flt(mu_ref_n, mu_ref_d, 2),
           rat_str(mu_ref_n, mu_ref_d, 3), rat_flt(mu_ref_n, mu_ref_d, 3),
           rat_str(mu_ref_n, mu_ref_d, 4), rat_flt(mu_ref_n, mu_ref_d, 4))
}
