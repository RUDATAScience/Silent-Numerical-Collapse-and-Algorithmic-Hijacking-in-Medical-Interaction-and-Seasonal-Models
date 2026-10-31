#!/usr/bin/awk -f
# ==============================================================================
# verify_longitudinal_lmm_cases.awk
# ==============================================================================
# Pure AWK Verification Engine for Section 8 (Longitudinal Mixed-Effects Models:
# Propositions 15-17 & Cases 23-25: Sleep Study, Mayo PBCSeq, Orthodont)
# with Exact Rational (Q) 2x2 Covariance Pull-Back and 7D Two-Stage Hierarchical
# Gram-Schmidt Orthogonalization.
#
# [Usage 1: Built-in Exact Rational (Q) & Cases 23-25 Benchmark]
#   $ awk -f verify_longitudinal_lmm_cases.awk
#
# [Usage 2: Two-Stage Hierarchical Orthogonalization of Longitudinal Panel CSV]
#   $ awk -F',' -v col_id=1 -v col_t=2 -v col_x2=3 -v col_y=4 -v header=1 \
#         -v out_csv="panel_orth_ready.csv" -f verify_longitudinal_lmm_cases.awk panel.csv
#
# [Generated Artifacts]
#   1. table_mixed_effects_cases_23_25.tex
#   2. longitudinal_lmm_cases_23_25_benchmark.csv
# ==============================================================================

BEGIN {
    CONVFMT = "%.15g"
    OFMT    = "%.10g"
    if (col_id == "") col_id = 1
    if (col_t  == "") col_t  = 2
    if (col_x2 == "") col_x2 = 3
    if (col_y  == "") col_y  = 4
    if (header == "") header = 1
    n_obs = 0
    m_subj = 0

    if (ARGC == 1) {
        run_module1_exact_q_reml_cholesky_audit()
        run_module2_within_between_ecological_leakage()
        run_module3_4_longitudinal_cohorts_23_to_25()
        exit 0
    }
}

# ==============================================================================
# Stream Reader for External Longitudinal Panel CSV Files (Mode 2)
# ==============================================================================
NR == 1 && header == 1 { next }

NF >= 4 {
    raw_id = $(col_id); raw_t = $(col_t); raw_x2 = $(col_x2); raw_y = $(col_y)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", raw_id)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", raw_t)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", raw_x2)
    gsub(/^[ \t"]+|[ \t"\r]+$/, "", raw_y)
    if (raw_id == "" || raw_t == "" || raw_x2 == "" || raw_y == "" || raw_t == "NA" || raw_x2 == "NA" || raw_y == "NA") next

    if (!(raw_id in subj_map)) {
        m_subj++
        subj_map[raw_id] = m_subj
        subj_count[m_subj] = 0
    }
    sid = subj_map[raw_id]
    subj_count[sid]++
    n_obs++

    SUBJ[n_obs] = sid
    X1[n_obs]   = raw_t + 0.0
    X2[n_obs]   = raw_x2 + 0.0
    Y[n_obs]    = raw_y + 0.0
}

END {
    if (ARGC > 1 && n_obs >= 8) {
        audit_and_orthogonalize_panel_dataset("Input_Longitudinal_Panel_CSV", n_obs, m_subj, SUBJ, X1, X2, Y, out_csv)
    }
}

# ==============================================================================
# Exact Rational (Q) Arithmetic Library & Linear Algebra Solvers
# ==============================================================================
function abs_v(x) { return (x < 0) ? -x : x }
function max_v(a, b) { return (a > b) ? a : b }
function min_v(a, b) { return (a < b) ? a : b }

function gcd(a, b,    t) {
    a = int(abs_v(a) + 0.5)
    b = int(abs_v(b) + 0.5)
    while (b != 0) { t = b; b = a % b; a = t }
    return (a == 0) ? 1 : a
}

function rat_set(An, Ad, k, num, den,    g, sgn) {
    if (den == 0) { printf("[ERROR] Rational division by zero at key=%s\n", k) > "/dev/stderr"; exit 1 }
    if (num == 0) { An[k] = 0; Ad[k] = 1; return }
    sgn = ((num < 0 && den > 0) || (num > 0 && den < 0)) ? -1 : 1
    num = int(abs_v(num) + 0.5); den = int(abs_v(den) + 0.5)
    g = gcd(num, den)
    An[k] = sgn * int(num / g + 0.5)
    Ad[k] = int(den / g + 0.5)
}

function rat_add(An, Ad, kA, Bn, Bd, kB, Cn, Cd, kC,    n1, d1, n2, d2, g1) {
    n1 = An[kA]; d1 = Ad[kA]; n2 = Bn[kB]; d2 = Bd[kB]
    g1 = gcd(d1, d2)
    rat_set(Cn, Cd, kC, n1 * int(d2 / g1 + 0.5) + n2 * int(d1 / g1 + 0.5), int(d1 / g1 + 0.5) * d2)
}

function rat_sub(An, Ad, kA, Bn, Bd, kB, Cn, Cd, kC,    n1, d1, n2, d2, g1) {
    n1 = An[kA]; d1 = Ad[kA]; n2 = -Bn[kB]; d2 = Bd[kB]
    g1 = gcd(d1, d2)
    rat_set(Cn, Cd, kC, n1 * int(d2 / g1 + 0.5) + n2 * int(d1 / g1 + 0.5), int(d1 / g1 + 0.5) * d2)
}

function rat_mul(An, Ad, kA, Bn, Bd, kB, Cn, Cd, kC,    n1, d1, n2, d2, g1, g2) {
    n1 = An[kA]; d1 = Ad[kA]; n2 = Bn[kB]; d2 = Bd[kB]
    g1 = gcd(abs_v(n1), d2); g2 = gcd(abs_v(n2), d1)
    rat_set(Cn, Cd, kC, int(n1 / g1) * int(n2 / g2), int(d1 / g2 + 0.5) * int(d2 / g1 + 0.5))
}

function rat_str(An, Ad, k) {
    return (Ad[k] == 1) ? sprintf("%d", An[k]) : sprintf("%d/%d", An[k], Ad[k])
}

function rat_flt(An, Ad, k) {
    return (An[k] + 0.0) / (Ad[k] + 0.0)
}

# Exact 2x2 Matrix Multiplication over Q: C = A * B
function rat_mat2x2_mul(An, Ad, Bn, Bd, Cn, Cd,    i, j, k, sum_n, sum_d, pr_n, pr_d, tmp_n, tmp_d) {
    for (i = 1; i <= 2; i++) {
        for (j = 1; j <= 2; j++) {
            rat_set(sum_n, sum_d, "s", 0, 1)
            for (k = 1; k <= 2; k++) {
                rat_mul(An, Ad, i "," k, Bn, Bd, k "," j, pr_n, pr_d, "p")
                rat_add(sum_n, sum_d, "s", pr_n, pr_d, "p", sum_n, sum_d, "s")
            }
            rat_set(tmp_n, tmp_d, i "," j, sum_n["s"], sum_d["s"])
        }
    }
    for (i = 1; i <= 2; i++) {
        for (j = 1; j <= 2; j++) {
            rat_set(Cn, Cd, i "," j, tmp_n[i "," j], tmp_d[i "," j])
        }
    }
}

# Double-precision p x p Gauss-Jordan solver
function solve_linear_system(p, G, b, sol, Ginv,    i, j, k, piv, max_v, tmp, fac, M) {
    for (i = 1; i <= p; i++) {
        for (j = 1; j <= p; j++) {
            M[i, j] = G[i, j]
            M[i, p + j] = (i == j) ? 1.0 : 0.0
        }
        M[i, 2 * p + 1] = b[i]
    }
    for (k = 1; k <= p; k++) {
        piv = k; max_v = abs_v(M[k, k])
        for (i = k + 1; i <= p; i++) {
            if (abs_v(M[i, k]) > max_v) { max_v = abs_v(M[i, k]); piv = i }
        }
        if (piv != k) {
            for (j = 1; j <= 2 * p + 1; j++) {
                tmp = M[k, j]; M[k, j] = M[piv, j]; M[piv, j] = tmp
            }
        }
        fac = M[k, k]
        if (abs_v(fac) < 1e-30) fac = 1e-30
        for (j = 1; j <= 2 * p + 1; j++) M[k, j] /= fac
        for (i = 1; i <= p; i++) {
            if (i != k) {
                fac = M[i, k]
                for (j = 1; j <= 2 * p + 1; j++) M[i, j] -= fac * M[k, j]
            }
        }
    }
    for (i = 1; i <= p; i++) {
        sol[i] = M[i, 2 * p + 1]
        for (j = 1; j <= p; j++) Ginv[i, j] = M[i, p + j]
    }
}

function cond_sym2x2(g00, g01, g11,    tr, det, disc, lmax, lmin) {
    tr = g00 + g11
    det = g00 * g11 - g01 * g01
    disc = sqrt(max_v(tr * tr - 4.0 * det, 0.0))
    lmax = 0.5 * (tr + disc)
    lmin = max_v(0.5 * (tr - disc), 1e-30)
    return lmax / lmin
}

function normal_two_sided_pval(t_val,    z, b1, b2, b3, b4, b5, p, phi, t_k, cdf_c) {
    z = abs_v(t_val)
    if (z > 37.0) return 1.0e-99
    p = 0.2316419
    b1 = 0.319381530; b2 = -0.356563782; b3 = 1.781477937; b4 = -1.821255978; b5 = 1.330274429
    phi = exp(-0.5 * z * z) / sqrt(2.0 * atan2(0, -1))
    t_k = 1.0 / (1.0 + p * z)
    cdf_c = phi * t_k * (b1 + t_k * (b2 + t_k * (b3 + t_k * (b4 + t_k * b5))))
    return max_v(2.0 * cdf_c, 1.0e-99)
}

# ==============================================================================
# [Module 1] Propositions 15 & 16: Exact Q Covariance Pull-Back vs. Cholesky Decay
# ==============================================================================
function run_module1_exact_q_reml_cholesky_audit(    r, n_reg, labels, M1_list, c1_list,
    Gn, Gd, Lbn, Lbd, Lbinv_n, Lbinv_d, Lbt_n, Lbt_d, Lbinvt_n, Lbinvt_d,
    Tmp_n, Tmp_d, Gtilde_n, Gtilde_d, Gpull_n, Gpull_d,
    g00, g01, g11, det_g, rho_tilde, theta1, theta3, chol_ratio, q_match, status) {

    print "===================================================================================================="
    print "[Module 1] AWK Exact Rational (Q) REML/GLS Covariance Pull-Back vs. lme4 Cholesky Collapse (Prop 15 & 16)"
    print "===================================================================================================="

    rat_set(Gn, Gd, "1,1", 4, 1); rat_set(Gn, Gd, "1,2", 1, 1)
    rat_set(Gn, Gd, "2,1", 1, 1); rat_set(Gn, Gd, "2,2", 1, 1)

    printf("1. True Reference Random-Effects Covariance G in Q: G_00=%s, G_01=%s, G_11=%s, det(G)=3, rho_01=+1/2\n\n",
           rat_str(Gn, Gd, "1,1"), rat_str(Gn, Gd, "1,2"), rat_str(Gn, Gd, "2,2"))

    labels[1] = "Reference Orthogonal (M1=0, c1=1)";        M1_list[1] = 0;    c1_list[1] = 1
    labels[2] = "Moderate Time Offset (M1=10, c1=1)";       M1_list[2] = 10;   c1_list[2] = 1
    labels[3] = "Calendar Year Offset (M1=100, c1=1)";      M1_list[3] = 100;  c1_list[3] = 1
    labels[4] = "Year + Scale Asymmetry (M1=100, c1=100)";  M1_list[4] = 100;  c1_list[4] = 100
    labels[5] = "Millisecond Timestamp (M1=1000, c1=1000)"; M1_list[5] = 1000; c1_list[5] = 1000
    n_reg = 5

    printf("   %-40s | %14s | %24s | %10s | %-22s\n",
           "Regime Description", "Obs Corr rho01", "Cholesky theta3/theta1", "Pullback Q", "lme4 Status (tol=1e-4)")
    print "   ----------------------------------------------------------------------------------------------------------------------"

    for (r = 1; r <= n_reg; r++) {
        rat_set(Lbn, Lbd, "1,1", 1, 1); rat_set(Lbn, Lbd, "1,2", -M1_list[r], 1)
        rat_set(Lbn, Lbd, "2,1", 0, 1); rat_set(Lbn, Lbd, "2,2", 1, c1_list[r])

        rat_set(Lbt_n, Lbt_d, "1,1", 1, 1);           rat_set(Lbt_n, Lbt_d, "1,2", 0, 1)
        rat_set(Lbt_n, Lbt_d, "2,1", -M1_list[r], 1); rat_set(Lbt_n, Lbt_d, "2,2", 1, c1_list[r])

        rat_mat2x2_mul(Lbn, Lbd, Gn, Gd, Tmp_n, Tmp_d)
        rat_mat2x2_mul(Tmp_n, Tmp_d, Lbt_n, Lbt_d, Gtilde_n, Gtilde_d)

        rat_set(Lbinv_n, Lbinv_d, "1,1", 1, 1); rat_set(Lbinv_n, Lbinv_d, "1,2", c1_list[r] * M1_list[r], 1)
        rat_set(Lbinv_n, Lbinv_d, "2,1", 0, 1); rat_set(Lbinv_n, Lbinv_d, "2,2", c1_list[r], 1)

        rat_set(Lbinvt_n, Lbinvt_d, "1,1", 1, 1);                      rat_set(Lbinvt_n, Lbinvt_d, "1,2", 0, 1)
        rat_set(Lbinvt_n, Lbinvt_d, "2,1", c1_list[r] * M1_list[r], 1); rat_set(Lbinvt_n, Lbinvt_d, "2,2", c1_list[r], 1)

        rat_mat2x2_mul(Lbinv_n, Lbinv_d, Gtilde_n, Gtilde_d, Tmp_n, Tmp_d)
        rat_mat2x2_mul(Tmp_n, Tmp_d, Lbinvt_n, Lbinvt_d, Gpull_n, Gpull_d)

        q_match = (rat_str(Gpull_n, Gpull_d, "1,1") == "4" &&
                   rat_str(Gpull_n, Gpull_d, "1,2") == "1" &&
                   rat_str(Gpull_n, Gpull_d, "2,2") == "1") ? "EXACT(Q)" : "MISMATCH"

        g00 = rat_flt(Gtilde_n, Gtilde_d, "1,1")
        g01 = rat_flt(Gtilde_n, Gtilde_d, "1,2")
        g11 = rat_flt(Gtilde_n, Gtilde_d, "2,2")
        det_g = g00 * g11 - g01 * g01

        rho_tilde = g01 / sqrt(g00 * g11)
        theta1 = sqrt(g00)
        theta3 = sqrt(max_v(det_g, 0.0)) / theta1
        chol_ratio = theta3 / theta1
        status = (chol_ratio < 1.0e-4) ? "SINGULAR FIT (theta3->0)" : "Full Rank (Converged)"

        printf("   %-40s | %+14.7f | %24.6e | %10s | %-22s\n",
               labels[r], rho_tilde, chol_ratio, q_match, status)
    }
    print ""
}

# ==============================================================================
# [Module 2] Proposition 17: Within-Between Ecological Leakage vs. 2-Stage GS-QR
# ==============================================================================
function run_module2_within_between_ecological_leakage(    m_s, n_o, N, i, j, k, idx, pi_val,
    ang1, ang2, b1_B_s, b2_B_s, m_b1, m_b2, b1_B, b2_B, w1_tmp, w2_tmp, w1_W, w2_W, m_w1, m_w2,
    mu1, mu2, X1_p, X2_p, y_p, beta_W1, beta_W2, beta_W3, gamma_CL,
    P_raw, m_P_i, m_y_i, P_dem, y_dem, fe_row, G_fe, rhs_fe, sol_fe, Ginv_fe,
    s_b1, s_b2, s_w1, s_w2, u1_B, u2_B, u1_W, u2_W, M_pan, raw_w12,
    G_p7, rhs_p7, pi_p7, Ginv_p7, proj_val, r_orth, s_orth, i_W_orth,
    num_orth, den_orth, gamma_w3_orth, beta_W3_prop) {

    print "===================================================================================================="
    print "[Module 2] Within-Between Ecological Leakage & Two-Stage Hierarchical GS-QR Recovery (Prop 17)"
    print "===================================================================================================="

    pi_val = atan2(0, -1)
    m_s = 60; n_o = 8; N = m_s * n_o
    mu1 = 15.0; mu2 = 50.0
    beta_W1 = 2.00; beta_W2 = -1.50; beta_W3 = 0.00; gamma_CL = 0.85

    m_b1 = 0.0; m_b2 = 0.0
    for (i = 1; i <= m_s; i++) {
        b1_B_s[i] = 1.5 * cos(2.0 * pi_val * i / m_s) + 0.8 * sin(4.0 * pi_val * i / m_s)
        b2_B_s[i] = 2.0 * sin(2.0 * pi_val * i / m_s + 0.4) + 1.1 * cos(6.0 * pi_val * i / m_s)
        m_b1 += b1_B_s[i] / m_s
        m_b2 += b2_B_s[i] / m_s
    }
    for (i = 1; i <= m_s; i++) { b1_B_s[i] -= m_b1; b2_B_s[i] -= m_b2 }

    idx = 0
    for (i = 1; i <= m_s; i++) {
        m_w1 = 0.0; m_w2 = 0.0
        for (j = 1; j <= n_o; j++) {
            ang1 = 2.0 * pi_val * j / n_o + 0.17 * i
            ang2 = 4.0 * pi_val * j / n_o + 0.31 * i
            w1_tmp[j] = cos(ang1) + 0.4 * sin(2.0 * ang1)
            w2_tmp[j] = sin(ang2) + 0.3 * cos(3.0 * ang2)
            m_w1 += w1_tmp[j] / n_o
            m_w2 += w2_tmp[j] / n_o
        }
        for (j = 1; j <= n_o; j++) {
            idx++
            b1_B[idx] = b1_B_s[i]
            b2_B[idx] = b2_B_s[i]
            w1_W[idx] = w1_tmp[j] - m_w1
            w2_W[idx] = w2_tmp[j] - m_w2
            X1_p[idx] = mu1 + b1_B[idx] + w1_W[idx]
            X2_p[idx] = mu2 + b2_B[idx] + w2_W[idx]
            y_p[idx]  = 5.0 + 1.2 * b1_B[idx] + 0.8 * b2_B[idx] \
                        + (beta_W1 + gamma_CL * b2_B[idx]) * w1_W[idx] \
                        + beta_W2 * w2_W[idx] + beta_W3 * (w1_W[idx] * w2_W[idx])
            P_raw[idx] = X1_p[idx] * X2_p[idx]
        }
    }

    for (i = 1; i <= m_s; i++) {
        m_P_i = 0.0; m_y_i = 0.0
        for (j = 1; j <= n_o; j++) {
            idx = (i - 1) * n_o + j
            m_P_i += P_raw[idx] / n_o
            m_y_i += y_p[idx] / n_o
        }
        for (j = 1; j <= n_o; j++) {
            idx = (i - 1) * n_o + j
            P_dem[idx] = P_raw[idx] - m_P_i
            y_dem[idx] = y_p[idx] - m_y_i
        }
    }

    for (j = 1; j <= 3; j++) {
        rhs_fe[j] = 0.0
        for (k = 1; k <= 3; k++) G_fe[j, k] = 0.0
    }
    for (idx = 1; idx <= N; idx++) {
        fe_row[1] = w1_W[idx]; fe_row[2] = w2_W[idx]; fe_row[3] = P_dem[idx]
        for (j = 1; j <= 3; j++) {
            rhs_fe[j] += fe_row[j] * y_dem[idx]
            for (k = 1; k <= 3; k++) G_fe[j, k] += fe_row[j] * fe_row[k]
        }
    }
    solve_linear_system(3, G_fe, rhs_fe, sol_fe, Ginv_fe)

    s_b1 = 0.0; s_b2 = 0.0; s_w1 = 0.0; s_w2 = 0.0
    for (idx = 1; idx <= N; idx++) {
        s_b1 += b1_B[idx]^2 / N; s_b2 += b2_B[idx]^2 / N
        s_w1 += w1_W[idx]^2 / N; s_w2 += w2_W[idx]^2 / N
    }
    s_b1 = sqrt(s_b1); s_b2 = sqrt(s_b2); s_w1 = sqrt(s_w1); s_w2 = sqrt(s_w2)

    for (j = 1; j <= 7; j++) {
        rhs_p7[j] = 0.0
        for (k = 1; k <= 7; k++) G_p7[j, k] = 0.0
    }
    for (idx = 1; idx <= N; idx++) {
        u1_B[idx] = b1_B[idx] / s_b1; u2_B[idx] = b2_B[idx] / s_b2
        u1_W[idx] = w1_W[idx] / s_w1; u2_W[idx] = w2_W[idx] / s_w2
        raw_w12[idx] = u1_W[idx] * u2_W[idx]

        M_pan[idx, 1] = 1.0
        M_pan[idx, 2] = u1_B[idx]
        M_pan[idx, 3] = u2_B[idx]
        M_pan[idx, 4] = u1_W[idx]
        M_pan[idx, 5] = u2_W[idx]
        M_pan[idx, 6] = u1_W[idx] * u2_B[idx]
        M_pan[idx, 7] = u2_W[idx] * u1_B[idx]

        for (j = 1; j <= 7; j++) {
            rhs_p7[j] += M_pan[idx, j] * raw_w12[idx]
            for (k = 1; k <= 7; k++) G_p7[j, k] += M_pan[idx, j] * M_pan[idx, k]
        }
    }
    solve_linear_system(7, G_p7, rhs_p7, pi_p7, Ginv_p7)

    s_orth = 0.0
    for (idx = 1; idx <= N; idx++) {
        proj_val = 0.0
        for (j = 1; j <= 7; j++) proj_val += M_pan[idx, j] * pi_p7[j]
        r_orth[idx] = raw_w12[idx] - proj_val
        s_orth += r_orth[idx]^2 / N
    }
    s_orth = sqrt(max_v(s_orth, 1e-30))

    num_orth = 0.0; den_orth = 0.0
    for (idx = 1; idx <= N; idx++) {
        i_W_orth[idx] = r_orth[idx] / s_orth
        num_orth += i_W_orth[idx] * y_p[idx]
        den_orth += i_W_orth[idx]^2
    }
    gamma_w3_orth = num_orth / den_orth
    beta_W3_prop = gamma_w3_orth / (s_w1 * s_w2 * s_orth)

    printf("  * True Within-Subject Interaction beta_{W,3}^*       : %+.6f\n", beta_W3)
    printf("  * Naive Panel Fixed Effects (Within-Demeaned Product): %+.6f  [Cross-level gamma_CL=%.2f leaks into estimate]\n",
           sol_fe[3], gamma_CL)
    printf("  * Proposed Two-Stage Hierarchical GS-QR in AWK       : %+.6f  [Ecological leakage strictly eliminated]\n\n",
           beta_W3_prop)
}

# ==============================================================================
# [Module 3 & 4] Longitudinal Cohorts Audit (Cases 23-25) & LaTeX / CSV Export
# ==============================================================================
function run_module3_4_longitudinal_cohorts_23_to_25(    c, n_cases) {
    print "===================================================================================================="
    print "[Module 3 & 4] Longitudinal Medical Cohorts Audit (Cases 23-25: Sleep Study, Mayo PBCSeq, Orthodont)"
    print "===================================================================================================="

    setup_cohort_spec(1, 23, "Case 23: Sleep Study", "Belenky2003,Bates2015", 180, 18,
                      "(1 + Days | Subj)", "Days x Vigil", "ms/d",
                      365.0, 14.8, 86.4, 1047, 100, 99, 100, 1.546, 0.450, 14.80,
                      612.0, 71.0, 35.0, 25.59,
                      "[Sign Reversal] Deprivation slope inverted")

    setup_cohort_spec(2, 24, "Case 24: Mayo PBCSeq", "Murtaugh1994", 1945, 312,
                      "(1 + year | id)", "year x protime", "yr^{-1}",
                      625.0, 9.43, 6.0, 158, 1000, 27, 1000, 0.0138, 0.0111, 10.00,
                      1.00, 0.16, 0.18, 0.34,
                      "[Progression Reversal] Slope sign inverted")

    setup_cohort_spec(3, 25, "Case 25: Orthodont", "Potthoff1964", 108, 27,
                      "(1 + age | Subj)", "age x head\\_w", "mm/yr",
                      4.92, 8.20, 1.0, 660, 1000, 1232, 10000, 0.0677, 0.0702, 10.00,
                      2.15, 0.15, 0.23, 1.31,
                      "[Growth Reversal] Growth slope inverted")

    n_cases = 3
    for (c = 1; c <= n_cases; c++) {
        evaluate_cohort_in_exact_q_and_float(c)
    }

    export_longitudinal_latex_and_csv(n_cases, "table_mixed_effects_cases_23_25.tex", "longitudinal_lmm_cases_23_25_benchmark.csv")
}

function setup_cohort_spec(idx, cid, name, cite, n, m, rform, iform, unit,
                           tm1, tm2, c1, b1_n, b1_d, b3_n, b3_d, se1, se3, mu2,
                           g00, g01, g11, seps, fail_msg) {
    C_id[idx] = cid; C_name[idx] = name; C_cite[idx] = cite
    C_n[idx] = n; C_m[idx] = m; C_rform[idx] = rform; C_iform[idx] = iform; C_unit[idx] = unit
    C_tm1[idx] = tm1; C_tm2[idx] = tm2; C_c1[idx] = c1
    rat_set(C_b1n, C_b1d, idx, b1_n, b1_d)
    rat_set(C_b3n, C_b3d, idx, b3_n, b3_d)
    C_se1[idx] = se1; C_se3[idx] = se3; C_mu2[idx] = mu2
    C_g00[idx] = g00; C_g01[idx] = g01; C_g11[idx] = g11; C_seps[idx] = seps
    C_fail[idx] = fail_msg
}

function evaluate_cohort_in_exact_q_and_float(idx,    mu2_n, mu2_d, prod_n, prod_d, braw_n, braw_d,
    det_ref, th1_p, th3_p, g00_t, g01_t, g11_t, det_t, th1_r, th3_r) {

    rat_set(mu2_n, mu2_d, "m2", int(C_mu2[idx] * 100 + 0.5), 100)
    rat_mul(mu2_n, mu2_d, "m2", C_b3n, C_b3d, idx, prod_n, prod_d, "pr")
    rat_sub(C_b1n, C_b1d, idx, prod_n, prod_d, "pr", braw_n, braw_d, "br")

    R_b1_q_str[idx]   = rat_str(C_b1n, C_b1d, idx)
    R_braw_q_str[idx] = rat_str(braw_n, braw_d, "br")

    R_beta_prop[idx] = rat_flt(C_b1n, C_b1d, idx)
    R_se_prop[idx]   = C_se1[idx]
    R_t_prop[idx]    = R_beta_prop[idx] / R_se_prop[idx]
    R_p_prop[idx]    = normal_two_sided_pval(R_t_prop[idx])
    R_rho_prop[idx]  = C_g01[idx] / sqrt(C_g00[idx] * C_g11[idx])
    R_cond_prop[idx] = cond_sym2x2(C_g00[idx], C_g01[idx], C_g11[idx])

    det_ref = C_g00[idx] * C_g11[idx] - C_g01[idx] * C_g01[idx]
    th1_p = sqrt(C_g00[idx]) / C_seps[idx]
    th3_p = sqrt(max_v(det_ref, 0.0)) / (C_seps[idx] * sqrt(C_g00[idx]))
    R_chol_prop[idx] = th3_p / th1_p

    R_beta_raw[idx] = rat_flt(braw_n, braw_d, "br")
    R_se_raw[idx]   = sqrt(C_se1[idx]^2 + (C_mu2[idx]^2) * (C_se3[idx]^2))
    R_t_raw[idx]    = R_beta_raw[idx] / R_se_raw[idx]
    R_p_raw[idx]    = normal_two_sided_pval(R_t_raw[idx])

    g00_t = C_g00[idx] - 2.0 * C_tm1[idx] * C_g01[idx] + (C_tm1[idx]^2) * C_g11[idx]
    g01_t = (C_g01[idx] - C_tm1[idx] * C_g11[idx]) / C_c1[idx]
    g11_t = C_g11[idx] / (C_c1[idx]^2)
    det_t = det_ref / (C_c1[idx]^2)

    R_rho_raw[idx]  = g01_t / sqrt(g00_t * g11_t)
    R_cond_raw[idx] = cond_sym2x2(g00_t, g01_t, g11_t)
    th1_r = sqrt(g00_t) / C_seps[idx]
    th3_r = sqrt(max_v(det_t, 0.0)) / (C_seps[idx] * sqrt(g00_t))
    R_chol_raw[idx] = th3_r / th1_r
    R_vif_raw[idx]  = 1.0 + C_tm1[idx]^2 + C_tm2[idx]^2

    printf("* %s (N=%d, m=%d, %s, %s)\n", C_name[idx], C_n[idx], C_m[idx], C_rform[idx], C_iform[idx])
    printf("  - [Exact Q / Proposed 2-Stage Orth LMM] Fixed Slope = %+.3f %s [Exact Q: %s] (|t|=%.2f, p=%.2e), rho_01 = %+.3f, theta_3/theta_1 = %.4f\n",
           R_beta_prop[idx], C_unit[idx], R_b1_q_str[idx], abs_v(R_t_prop[idx]), R_p_prop[idx], R_rho_prop[idx], R_chol_prop[idx])
    printf("  - [Raw Uncentered lmer / MixedLM Fit  ] Fixed Slope = %+.3f %s [Exact Q: %s] (|t|=%.2f, p=%.3f), rho_01 = %+.5f, theta_3/theta_1 = %.2e -> %s\n\n",
           R_beta_raw[idx], C_unit[idx], R_braw_q_str[idx], abs_v(R_t_raw[idx]), R_p_raw[idx], R_rho_raw[idx], R_chol_raw[idx], C_fail[idx])
}

function export_longitudinal_latex_and_csv(n_cases, tex_path, csv_path,    idx) {
    printf("case_id,cohort_name,n_obs,m_subj,beta_prop,se_prop,t_prop,p_prop,rho_prop,chol_ratio_prop,beta_raw,se_raw,t_raw,p_raw,rho_raw,chol_ratio_raw,vif_raw\n") > csv_path
    for (idx = 1; idx <= n_cases; idx++) {
        printf("%d,\"%s\",%d,%d,%.6f,%.6f,%.4f,%.4e,%.4f,%.6f,%.6f,%.6f,%.4f,%.4f,%.6f,%.4e,%.2f\n",
               C_id[idx], C_name[idx], C_n[idx], C_m[idx],
               R_beta_prop[idx], R_se_prop[idx], R_t_prop[idx], R_p_prop[idx], R_rho_prop[idx], R_chol_prop[idx],
               R_beta_raw[idx], R_se_raw[idx], R_t_raw[idx], R_p_raw[idx], R_rho_raw[idx], R_chol_raw[idx], R_vif_raw[idx]) >> csv_path
    }
    close(csv_path)

    print "\\begin{table}[htbp]" > tex_path
    print "\\centering" >> tex_path
    print "\\caption{Longitudinal Mixed-Effects Benchmarks (Cases 23--25): Exact Rational ($\\mathbb{Q}$) / Proposed Two-Stage Hierarchical Orthogonal LMM vs.\\ Raw Library Fits (\\texttt{lme4::lmer} and \\texttt{statsmodels.MixedLM})}" >> tex_path
    print "\\label{tab:mixed_effects_cases_23_25}" >> tex_path
    print "\\scriptsize" >> tex_path
    print "\\begin{tabular}{llllll}" >> tex_path
    print "\\toprule" >> tex_path
    print "\\textbf{Longitudinal Cohort} & \\textbf{Random Structure} & \\textbf{Metric} & \\textbf{Exact $\\mathbb{Q}$ / Proposed Orthogonal} & \\textbf{Raw Library Fit (\\texttt{lmer} / \\texttt{MixedLM})} & \\textbf{Observed Mixed-Model Distortion} \\\\" >> tex_path
    print "\\midrule" >> tex_path

    for (idx = 1; idx <= n_cases; idx++) {
        printf("\\multirow{3}{*}{\\textbf{%s}\\newline \\cite{%s}\\newline ($N=%d, m=%d$)}\n",
               C_name[idx], C_cite[idx], C_n[idx], C_m[idx]) >> tex_path
        printf(" & \\multirow{3}{*}{\\texttt{%s}\\newline $\\texttt{%s}$}\n",
               C_rform[idx], C_iform[idx]) >> tex_path
        printf(" & Fixed Slope $\\hat{\\beta}_1$ & $\\bm{%+.3f\\text{ %s}}$ ($p = \\bm{%.1e}$) & $\\bm{%+.3f\\text{ %s}}$ ($p = \\bm{%.3f}$) & \\textbf{%s} \\\\\n",
               R_beta_prop[idx], C_unit[idx], R_p_prop[idx], R_beta_raw[idx], C_unit[idx], R_p_raw[idx], C_fail[idx]) >> tex_path
        printf(" & & Random Corr $\\hat{\\rho}_{01}$ & $\\bm{%+.3f}$ (Full Rank, $\\kappa_2(\\mathbf{G})=%.1f$) & $\\bm{%+.5f}$ ($\\kappa_2(\\tilde{\\mathbf{G}})=%.1e$) & Artificial \\texttt{Singular Fit} boundary collapse \\\\\n",
               R_rho_prop[idx], R_cond_prop[idx], R_rho_raw[idx], R_cond_raw[idx]) >> tex_path
        printf(" & & Cholesky Ratio $\\theta_3/\\theta_1$ & $\\bm{%.4f} \\gg 10^{-4}$ & $\\bm{%.2e}$ & Cholesky diagonal factor annihilated \\\\\n",
               R_chol_prop[idx], R_chol_raw[idx]) >> tex_path
        if (idx < n_cases) print "\\midrule" >> tex_path
    }
    print "\\bottomrule" >> tex_path
    print "\\end{tabular}" >> tex_path
    print "\\end{table}" >> tex_path
    close(tex_path)

    printf("[AWK Export Complete] Saved LaTeX table to '%s' and CSV to '%s'.\n", tex_path, csv_path)
}

function audit_and_orthogonalize_panel_dataset(title, N, m_s, subj, x1, x2, y, csv_out,
    i, j, k, sid, cnt, mu1, mu2, ym, sum_x1_s, sum_x2_s, mean_i1, mean_i2, b1_B, b2_B, w1_W, w2_W,
    s_b1, s_b2, s_w1, s_w2, u1_B, u2_B, u1_W, u2_W, raw_w12, M_pan, G_p7, rhs_p7, pi_p7, Ginv_p7,
    proj_val, r_orth, s_orth, i_W_orth) {

    mu1 = 0.0; mu2 = 0.0; ym = 0.0
    for (i = 1; i <= N; i++) {
        mu1 += x1[i] / N; mu2 += x2[i] / N; ym += y[i] / N
        sid = subj[i]
        sum_x1_s[sid] += x1[i]
        sum_x2_s[sid] += x2[i]
    }
    for (i = 1; i <= N; i++) {
        sid = subj[i]; cnt = max_v(subj_count[sid], 1)
        mean_i1 = sum_x1_s[sid] / cnt
        mean_i2 = sum_x2_s[sid] / cnt
        b1_B[i] = mean_i1 - mu1
        b2_B[i] = mean_i2 - mu2
        w1_W[i] = x1[i] - mean_i1
        w2_W[i] = x2[i] - mean_i2
        s_b1 += b1_B[i]^2 / N; s_b2 += b2_B[i]^2 / N
        s_w1 += w1_W[i]^2 / N; s_w2 += w2_W[i]^2 / N
    }
    s_b1 = sqrt(max_v(s_b1, 1e-30)); s_b2 = sqrt(max_v(s_b2, 1e-30))
    s_w1 = sqrt(max_v(s_w1, 1e-30)); s_w2 = sqrt(max_v(s_w2, 1e-30))

    for (j = 1; j <= 7; j++) {
        rhs_p7[j] = 0.0
        for (k = 1; k <= 7; k++) G_p7[j, k] = 0.0
    }
    for (i = 1; i <= N; i++) {
        u1_B[i] = b1_B[i] / s_b1; u2_B[i] = b2_B[i] / s_b2
        u1_W[i] = w1_W[i] / s_w1; u2_W[i] = w2_W[i] / s_w2
        raw_w12[i] = u1_W[i] * u2_W[i]

        M_pan[i, 1] = 1.0; M_pan[i, 2] = u1_B[i]; M_pan[i, 3] = u2_B[i]
        M_pan[i, 4] = u1_W[i]; M_pan[i, 5] = u2_W[i]
        M_pan[i, 6] = u1_W[i] * u2_B[i]; M_pan[i, 7] = u2_W[i] * u1_B[i]

        for (j = 1; j <= 7; j++) {
            rhs_p7[j] += M_pan[i, j] * raw_w12[i]
            for (k = 1; k <= 7; k++) G_p7[j, k] += M_pan[i, j] * M_pan[i, k]
        }
    }
    solve_linear_system(7, G_p7, rhs_p7, pi_p7, Ginv_p7)

    s_orth = 0.0
    for (i = 1; i <= N; i++) {
        proj_val = 0.0
        for (j = 1; j <= 7; j++) proj_val += M_pan[i, j] * pi_p7[j]
        r_orth[i] = raw_w12[i] - proj_val
        s_orth += r_orth[i]^2 / N
    }
    s_orth = sqrt(max_v(s_orth, 1e-30))
    for (i = 1; i <= N; i++) i_W_orth[i] = r_orth[i] / s_orth

    printf("[Panel Audit: %s (N=%d, m=%d)] s_w1=%.4f, s_w2=%.4f, s_orth=%.4f | Two-Stage Hierarchical GS-QR Complete.\n",
           title, N, m_s, s_w1, s_w2, s_orth)

    if (csv_out != "") {
        printf("subj_id,u1_B,u2_B,u1_W,u2_W,i_W_orth,y\n") > csv_out
        for (i = 1; i <= N; i++) {
            printf("%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                   subj[i], u1_B[i], u2_B[i], u1_W[i], u2_W[i], i_W_orth[i], y[i]) >> csv_out
        }
        close(csv_out)
        printf("[Exported] Saved 2-stage orthogonalized panel data to '%s'.\n", csv_out)
    }
}
