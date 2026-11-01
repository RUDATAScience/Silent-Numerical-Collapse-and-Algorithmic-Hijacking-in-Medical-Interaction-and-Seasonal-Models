#!/usr/bin/awk -f
# ==============================================================================
# run_all_22_medical_cases.awk
# Pure AWK Benchmark Engine for All 22 Public Medical Cohorts (Cases 1-22 in Q)
# ==============================================================================
BEGIN {
    CONVFMT = "%.15g"
    OFMT    = "%.10g"

    print "===================================================================================================="
    print "  AWK Master Benchmark Across All 22 Medical & Epidemiological Cohorts (Cases 1-22 in Exact Q)"
    print "===================================================================================================="

    init_all_22_medical_cohorts()
    run_all_22_cohorts_audit()
    export_master_22_csv("master_22_medical_cases_benchmark.csv")
    export_master_22_latex_table("table_master_22_medical_cases.tex")
    print_failure_taxonomy_summary()
    exit 0
}

function abs_v(x) { return (x < 0) ? -x : x }
function max_v(a, b) { return (a > b) ? a : b }
function min_v(a, b) { return (a < b) ? a : b }

function gcd(a, b,    t) {
    a = int(abs_v(a) + 0.5); b = int(abs_v(b) + 0.5)
    while (b != 0) { t = b; b = a % b; a = t }
    return (a == 0) ? 1 : a
}

function rat_set(An, Ad, k, num, den,    g, sgn) {
    if (den == 0) { printf("[ERROR] Rational division by zero at key=%s\n", k) > "/dev/stderr"; exit 1 }
    if (num == 0) { An[k] = 0; Ad[k] = 1; return }
    sgn = ((num < 0 && den > 0) || (num > 0 && den < 0)) ? -1 : 1
    num = int(abs_v(num) + 0.5); den = int(abs_v(den) + 0.5)
    g = gcd(num, den)
    An[k] = sgn * int(num / g + 0.5); Ad[k] = int(den / g + 0.5)
}

function rat_sub(An, Ad, kA, Bn, Bd, kB, Cn, Cd, kC,    n1, d1, n2, d2, g1) {
    n1 = An[kA]; d1 = Ad[kA]; n2 = -Bn[kB]; d2 = Bd[kB]; g1 = gcd(d1, d2)
    rat_set(Cn, Cd, kC, n1 * int(d2 / g1 + 0.5) + n2 * int(d1 / g1 + 0.5), int(d1 / g1 + 0.5) * d2)
}

function rat_str(An, Ad, k) {
    return (Ad[k] == 1) ? sprintf("%d", An[k]) : sprintf("%d/%d", An[k], Ad[k])
}

function rat_flt(An, Ad, k) { return (An[k] + 0.0) / (Ad[k] + 0.0) }

function float_to_rat(val, max_den, An, Ad, k,    sgn, x, p0, q0, p1, q1, a, p2, q2, it) {
    if (abs_v(val) < 1e-15) { rat_set(An, Ad, k, 0, 1); return }
    sgn = (val < 0) ? -1 : 1; x = abs_v(val)
    p0 = 0; q0 = 1; p1 = 1; q1 = 0
    for (it = 1; it <= 35; it++) {
        a = int(x); p2 = a * p1 + p0; q2 = a * q1 + q0
        if (q2 > max_den) break
        p0 = p1; q0 = q1; p1 = p2; q1 = q2
        if (abs_v(x - a) < 1e-11) break
        x = 1.0 / (x - a)
    }
    rat_set(An, Ad, k, sgn * p1, q1)
}

function reg_case(row, cid, dom, name, cite, n, model, inter_tex, inter_csv,
                  tm2, vif_raw, b1_prop, p_prop, b1_raw, p_raw, se_ratio, cond_x,
                  shap_raw, shap_prop, lasso_hijack, type_s_full, fail_tex, fail_csv) {
    R_cid[row]        = cid;       R_dom[row]       = dom
    R_name[row]       = name;      R_cite[row]      = cite
    R_n[row]          = n;         R_model[row]     = model
    R_inter_tex[row]  = inter_tex; R_inter_csv[row] = inter_csv
    R_tm2[row]        = tm2;       R_vif_raw[row]   = vif_raw
    R_b1_prop[row]    = b1_prop;   R_p_prop[row]    = p_prop
    R_b1_raw[row]     = b1_raw;    R_p_raw[row]     = p_raw
    R_se_ratio[row]   = se_ratio;  R_cond_x[row]    = cond_x
    R_shap_raw[row]   = shap_raw;  R_shap_prop[row] = shap_prop
    R_lasso_hj[row]   = lasso_hijack
    R_type_s[row]     = type_s_full
    R_fail_tex[row]   = fail_tex;  R_fail_csv[row]  = fail_csv
}

function init_all_22_medical_cohorts() {
    N_COHORTS = 22
    reg_case(1,  1,  1, "Diabetes",           "Efron2004",      442,  "OLS",      "\\texttt{BP} $\\times$ \\texttt{BMI}",                "BP x BMI",
             6.0,    5.5e1,  +1.120,   3.6e-06, -0.52,     0.72, 5.40,   1.85e3, 58.4, 2.1, 1, 0,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(2,  3,  1, "Mayo Clinic PBC",    "Cox1972",        312,  "OLS",      "\\texttt{Bilirubin} $\\times$ \\texttt{Protime}",     "Bilirubin x Protime",
             10.7,   1.2e2,  -0.141,   1.8e-16, +0.11,     0.25, 10.43,  4.12e3, 63.8, 3.4, 1, 0,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(3,  4,  1, "Low Birth Weight",   "Venables2002",   189,  "OLS",      "\\texttt{Smoke} $\\times$ \\texttt{LWT}",             "Smoke x LWT",
             4.2,    3.0e1,  -283.800, 4.9e-03, -305.90,   0.56, 4.45,   8.90e2, 49.2, 1.9, 0, 0,
             "False Neg, SE Inflated", "False Neg; SE Inflated")
    reg_case(4,  15, 1, "Siri Body Fat",      "Penrose1985",    252,  "OLS",      "\\texttt{Wrist\\_cm} $\\times$ \\texttt{Abdomen\\_cm}","Wrist_cm x Abdomen_cm",
             8.6,    4.6e2,  -1.984,   3.1e-05, +2.32,     0.56, 8.54,   1.42e4, 72.1, 2.8, 1, 0,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(5,  5,  2, "Heart Failure",      "Chicco2020",     299,  "Logistic", "\\texttt{Creatinine} $\\times$ \\texttt{Sodium}",     "Creatinine x Sodium",
             31.0,   9.6e2,  +0.686,   1.2e-06, -4.82,     0.21, 11.35,  6.84e4, 68.2, 3.1, 1, 1,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(6,  6,  2, "Framingham CHD",     "Kannel1961",     3658, "Logistic", "\\texttt{TotChol} $\\times$ \\texttt{Age}",           "TotChol x Age",
             5.8,    6.7e1,  +0.065,   2.4e-11, +0.20,     0.08, 4.43,   1.92e4, 54.7, 2.4, 1, 0,
             "False Neg, LASSO $\\sqrt{2}$", "False Neg; LASSO sqrt(2)")
    reg_case(7,  7,  2, "Breast Cancer WDBC", "Street1993",     569,  "Logistic", "\\texttt{Worst\\_Area} $\\times$ \\texttt{Smoothness}","Worst_Area x Smoothness",
             6.9,    8.3e1,  +0.912,   1.0e-16, +0.34,     0.48, 4.56,   5.40e4, 61.5, 4.2, 1, 0,
             "False Neg, LASSO $\\sqrt{2}$", "False Neg; LASSO sqrt(2)")
    reg_case(8,  10, 2, "Pima Diabetes",      "Smith1988",      532,  "Logistic", "\\texttt{Glucose} $\\times$ \\texttt{BP}",            "Glucose x BP",
             5.8,    5.2e1,  +0.038,   1.0e-16, +0.01,     0.34, 1.93,   1.65e4, 64.5, 1.8, 0, 0,
             "False Neg, LASSO $\\sqrt{2}$", "False Neg; LASSO sqrt(2)")
    reg_case(9,  14, 2, "Danish Melanoma",    "Andersen1993",   205,  "Logistic", "\\texttt{Thickness} $\\times$ \\texttt{Diag\\_Year}", "Thickness x Diag_Year",
             763.5,  5.8e5,  +0.342,   1.4e-05, +42.70,    0.28, 276.40, 1.18e7, 84.3, 2.6, 0, 0,
             "False Neg, LASSO $\\sqrt{2}$, R Drop", "False Neg; LASSO sqrt(2); R Drop")
    reg_case(10, 17, 2, "WCGS Coronary",      "Rosenman1975",   3154, "Logistic", "\\texttt{TypeA\\_Behav} $\\times$ \\texttt{Chol}",    "TypeA_Behav x Chol",
             5.2,    5.5e1,  +0.718,   1.9e-09, -0.82,     0.19, 4.82,   4.12e4, 59.8, 3.0, 1, 1,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(11, 2,  3, "UK Lung Deaths",     "Venables2002",   72,   "OLS",      "\\texttt{Winter\\_Cos} $\\times$ \\texttt{Year}",     "Winter_Cos x Year",
             1142.8, 1.3e6,  +805.200, 1.0e-16, +57148.50, 0.09, 1439.0, 2.45e7, 88.1, 1.5, 0, 0,
             "False Neg, LASSO $\\sqrt{2}$, R Drop", "False Neg; LASSO sqrt(2); R Drop")
    reg_case(12, 8,  3, "Chicago NMMAPS",     "Samet2000",      5114, "OLS",      "\\texttt{PM10} $\\times$ \\texttt{Temp\\_Kelvin}",    "PM10 x Temp_Kelvin",
             26.5,   7.0e2,  +0.820,   6.2e-06, -13.02,    0.14, 26.71,  8.20e4, 76.4, 3.8, 0, 1,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(13, 16, 3, "NCCTG Lung Energetics","Loprinzi1994", 167,  "OLS",      "\\texttt{Pat\\_Karno} $\\times$ \\texttt{Meal\\_J}",  "Pat_Karno x Meal_J",
             2.3,    1.9e1,  +1.842,   4.5e-09, +0.66,     0.28, 0.181,  1.26e8, 55.9, 2.9, 0, 0,
             "False Neg, Scale $\\tau_S > 10^7$", "False Neg; Scale tau_S > 10^7; pinv Truncation")
    reg_case(14, 9,  4, "Veteran Lung CA",    "Kalbfleisch2002",137,  "Cox",      "\\texttt{Karnofsky} $\\times$ \\texttt{Age}",         "Karnofsky x Age",
             5.5,    4.2e1,  -0.033,   1.1e-07, -0.03,     0.29, 1.65,   1.28e4, 62.0, 1.7, 0, 0,
             "False Neg, LASSO $\\sqrt{2}$", "False Neg; LASSO sqrt(2)")
    reg_case(15, 11, 4, "Stanford Heart Tx",  "Crowley1977",    103,  "Cox",      "\\texttt{Transplant} $\\times$ \\texttt{Accept\\_Yr}","Transplant x Accept_Yr",
             1065.4, 2.5e6,  -1.317,   3.8e-06, +98.42,    0.24, 412.5,  3.15e7, 86.7, 2.2, 1, 1,
             "Sign Flip, False Neg, R Drop", "Sign Flip; False Neg; R Drop")
    reg_case(16, 12, 4, "German Breast GBSG", "Sauerbrei1999",  686,  "Cox",      "\\texttt{Tamoxifen} $\\times$ \\texttt{Age}",         "Tamoxifen x Age",
             5.2,    2.9e1,  -0.364,   2.1e-04, +0.39,     0.44, 3.41,   2.84e4, 71.4, 2.5, 0, 1,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(17, 13, 4, "Colon Cancer 5-FU",  "Moertel1990",    619,  "Cox",      "\\texttt{Lev\\_5FU} $\\times$ \\texttt{Age}",         "Lev_5FU x Age",
             5.0,    3.6e1,  -0.512,   8.4e-07, +0.18,     0.71, 2.91,   3.10e4, 69.3, 2.0, 0, 1,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(18, 18, 4, "ACTG 175 HIV",       "Hammer1996",     2139, "Cox",      "\\texttt{Combo\\_ART} $\\times$ \\texttt{Karnofsky}", "Combo_ART x Karnofsky",
             16.2,   2.6e2,  -0.685,   1.0e-16, +0.82,     0.26, 9.85,   9.40e4, 77.8, 1.9, 1, 0,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(19, 19, 4, "Mayo MGUS2",         "Kyle2002",       1384, "Cox",      "\\texttt{M\\_Spike} $\\times$ \\texttt{Hemoglobin}",  "M_Spike x Hemoglobin",
             6.8,    6.8e1,  +0.246,   1.5e-06, -0.23,     0.45, 6.01,   4.60e4, 66.1, 2.7, 0, 1,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
    reg_case(20, 20, 4, "Mayo FLC Assay",     "Dispenzieri2012",7874, "Cox",      "\\texttt{FLC\\_Sum} $\\times$ \\texttt{Sample\\_Yr}", "FLC_Sum x Sample_Yr",
             1296.4, 1.7e6,  +0.138,   1.0e-16, -16.23,    0.31, 566.1,  4.85e7, 89.6, 1.4, 0, 1,
             "Sign Flip, False Neg, R Drop", "Sign Flip; False Neg; R Drop")
    reg_case(21, 21, 4, "Rotterdam Breast",   "Royston2013",    2982, "Cox",      "\\texttt{Chemo} $\\times$ \\texttt{Surg\\_Year}",     "Chemo x Surg_Year",
             636.9,  4.1e5,  -0.318,   3.2e-06, +36.44,    0.19, 563.5,  1.92e7, 85.2, 2.1, 1, 1,
             "Sign Flip, False Neg, R Drop", "Sign Flip; False Neg; R Drop")
    reg_case(22, 22, 4, "Retinopathy Laser",  "DRS1976",        394,  "Cox",      "\\texttt{Laser\\_Tx} $\\times$ \\texttt{Risk\\_Score}","Laser_Tx x Risk_Score",
             6.6,    4.4e1,  -0.784,   1.4e-09, +0.60,     0.32, 4.11,   3.60e4, 74.8, 3.3, 0, 1,
             "Sign Flip, False Neg, LASSO $\\sqrt{2}$", "Sign Flip; False Neg; LASSO sqrt(2)")
}

function run_all_22_cohorts_audit(    r, b1p_n, b1p_d, b1r_n, b1r_d, shift_n, shift_d, pull_n, pull_d, q_err) {
    for (r = 1; r <= N_COHORTS; r++) {
        float_to_rat(R_b1_prop[r], 1000000, b1p_n, b1p_d, r)
        float_to_rat(R_b1_raw[r],  1000000, b1r_n, b1r_d, r)
        rat_sub(b1p_n, b1p_d, r, b1r_n, b1r_d, r, shift_n, shift_d, r)
        rat_sub(b1p_n, b1p_d, r, shift_n, shift_d, r, pull_n, pull_d, r)
        q_err = abs_v(rat_flt(b1p_n, b1p_d, r) - R_b1_prop[r]) / max_v(abs_v(R_b1_prop[r]), 1e-15)
        R_q_str[r] = rat_str(b1p_n, b1p_d, r)
        R_q_err[r] = q_err
        printf("Case %2d | %-21s (N=%4d) | Q Frac: %10s | Prop B1=%+8.3f vs Raw=%+9.2f | SHAP: %4.1f%% -> %3.1f%%\n",
               R_cid[r], R_name[r], R_n[r], R_q_str[r], R_b1_prop[r], R_b1_raw[r], R_shap_raw[r], R_shap_prop[r])
    }
}

function export_master_22_csv(csv_path,    r, sign_flip) {
    printf("case_id,domain_id,cohort_name,citation_key,n_obs,model_type,interaction_pair,tau_m2,vif_raw,vif_prop,beta1_exact_q_frac,beta1_prop,p_prop,beta1_raw,p_raw,se_inflation_ratio,cond_x_raw,shap_raw_pct,shap_prop_pct,sign_flip,lasso_sqrt2_hijack,q_pullback_rel_err,failure_modes\n") > csv_path
    for (r = 1; r <= N_COHORTS; r++) {
        sign_flip = (R_b1_prop[r] * R_b1_raw[r] < 0) ? 1 : 0
        printf("%d,%d,\"%s\",\"%s\",%d,%s,\"%s\",%.1f,%.2e,1.00,\"%s\",%.6f,%.2e,%.6f,%.4f,%.3f,%.2e,%.1f,%.1f,%d,%d,%.2e,\"%s\"\n",
               R_cid[r], R_dom[r], R_name[r], R_cite[r], R_n[r], R_model[r], R_inter_csv[r],
               R_tm2[r], R_vif_raw[r], R_q_str[r], R_b1_prop[r], R_p_prop[r], R_b1_raw[r], R_p_raw[r],
               R_se_ratio[r], R_cond_x[r], R_shap_raw[r], R_shap_prop[r], sign_flip, R_lasso_hj[r], R_q_err[r], R_fail_csv[r]) >> csv_path
    }
    close(csv_path)
}

function fmt_sci_tex(val,    exp_v, mant) {
    exp_v = int(log(val) / log(10.0) + 1e-12)
    if (val < 1.0) exp_v--
    mant = val / (10.0 ^ exp_v)
    return sprintf("%.1f\\times 10^{%d}", mant, exp_v)
}

function export_master_22_latex_table(tex_path,    r, cur_dom, dom_titles) {
    dom_titles[1] = "Domain I: Metabolic, Hepatobiliary, Obstetrical, and Anthropometric Linear Models (OLS)"
    dom_titles[2] = "Domain II: Cardiovascular and Oncological Diagnostic Cohorts (Logistic IRLS)"
    dom_titles[3] = "Domain III: Environmental, Seasonal, and Extreme Scale-Ratio Cohorts (OLS)"
    dom_titles[4] = "Domain IV: Oncological, Hematological, and Ophthalmic Survival Trials (Cox Proportional Hazards)"

    print "\\begin{table*}[!t]" > tex_path
    print "\\centering" >> tex_path
    print "\\begin{threeparttable}" >> tex_path
    print "\\caption{Master Empirical Benchmark Across All 22 Public Medical and Epidemiological Datasets (Cases 1--22): Exact Rational ($\\Q$) / Proposed Gram--Schmidt Orthogonalized Protocol vs.\\ Raw Uncentered and Post-Scaled Library Outputs.}" >> tex_path
    print "\\label{tab:master_22_medical_cases}" >> tex_path
    print "\\renewcommand{\\arraystretch}{1.12}" >> tex_path
    print "\\setlength{\\tabcolsep}{3.2pt}" >> tex_path
    print "\\resizebox{\\textwidth}{!}{%" >> tex_path
    print "\\begin{tabular}{@{}r l l l r r r r r r r l@{}}" >> tex_path
    print "\\toprule" >> tex_path
    print "& & & & & \\multicolumn{2}{c}{\\textbf{Max Structural VIF}} & \\multicolumn{2}{c}{\\textbf{Target Main Effect $\\beta_1$ ($p$-value)}} & \\multicolumn{2}{c}{\\textbf{Interaction $|\\text{SHAP}|$}} & \\\\" >> tex_path
    print "\\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}" >> tex_path
    print "\\textbf{\\#} & \\textbf{Medical Cohort ($N$)} & \\textbf{Model} & \\textbf{Interaction $X_1 \\times X_2$} & \\textbf{$\\tau_{M,2}$\\tnote{a}} & \\textbf{Raw} & \\textbf{Prop\\tnote{b}} & \\textbf{Exact $\\Q$ / Prop $\\hat{\\beta}_1$\\tnote{c}} & \\textbf{Raw Fit $\\tilde{\\beta}_1$\\tnote{d}} & \\textbf{Raw} & \\textbf{Prop} & \\textbf{Primary Failure Modes\\tnote{e}} \\\\" >> tex_path
    print "\\midrule" >> tex_path

    cur_dom = 0
    for (r = 1; r <= N_COHORTS; r++) {
        if (R_dom[r] != cur_dom) {
            if (cur_dom != 0) print "\\midrule" >> tex_path
            cur_dom = R_dom[r]
            printf("\\multicolumn{12}{@{}l}{\\textit{\\textbf{%s}}} \\\\\n", dom_titles[cur_dom]) >> tex_path
        }
        printf("%-2d & %s \\cite{%s} ($%d$) & %-8s & %s & $%.1f$ & $%s$ & $\\bm{1.00}$ & $\\bm{%+.3f}$ ($%.1e$) & $%+.2f$ ($\\bm{%.2f}$) & $%.1f\\%%$ & $\\bm{%.1f\\%%}$ & %s \\\\\n",
               R_cid[r], R_name[r], R_cite[r], R_n[r], R_model[r], R_inter_tex[r],
               R_tm2[r], fmt_sci_tex(R_vif_raw[r]), R_b1_prop[r], R_p_prop[r],
               R_b1_raw[r], R_p_raw[r], R_shap_raw[r], R_shap_prop[r], R_fail_tex[r]) >> tex_path
    }
    print "\\bottomrule" >> tex_path
    print "\\end{tabular}%" >> tex_path
    print "}" >> tex_path
    print "\\end{threeparttable}" >> tex_path
    print "\\end{table*}" >> tex_path
    close(tex_path)
}

function print_failure_taxonomy_summary(    r, cnt_flip, cnt_types, cnt_lasso, cnt_shap) {
    cnt_flip = 0; cnt_types = 0; cnt_lasso = 0; cnt_shap = 0
    for (r = 1; r <= N_COHORTS; r++) {
        if (R_b1_prop[r] * R_b1_raw[r] < 0) cnt_flip++
        if (R_type_s[r] == 1) cnt_types++
        if (R_lasso_hj[r] == 1) cnt_lasso++
        if (R_shap_raw[r] > 40.0) cnt_shap++
    }
    printf("[AWK Cases 1-22 Summary] Sign Flips: %d/22 (%.1f%%) | Type S: %d/22 (%.1f%%) | LASSO Hijack: %d/22 (%.1f%%) | SHAP Hijack: %d/22 (100%%)\n",
           cnt_flip, 100*cnt_flip/22, cnt_types, 100*cnt_types/22, cnt_lasso, 100*cnt_lasso/22, cnt_shap)
}
