# Silent Numerical Collapse and Algorithmic Hijacking in Medical Interaction and Seasonal Models
### Exact Rational ($\mathbb{Q}$) Benchmarks vs. Floating-Point Statistical, Machine Learning, and Bayesian HMC-NUTS Libraries Across 28 Clinical Cohorts

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![LaTeX: LuaLaTeX](https://img.shields.io/badge/LaTeX-LuaLaTeX%20%2B%20pgfplots-teal.svg)](#manuscript-compilation-overleaf--lualatex)
[![POSIX: AWK](https://img.shields.io/badge/Engine-POSIX%20%2F%20GNU%20AWK%20Exact%20Q-orange.svg)](#2-zero-dependency-posixgnu-awk-execution-no-pythonr-required)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B%20%7C%20SymPy%20%7C%20PyMC%20%7C%20Stan-green.svg)](#3-python--sympy--stan--pymc-verification-suite)

This repository contains the complete mathematical proofs, exact rational ($\mathbb{Q}$) computer algebra benchmarks, zero-dependency POSIX/GNU `awk` engines, Python/Stan/PyMC verification suites, and the full LuaLaTeX manuscript (`main.tex`) for auditing and curing numerical and statistical failures in multiplicative interaction ($X_1 \times X_2$) and seasonal harmonic models across **20 Theoretical Propositions** and **28 Public Medical Cohorts (Cases 1–28)**.

---

## 1. Overview of Key Findings

In clinical biostatistics, epidemiology, and medical AI, physiological homeostasis (e.g., serum sodium, prothrombin time, blood pressure, Kelvin temperature) and chronological registries (e.g., calendar year of surgery/diagnosis) induce large location-to-scale ratios $\tau_{M,j} = \vert{}\mu_j / \sigma_j\vert{} \gg 1$ and severe unit-scale asymmetries $\tau_S = \max(\sigma_1/\sigma_2, \sigma_2/\sigma_1)$. Anchoring every calculation to **exact symbolic LU decomposition over the rational field $\mathbb{Q}$ (`SymPy.Rational` and pure AWK rational arithmetic)**, we prove and empirically demonstrate that standard library workflows (`y ~ x1 * x2` or post-multiplication `StandardScaler`) fail across four major paradigms:

1. **Frequentist OLS, Logistic IRLS, and Cox PH (Propositions 1–8; Cases 1–22):**
   - Uncentered products shift primary main effects via $\tilde{\beta}_1 = \beta_1 - \mu_2 \beta_3$ and inflate variance by $\mathrm{VIF} = 1 + \tau_{M,1}^2 + \tau_{M,2}^2$ (which remains **strictly invariant** under post-multiplication `StandardScaler`).
   - Across 22 clinical cohorts, raw/post-scaled library fits induce **main-effect sign reversals in 59.1% (13/22)** of cohorts (**45.5% [10/22] statistically significant in the wrong clinical direction**, $p < 0.005$) and **silent SVD singular-value truncation (`pinv`)** when $\kappa_2(\mathbf{X}^\top\mathbf{X}) > \varepsilon_{\mathrm{mach}}^{-1}$ (Case 16).
   - Default $L_2$ regularization in `sklearn.linear_model.LogisticRegression(C=1.0)` penalizes $(\beta_1 - M_2\beta_3)^2$, silently erasing true clinical risk factors toward $\widetilde{\mathrm{OR}}_1 \to 1.00$ (Case 4).
2. **Sparse Regularization (LASSO) and GBDT / Interventional TreeSHAP (Propositions 9–14; Cases 1–22):**
   - **$\sqrt{2}$-Norm Shortcut Theorem (Theorem 10):** In post-scaled LASSO under a purely additive ground truth ($\beta_3^* = 0$), the KKT entry correlation of the product term scales as $\lambda_3^{\mathrm{entry}} = \beta \frac{2\tau_M}{\sqrt{2\tau_M^2+1}} \to \sqrt{2}\beta$, displacing true main effects whenever $\tau_M > 1/\sqrt{2} \approx 0.7071$ (observed in **40.9% [9/22]** of cohorts).
   - **Diagonal Split Hijacking (Propositions 12–14):** In CART and GBDT, axis-aligned splits on $X_1 X_2$ act as $45^\circ$ oblique hyperplane cuts that double root impurity reduction ($\Delta\mathcal{L}(X_3) = 2\Delta\mathcal{L}(X_1)$), hijacking **49.2%–89.6%** of global Interventional TreeSHAP attribution across all 22 cohorts.
3. **Longitudinal Linear Mixed-Effects Models (Propositions 15–17; Cases 23–25):**
   - Uncentered longitudinal covariates transform the random-effects covariance via $\tilde{\mathbf{G}} = \mathbf{L}_b \mathbf{G} \mathbf{L}_b^\top$, driving random intercept-slope correlation to the singular boundary ($\tilde{\rho}_{01} \to \pm 1$) and collapsing `lme4`'s relative Cholesky factor ratio as $\theta_3 / \theta_1 = \mathcal{O}((c_1 M_1^2)^{-1}) \to 0$ (`boundary (singular) fit`).
   - Within-demeaning of raw products ($X_{ij,1}X_{ij,2} - \overline{X_1 X_2}_{i\cdot}$) suffers cross-level ecological leakage ($b_{i,2}^{\mathrm{B}} w_{ij,1}^{\mathrm{W}}$), which is eliminated by **Two-Stage Hierarchical Gram–Schmidt Orthogonalization**.
4. **Bayesian HMC-NUTS Sampling in Stan and PyMC (Propositions 18.1–20; Cases 26–28):**
   - Isotropic priors $\mathcal{N}(\mathbf{0}, \tau_0^2\mathbf{I})$ on uncentered coordinates induce severe anisotropic shrinkage $\lambda_0 \Vert{}\mathbf{T}^{-1}\boldsymbol{\beta}\Vert{}_2^2$, locking small-scale main effects onto parasitic offsets $M_2\beta_3$.
   - Diagonal mass adaptation (`adapt_diag`) fails to remove off-diagonal posterior correlation $\rho_{\mathrm{post}} \to \mp 1$, inflating the preconditioned Hessian condition number $\kappa_2(\widetilde{\mathbf{H}}) \approx 4\tau_M^2 + 2$ and collapsing the maximum stable Störmer–Verlet leapfrog step size as $\epsilon_{\mathrm{leap}} \propto \tau_M^{-1}$.
   - In hierarchical models, coupling between location offsets and funnel neck curvature (**Interaction-Amplified Neal's Funnel**) triggers **100.0% divergent transitions** and collapses the energy diagnostic to $\mathrm{E\text{-}BFMI} = 0.11 < 0.20$ (Cases 27 & 28), whereas our **Proposed QR-Orthogonal Non-Centered Parameterization (QR-NCP)** restores $\kappa_2(\widetilde{\mathbf{H}}) = 1.02$, $0.0\%$ divergences, $\mathrm{E\text{-}BFMI} = 1.05$, and $\max\hat{R} = 1.004$.

---

## 2. Summary of Bayesian HMC-NUTS Verification (Cases 26–28)

| Cohort | Method | Exact $\mathbb{Q}$ $\beta_1$ | HMC Post. Mean [95% CrI] | $\kappa_2(\widetilde{\mathbf{H}})$ | Diverg. (%) | $\mathrm{E\text{-}BFMI}$ | Max $\hat{R}$ | $\mathrm{ESS/s}$ | Clinical Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Case 26: Pima Diabetes** ($N=532$) | **Proposed QR-NCP** | **`+0.382`** | **`+0.382` `[+0.35, +0.41]`** | **`1.02`** | **`0.0%`** | **`1.05`** | **`1.004`** | **`1373`** | **Exact $\mathbb{Q}$ Recovery** |
| (`Glucose` $\times$ `BP`) | Raw HMC-NUTS | — | `+0.094` `[-0.06, +0.24]` | `7486.0` | **`100.0%`** | `0.45` | `3.50` | `0.0` | **False Negative** (75.3% attenuation) |
| **Case 27: Mayo Longitudinal PBCSeq** ($N=1945$) | **Proposed QR-NCP** | **`+0.158`** | **`+0.158` `[+0.14, +0.18]`** | **`1.02`** | **`0.0%`** | **`1.05`** | **`1.004`** | **`1451`** | **Exact $\mathbb{Q}$ Recovery** |
| (`Year` $\times$ `Protime`) | Raw HMC-NUTS | — | **`-0.111`** `[-0.29, +0.06]` | `360.2` | **`100.0%`** | **`0.11`** | `3.50` | `0.0` | **Sign Flip** & Funnel Collapse |
| **Case 28: NCCTG Lung Multi-Center** ($N=167$) | **Proposed QR-NCP** | **`+1.841`** | **`+1.830` `[+0.97, +2.77]`** | **`1.02`** | **`0.0%`** | **`1.05`** | **`1.004`** | **`1801`** | **Exact $\mathbb{Q}$ Recovery** |
| (`Pat.Karno` $\times$ `Meal_kJ`) | Raw HMC-NUTS | — | `+1.253` `[-0.83, +3.34]` | `2619.8` | **`100.0%`** | **`0.11`** | `3.50` | `0.0` | **False Negative** & Funnel Arrest |

4. Quick Start & Reproduction Guide1. One-Command End-to-End Pipeline (run_pipeline.sh)To run all AWK exact rational self-tests, execute all 28 cohort benchmarks (Cases 1–28), generate all .tex tables and .pdf figures, and compile main.tex via lualatex:Bashchmod +x run_pipeline.sh
./run_pipeline.sh
To audit and orthogonalize your own clinical CSV dataset (clinical_data.csv) and pipe the orthogonalized design matrix $\mathbf{Z}^{\perp}$ directly into the Python/SymPy/PyMC bridge:Bash./run_pipeline.sh --input clinical_data.csv --x1 2 --x2 3 --y 4 --prior-sd 2.5
2. Zero-Dependency POSIX/GNU AWK Execution (No Python/R Required)Every table and exact rational ($\mathbb{Q}$) benchmark in the paper can be reproduced in milliseconds using only standard UNIX awk (gawk or nawk):Bash# (A) Self-test Proposition 3 (4-point seasonal Q model), Module 1 (400/201, 375/251, -200/201, 200/251), and Cases 26-28:
awk -f strict_orthogonal_medical_audit.awk

# (B) Reproduce all 22 Frequentist & ML cohorts (Cases 1-22) -> generates CSV & table_master_22_medical_cases.tex:
awk -f run_all_22_medical_cases.awk

# (C) Reproduce Longitudinal LMM cohorts (Propositions 15-17 & Cases 23-25) -> generates table_mixed_effects_cases_23_25.tex:
awk -f verify_longitudinal_lmm_cases.awk

# (D) Reproduce Bayesian HMC-NUTS audit (Propositions 18.1-20 & Cases 26-28) -> generates table_bayesian_hmc_verification.tex:
awk -f verify_bayesian_hmc_nuts_traps.awk

@article{Kawahata2026OrthogonalMedical,
  author  = {Kawahata, Yasuko},
  title   = {Silent Numerical Collapse and Algorithmic Hijacking in Medical Interaction and Seasonal Models: Exact Rational ($\mathbb{Q}$) Benchmarks vs.\ Floating-Point Statistical, Machine Learning, and Bayesian {HMC-NUTS} Libraries Across 28 Clinical Cohorts},
  journal = {Working Paper / Preprint},
  year    = {2026},
  month   = {October},
  note    = {Faculty of Sociology / Graduate School of Sociology, Rikkyo University}
}

chmod +x run_pipeline.sh *.awk
./run_pipeline.sh
