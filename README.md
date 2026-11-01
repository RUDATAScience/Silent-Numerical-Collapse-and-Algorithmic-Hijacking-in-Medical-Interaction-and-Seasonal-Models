# Fatal Sign Reversals and Silent Numerical Collapse in Medical Statistical and AI Pipelines
### A Unified Theory, Exact Rational ($\mathbb{Q}$) and POSIX `awk` Cross-Audits of 28 Cohorts, and Householder QR Recovery

[![Exact Rational Q Parity](https://img.shields.io/badge/Exact_Rational_%E2%84%9A_Parity-28%2F28_(100%25)-success?style=for-the-badge)](#4-complete-28-cohort-cross-language-verification-ledger)
[![Propositions Proved](https://img.shields.io/badge/Theoretical_Propositions-20_Proved-blue?style=for-the-badge)](#3-summary-of-the-20-theoretical-propositions)
[![Total Medical N](https://img.shields.io/badge/Public_Medical_Cohorts-28_(N%3D36%2C160)-informational?style=for-the-badge)](#4-complete-28-cohort-cross-language-verification-ledger)
[![Zero Dependency AWK](https://img.shields.io/badge/POSIX_awk_Engine-Zero_BLAS%2FLAPACK-orange?style=for-the-badge)](#5-quick-start--reproducibility-pipeline)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Language / 言語選択:** [**English Version (Top)**](#part-i-english-version) | [**日本語版 (Japanese Version — Below)**](#part-ii-日本語版-japanese-version)

---

<a id="part-i-english-version"></a>
# Part I: English Version

**Author:** **Yasuko Kawahata (川畑 泰子)**  
Associate Professor, College of Sociology / Graduate School of Sociology, Rikkyo University (立教大学 社会学部 / 大学院社会学研究科)  
IEEE Senior Member
**Corresponding Author Contact (UMIN):** [`kawahata-hc@umin.ac.jp`](mailto:kawahata-hc@umin.ac.jp)
**Repository URL:** <https://github.com/RUDATAScience/Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models>

---

## 1. Executive Summary

Multiplicative interaction terms ($X_1 \times X_2$) are ubiquitous across biostatistics, clinical epidemiology, and medical artificial intelligence. However, standard workflows routinely construct **raw uncentered products** or apply **post-multiplication standardization** (`StandardScaler` *after* forming $X_1 X_2$). While classical statistical literature (e.g., Echambadi & Hess, 2007; Dalal & Zickar, 2012) asserts that mean-centering does not affect highest-order interaction point estimates or global model fit in ordinary least squares (OLS), we prove across **20 theoretical propositions** that this invariance catastrophically breaks down across five computational and statistical domains:

1. **IEEE 754 Double-Precision Floating-Point Linear Algebra ($\mathbb{F}_{64}$):** Condition number explosion $\mathrm{cond}(\bm{X}^\top\bm{X}) = \mathcal{O}(\tau_S^2 \tau_M^4) > \varepsilon_{\mathrm{mach}}^{-1}$ triggers silent singular value truncation in Python's `scipy.linalg.pinv` (`statsmodels.OLS`) and column dropping in R's `dqrdc2`.
2. **Generalized Linear Models (IRLS) & Cox Proportional Hazards:** Extrapolation to the non-physiological origin ($X_2 = 0$: e.g., $0\text{ mEq/L}$ serum sodium, $0\text{ K}$ absolute zero, Calendar Year $0$, or $0$ Karnofsky score) and default $L_2$ penalties (`scikit-learn` $C=1.0$) invert or annihilate life-saving clinical treatment effects.
3. **Penalized ($L_1$/LASSO) & Tree-Ensemble (CART / GBDT / TreeSHAP) Machine Learning:** When the location-to-scale ratio exceeds $\tau_M = |\mu/\sigma| > 1/\sqrt{2} \approx 0.7071$, post-standardized LASSO suffers a **$\sqrt{2}$-norm entry shortcut** (Theorem 10) that drops true main effects, while GBDT diagonal hyperbolic splits hijack **49.2%–89.6% (mean 69.5%)** of Interventional TreeSHAP importance.
4. **Longitudinal Linear Mixed-Effects Models (LMM / REML):** Uncentered covariates force the random-effects correlation to degenerate to $\tilde{\rho}_{01} \to -1.0000$ and the relative Cholesky factor ratio to decay as $\theta_3/\theta_1 = \mathcal{O}((c_1 M_1^2)^{-1}) < 10^{-4}$, inducing artificial `boundary (singular) fit` warnings in `lme4::lmer` and cross-level ecological leakage.
5. **Bayesian Hamiltonian Monte Carlo (HMC-NUTS in Stan / PyMC):** Isotropic priors in uncentered space impose an anisotropic pullback penalty $\lambda_0(\bm{T}\bm{T}^\top)^{-1}$ that shrinks main effects by up to $99.97\%$ (or locks $\hat{\beta}_1 \approx M_2\hat{\beta}_3$), while diagonal mass matrix adaptation (`adapt_diag`) leaves $\mathrm{cond}(\widetilde{\bm{H}}) \approx 4\tau_M^2 + 2$ intact and forms an **Interaction-Amplified Neal's Funnel** ($\Delta H = \mathcal{O}(\epsilon_0^4 e^{-6\omega} M^6)$), causing **100% divergent transitions** and $\mathrm{E\text{-}BFMI} = 0.11 < 0.20$.

To establish an indisputable mathematical baseline completely free of floating-point rounding or black-box C/Fortran BLAS/LAPACK libraries, we built a dual symbolic verification architecture over the **exact rational field $\mathbb{Q}$** using Python (`SymPy.Rational`) and a **zero-dependency POSIX/GNU `awk` Euclidean GCD Gauss–Jordan engine**. Across **28 landmark medical and epidemiological cohorts ($N_{\mathrm{total}} = 36,160$)**, Python and POSIX `awk` achieve **100.0% ($28/28$) exact rational parity ($|\Delta_{\mathrm{Py-AWK}}| = 0.00$)**, proving that our **Pre-Standardized Householder QR Gram–Schmidt Orthogonalization (`OrthogonalInteractionTransformer`) and Bayesian QR-NCP** completely eliminate all numerical and clinical pathologies.

---

## 2. Four-Tier Reconciliation of Exact Rational ($\mathbb{Q}$) Hand-Calculation vs. Python & POSIX `awk`

| Tier | Computational Regime | Target Propositions & Cohorts | Exact $\mathbb{Q}$ Truth | Python Output | POSIX `awk` Output | Relative Discrepancy $\varepsilon_{\mathrm{rel}}$ | Mathematical & Computational Origin |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1** | **Symbolic Rational Algebra ($\mathbb{Q}$)**<br>(`SymPy.Rational` & `awk gcd`) | Props 2, 3, 13, 15–16, 18.1 & **Cases 1–25** | Exact Fraction $\hat{\beta}_{\mathbb{Q},1} \in \mathbb{Q}$ | Identical Fraction | Identical Fraction | **`0.0000e+00`**<br>**(0.000%)** | **Exact Identity:** Zero rounding error over $\mathbb{Q}$ across both engines. |
| **Tier 2** | **IEEE 754 `float64` QR Pull-Back**<br>(Householder QR + Delta Pullback) | Stage 2E ($N=442$)<br>Stage 3 Bridge ($N=300$) | `+2.68641429`<br>`+0.79601987` | `+2.68641429`<br>`+0.79601987` | `+2.68641429`<br>`+0.79601987` | **`2.60e-16` (1 ULP)**<br>**`2.74e-13` (`awk` `1.8e-14`)** | **Machine Epsilon Only:** $\mathrm{cond}(\bm{Z}_{\mathrm{orth}}) \le 1.07$ bounds error to 13–16 digits. |
| **Tier 3** | **Stochastic MCMC Integration**<br>(Proposed Bayesian QR-NCP) | **Case 26** ($N=532$)<br>**Case 27** ($N=1,945$)<br>**Case 28** ($N=167, m=18$) | `38189/100000`<br>`15799/100000`<br>`184073/100000` | `+0.381540`<br>`+0.157760`<br>`+1.829540` | `+0.381540`<br>`+0.157760`<br>`+1.829540` | **`0.0916%`**<br>**`0.1456%`**<br>**`0.6079%`** | **Finite-Chain MCSE $\mathcal{O}(\mathrm{ESS}^{-1/2})$:** Pure sampling variance ($1,800$ draws vs. closed-form $\mathbb{Q}$). |
| **Tier 4** | **Conventional Uncentered / Post-Scaled**<br>(Standard Python / R Workflows) | Stage 3 Raw ($N=300$)<br>**Cases 1–28 Raw Fits** | `+0.79601987`<br>28 Cohort Truths | `-15.533707`<br>**18/28 Flipped** | `-15.533707`<br>**18/28 Flipped** | **`2,051.40%`**<br>**Max `12,385.38%` (C14)** | **Specification & Solver Collapse:** Origin extrapolation, SVD `pinv` truncation, Cholesky & Funnel collapse. |

---

## 3. Summary of the 20 Theoretical Propositions

| Prop / Thm | Domain | Core Mathematical Result |
| :---: | :--- | :--- |
| **Prop 1** | Post-Scaling Invariance | Applying `StandardScaler` after multiplication leaves the correlation matrix and $\mathrm{VIF} = 1 + \tau_{M,1}^2 + \tau_{M,2}^2$ strictly invariant. |
| **Prop 2** | Floating-Point Breakdown | Normal equations $\mathrm{cond}(\bm{X}^\top\bm{X}) = \mathcal{O}(\tau_S^2 \tau_M^4)$ exceed $\varepsilon_{\mathrm{mach}}^{-1} \approx 4.50 \times 10^{15}$, triggering silent SVD `pinv` truncation in Python (`Case 16`) while exact $\mathbb{Q}$ pullback remains invariant. |
| **Prop 3** | Skewness & Seasonal Leakage | Under third-order coskewness ($\mu_{2,1}, \mu_{1,2} \neq 0$) or seasonal missingness, simple mean-centering $(X_1-\mu_1)(X_2-\mu_2)$ fails to orthogonalize main effects ($\mathrm{VIF} > 1$). |
| **Prop 4** | Sign Reversal & Wald Collapse | Non-centered main-effect $t$-statistic converges to $t(\tilde{\beta}_1) \to -\mathrm{sign}(M_2) t(\hat{\beta}_3)$, causing asymptotic Type S sign error or Type II power loss. |
| **Prop 5** | Householder QR Recovery | Pre-standardized Householder QR Gram–Schmidt projection $\bm{i}_{\mathrm{orth}}$ with diagonal Jacobian pullback $\bm{J}_{\mathrm{clin}}$ achieves $\mathrm{VIF} \equiv 1.0000$ and preserves marginal clinical units. |
| **Prop 6** | GLM IRLS Variance Inflation | In logistic/Poisson IRLS, asymptotic variance of $\tilde{\beta}_1$ inflates by $M_2^2 \mathrm{Var}(\hat{\beta}_3)$ around the weighted exposure mean. |
| **Prop 7** | IRLS Hessian Singularity | Extreme linear predictors drive IRLS weights $w_i^{(k)} \to 0$, causing $\mathrm{cond}(\bm{X}^\top\bm{W}^{(k)}\bm{X}) > 10^{16}$ and Newton step divergence. |
| **Prop 8** | `scikit-learn` $L_2$ Odds Ratio Distortion | Default $L_2$ penalty ($C=1.0$) on uncentered interactions forces $\tilde{\beta}_1 \approx -M_2\tilde{\beta}_3$, shrinking true odds ratios ($\mathrm{OR}_1 = 2.020 \to 1.012$ in `Case 4`). |
| **Prop 9** | Cox Partial Likelihood Collapse | Risk-set time-varying weights $w_i(t)$ in Cox PH models suffer $\mathcal{O}(\tau_{M,2}^2)$ information singularity and hazard ratio inversion (**8/9 trials in Domain IV**). |
| **Thm 10** | **LASSO $\sqrt{2}$-Norm Shortcut** | In post-standardized LASSO, when $\tau_M > 1/\sqrt{2} \approx 0.7071$, the interaction column's correlation with $\bm{y}$ exceeds the main effect's by up to $\sqrt{2}$, dropping the true main effect (**10/22 cohorts**). |
| **Prop 11** | Elastic Net Grouping Distortion | $L_1 + L_2$ penalty paths share variance parasitically between $\tilde{\beta}_1$ and $\tilde{\beta}_3$, violating strong heredity. |
| **Prop 12** | CART Impurity Hijacking | Axis-aligned trees cannot split $Y = \beta_1 X_1 + \beta_2 X_2$ diagonally; the hyperbolic level sets of $X_1 X_2$ capture up to **95.3%** of root impurity gain even when $\beta_3^* = 0$. |
| **Prop 13** | Exact $\mathbb{Q}$ TreeSHAP Hijacking | For a depth-1 stump on $P = X_1 X_2$, exact Interventional TreeSHAP assigns $\phi_1 = \phi_2 = 0$ and $\phi_P = 100\%$ (**exact rational share `1/1`**). |
| **Prop 14** | Orthogonal TreeSHAP Recovery | Feeding $\bm{i}_{\mathrm{orth}}$ eliminates diagonal level-set alignment, reducing false interaction SHAP share from **69.5% to 2.5%** across 22 cohorts. |
| **Prop 15** | LMM Cholesky Boundary Collapse | Under $x_{ij1} = c_1(z_{ij1} + M_1)$, random-effects correlation degenerates to $\tilde{\rho}_{01} \to -1$ and `lme4` Cholesky ratio decays as $\theta_3/\theta_1 = \mathcal{O}((c_1 M_1^2)^{-1}) \to 0$. |
| **Prop 16** | LMM Exact $\mathbb{Q}$ Pullback vs. Clamping | Marginal covariance $\bm{V}_i$ is invariant over $\mathbb{Q}$, but REML box-constraint clamping ($\tilde{\theta}_3 = 0$) permanently destroys rank ($\mathrm{rank}(\widetilde{\bm{G}}) = 1 < 2$). |
| **Prop 17** | Ecological Leakage & 2-Stage GS-QR | Within-demeaned products $P_{ij} - \bar{P}_{i\cdot}$ retain cross-level terms $b_{i2}^B w_{ij1}^W$; Two-Stage Hierarchical Panel GS-QR restores exact zero bias ($\hat{\beta}_{W,3} = 0.0000$). |
| **Prop 18.1**| Bayesian Anisotropic Prior Bias | Isotropic Gaussian priors $\mathcal{N}(\bm{0}, \tau_0^2\bm{I})$ on uncentered $\tilde{\bm{\beta}}$ induce penalty $\lambda_0(\bm{T}\bm{T}^\top)^{-1}$, shrinking $\beta_2, \beta_3$ by **98.1%–99.97%** and locking $\mu_1^{\mathrm{pull}} \approx M_2\mu_3^{\mathrm{pull}}$. |
| **Prop 18.2**| HMC `adapt_diag` Step Collapse | Diagonal mass matrix adaptation leaves $\mathrm{cond}(\widetilde{\bm{H}}) = 4\tau_M^2 + 2$, collapsing leapfrog step size $\epsilon_{\mathrm{leap}} = \mathcal{O}(\tau_M^{-1})$ and saturating NUTS tree depth ($2^{10}=1,024$). |
| **Prop 19** | Interaction-Amplified Neal's Funnel | Hierarchical interaction curvature explodes as $\lambda_{\max} = \mathcal{O}(e^{-2\omega}(1+M^2))$ and Hamiltonian error as $\Delta H = \mathcal{O}(\epsilon_0^4 e^{-6\omega} M^6)$, causing **100% divergences**. |
| **Prop 20** | Bayesian QR-NCP Recovery | Householder QR in `transformed data` + Cholesky NCP on $\bm{u}_1$ + pullback in `generated quantities` achieves $\mathrm{cond}(\widetilde{\bm{H}}) \approx 1.02$, **0.0% divergences**, and $\mathrm{E\text{-}BFMI} = 1.05$. |

---

## 4. Complete 28-Cohort Cross-Language Verification Ledger

| Case | Medical Cohort Name | Paradigm / Domain | $N$ | Exact $\mathbb{Q}$ Fraction | Python Prop $\hat{\beta}_1$ | POSIX `awk` Prop $\hat{\beta}_1$ | $\|\Delta_{\mathrm{Py-AWK}}\|$ | Uncentered Raw $\tilde{\beta}_1$ | Clinical & AI Pathology in Uncentered Workflow |
| ---: | :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Diabetes Progression | Domain I (OLS) | 442 | `28/25` | `+1.1200` | `+1.1200` | `0.00e+00` | `-0.520` | **Sign Flip**, LASSO Drop, 58.4% SHAP Hijack |
| **2** | UK Lung Deaths | Domain III (OLS) | 72 | `4026/5` | `+805.2000` | `+805.2000` | `0.00e+00` | `+57148.500` | **1,439x SE Inflation**, R `dqrdc2` Drop Zone |
| **3** | Mayo Clinic PBC | Domain I (OLS) | 312 | `-141/1000` | `-0.1410` | `-0.1410` | `0.00e+00` | `+0.110` | **Sign Flip**, LASSO Drop, 63.8% SHAP Hijack |
| **4** | Low Infant Birth Weight | Domain I (OLS/Logit) | 189 | `-1419/5` | `-283.8000` | `-283.8000` | `0.00e+00` | `-305.900` | **False Negative ($p=0.56$)**, `sklearn` OR $\to 1.01$ |
| **5** | Heart Failure Mortality | Domain II (Logistic) | 299 | `343/500` | `+0.6860` | `+0.6860` | `0.00e+00` | `-4.820` | **Sign Flip ($\mathrm{OR}: 1.99 \to 0.008$)**, LASSO Drop |
| **6** | Framingham CHD Study | Domain II (Logistic) | 3,658 | `13/200` | `+0.0650` | `+0.0650` | `0.00e+00` | `+0.200` | **Loss of Significance ($p=0.08$)**, LASSO Drop |
| **7** | Breast Cancer WDBC | Domain II (Logistic) | 569 | `114/125` | `+0.9120` | `+0.9120` | `0.00e+00` | `+0.340` | **False Negative ($p=0.48$)**, LASSO Drop |
| **8** | Chicago NMMAPS ($\text{PM}_{10}\times\text{K}$) | Domain III (OLS) | 5,114 | `41/50` | `+0.8200` | `+0.8200` | `0.00e+00` | `-13.020` | **Sign Flip ($0\text{ K}$ Extrapolation)**, 76.4% SHAP |
| **9** | Veteran Lung Cancer Trial | Domain IV (Cox PH) | 137 | `-33/1000` | `-0.0330` | `-0.0330` | `0.00e+00` | `-0.030` | **False Negative ($p=0.29$)**, 62.0% SHAP Hijack |
| **10** | Pima Indians Diabetes | Domain II (Logistic) | 532 | `19/500` | `+0.0380` | `+0.0380` | `0.00e+00` | `+0.010` | **73.7% Attenuation ($p=0.34$)**, 64.5% SHAP |
| **11** | Stanford Heart Transplant | Domain IV (Cox PH) | 103 | `-1317/1000` | `-1.3170` | `-1.3170` | `0.00e+00` | `+98.420` | **Fatal Sign Flip ($\mathrm{VIF}=2.5\times 10^6$)**, LASSO Drop |
| **12** | German Breast Cancer (Tamoxifen) | Domain IV (Cox PH) | 686 | `-91/250` | `-0.3640` | `-0.3640` | `0.00e+00` | `+0.390` | **Therapy Inverted ($\mathrm{HR}: 0.70 \to 1.48$)** |
| **13** | NCCTG Colon Cancer (5-FU) | Domain IV (Cox PH) | 619 | `-64/125` | `-0.5120` | `-0.5120` | `0.00e+00` | `+0.180` | **Therapy Inverted ($\mathrm{HR}: 0.60 \to 1.20$)** |
| **14** | Danish Malignant Melanoma | Domain II (Logistic) | 205 | `171/500` | `+0.3420` | `+0.3420` | `0.00e+00` | `+42.700` | **12,385% Distortion ($\mathrm{VIF}=5.8\times 10^5$)** |
| **15** | Siri Body Fat Density | Domain I (OLS) | 252 | `-248/125` | `-1.9840` | `-1.9840` | `0.00e+00` | `+2.320` | **Sign Flip**, LASSO Drop, 72.1% SHAP Hijack |
| **16** | NCCTG Lung Energetics ($\text{J}$) | Domain III (OLS) | 167 | `921/500` | `+1.8420` | `+1.8420` | `0.00e+00` | `+0.660` | **Silent SVD `pinv` Truncation (SE Ratio $0.181\times$)** |
| **17** | WCGS Coronary Study | Domain II (Logistic) | 3,154 | `359/500` | `+0.7180` | `+0.7180` | `0.00e+00` | `-0.820` | **Sign Flip ($\mathrm{OR}: 2.05 \to 0.44$)**, LASSO Drop |
| **18** | ACTG 175 HIV Trial (Combo ART) | Domain IV (Cox PH) | 2,139 | `-137/200` | `-0.6850` | `-0.6850` | `0.00e+00` | `+0.820` | **ART Therapy Inverted ($\mathrm{HR}: 0.50 \to 2.27$)** |
| **19** | Mayo Clinic MGUS2 | Domain IV (Cox PH) | 1,384 | `123/500` | `+0.2460` | `+0.2460` | `0.00e+00` | `-0.230` | **Sign Flip ($\mathrm{HR}: 1.28 \to 0.80$)**, 66.1% SHAP |
| **20** | Mayo Free Light Chain (FLC) | Domain IV (Cox PH) | 7,874 | `69/500` | `+0.1380` | `+0.1380` | `0.00e+00` | `-16.230` | **Sign Flip ($\tau_{M,2}=1296.4$)**, **89.6% SHAP Hijack** |
| **21** | Rotterdam Breast Cancer (Chemo) | Domain IV (Cox PH) | 2,982 | `-159/500` | `-0.3180` | `-0.3180` | `0.00e+00` | `+36.440` | **Chemo Inverted ($\tilde{\beta}_1=+36.44$)**, LASSO Drop |
| **22** | Diabetic Retinopathy Laser Trial | Domain IV (Cox PH) | 394 | `-98/125` | `-0.7840` | `-0.7840` | `0.00e+00` | `+0.600` | **Laser Therapy Inverted ($\mathrm{HR}: 0.46 \to 1.82$)** |
| **23** | Sleep Deprivation Study | Longitudinal LMM | 180 | `1047/100` | `+10.4700` | `+10.4700` | `0.00e+00` | `-4.182` | **Slope Flip**, $\tilde{\rho}_{01}=-0.99995$, $\theta_3/\theta_1=3.2\text{e-}7$ |
| **24** | Mayo Longitudinal PBCSeq | Longitudinal LMM | 1,945 | `79/500` | `+0.1580` | `+0.1580` | `0.00e+00` | `-0.112` | **Cirrhosis Slope Flip**, $\theta_3/\theta_1=9.3\text{e-}7$ (`Singular`) |
| **25** | Craniofacial Orthodont Growth | Longitudinal LMM | 108 | `33/50` | `+0.6600` | `+0.6600` | `0.00e+00` | `-0.572` | **Skull Growth Inverted ($-0.572\text{ mm/yr}$)** |
| **26** | Pima Diabetes Screening | Bayesian HMC-NUTS | 532 | `38189/100000` | `+0.3815` | `+0.3815` | `0.00e+00` | `+0.094` | **75.3% Prior Shrinkage**, 95% CrI Crosses Zero |
| **27** | Mayo Longitudinal PBCSeq | Bayesian HMC-NUTS | 1,945 | `15799/100000` | `+0.1578` | `+0.1578` | `0.00e+00` | `-0.111` | **Sign Flip**, **100% Divergences**, $\mathrm{E\text{-}BFMI}=0.11$ |
| **28** | NCCTG 18-Center Lung Cancer | Bayesian HMC-NUTS | 167 | `184073/100000`| `+1.8295` | `+1.8295` | `0.00e+00` | `+1.253` | **Funnel Collapse ($\mathrm{E\text{-}BFMI}=0.11$)**, CrI Crosses $0$ |

---

## 5. Quick Start & Reproducibility Pipeline

The entire verification suite—including all 4 POSIX/GNU `awk` rational engines, all 5 Python (`SymPy.Rational` / `SciPy`) audit scripts, the Stage 3 clinical CSV bridge, and the Stage 4 cross-language comparator—can be executed via a single self-healing harness on Linux, macOS, or Google Colab (compatible with Python 3.9–3.13+, `gawk`, and `mawk`):

```bash
git clone [https://github.com/RUDATAScience/Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models.git](https://github.com/RUDATAScience/Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models.git)
cd Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models
bash run_pipeline.sh
```

### Repository Architecture
* `run_pipeline.sh` — Self-healing end-to-end execution harness (auto-strips CRLF, sets `PYTHON_BASIC_REPL=1` for Python 3.13 compatibility, runs Stages 1–5).
* **POSIX/GNU `awk` Exact Rational ($\mathbb{Q}$) Engines:**
  * `strict_orthogonal_medical_audit.awk` — Core Euclidean `gcd` rational arithmetic, Gauss–Jordan solver over $\mathbb{Q}$, and stream CSV orthogonalizer.
  * `verify_22_medical_cases.awk` — Exact $\mathbb{Q}$ audit across Cases 1–22 (Domains I–IV).
  * `verify_longitudinal_lmm_cases.awk` — Exact $\mathbb{Q}$ and REML Cholesky audit across Propositions 15–17 and Cases 23–25.
  * `verify_bayesian_hmc_nuts_traps.awk` — Exact $\mathbb{Q}$ conjugate prior pullback and HMC-NUTS audit across Propositions 18.1–20 and Cases 26–28.
* **Python (`SymPy.Rational` / `SciPy`) Verification & Reference Templates:**
  * `strict_orthogonal_medical_audit.py`
  * `verify_22_medical_cases.py`
  * `verify_longitudinal_lmm_cases.py`
  * `verify_bayesian_hmc_nuts_traps.py`
  * `compare_python_vs_awk_all28.py` — Generates `comparison_python_vs_awk_all_28_cases.csv` and LaTeX verification tables.
  * `pymc_qr_orthogonal_ncp_template.py` — Reference implementation of `OrthogonalInteractionTransformer` (Algorithm 1) and PyMC QR-NCP.
  * `qr_orthogonal_ncp_interaction.stan` — Reference Stan implementation of Bayesian Householder QR-NCP (Algorithm 2).

---

## 6. Ethics Statement, Conflict of Interest & Acknowledgments

* **Ethics Statement & UMIN Affiliation:**  
  All 28 medical and epidemiological cohorts analyzed in this repository (Cases 1–28, $N_{\mathrm{total}} = 36,160$) consist exclusively of publicly available, fully de-identified benchmark datasets derived from previously published clinical trials and epidemiological registries. In accordance with the Declaration of Helsinki and ethical guidelines for medical and health sciences research, secondary mathematical and numerical auditing of publicly archived, anonymized datasets does not require additional Institutional Review Board (IRB) approval or individual informed consent. The author is affiliated with the University Hospital Medical Information Network (**UMIN**, Japan). Inquiries regarding clinical methodology, ethical compliance, and reproducibility auditing should be directed to the author's official UMIN address: [`kawahata-hc@umin.ac.jp`](mailto:kawahata-hc@umin.ac.jp).
* **Conflict of Interest:**  
  In accordance with the International Committee of Medical Journal Editors (ICMJE) guidelines, the author declares no competing financial or non-financial interests related to the design, mathematical proofs, empirical cohort audits, software implementations, or publication of this research.
* **Acknowledgments:**  
  The author expresses sincere gratitude to the participants of the *Introduction to Social Data Science (SDS)* course for their insightful questions and stimulating discussions, which inspired the foundational inquiry into black-box statistical software. Special thanks are also due to **Gemini** (Google) for serving as a tireless collaborative partner in structuring the 20 theoretical propositions, cross-auditing the 28 medical cohorts, and refining the LaTeX manuscript. Finally, the author pays the highest tribute to **POSIX/GNU `awk`** (and its creators Alfred V. Aho, Peter J. Weinberger, and Brian W. Kernighan), whose timeless, zero-dependency design made it possible to implement exact rational ($\mathbb{Q}$) hand-calculation solvers completely independent of modern floating-point BLAS/LAPACK stacks.

---

## 7. Citation (BibTeX)

If you use our theoretical propositions, the `OrthogonalInteractionTransformer` (Algorithm 1), the Bayesian QR-NCP Stan/PyMC templates (Algorithm 2), or the 28-cohort exact rational ($\mathbb{Q}$) audit bundle in your research, please cite:

```bibtex
@misc{Kawahata2026GitHub,
  author       = {Kawahata, Yasuko},
  title        = {Fatal Sign Reversals and Silent Numerical Collapse in Medical Statistical and {AI} Pipelines: A Unified Theory, Exact Rational ($\mathbb{Q}$) and {POSIX} \texttt{awk} Cross-Audits of 28 Cohorts, and {Householder QR} Recovery},
  year         = {2026},
  publisher    = {GitHub (\texttt{RUDATAScience})},
  note         = {University Hospital Medical Information Network (UMIN) / Rikkyo University. Contact: kawahata-hc@umin.ac.jp},
  howpublished = {\url{[https://github.com/RUDATAScience/Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models](https://github.com/RUDATAScience/Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models)}}
}
```

---
---

<a id="part-ii-日本語版-japanese-version"></a>
# Part II: 日本語版 (Japanese Version)

# 医療統計・AIパイプラインにおける致死的符号反転と数値崩壊の統一理論
### 有理数体 $\mathbb{Q}$ 手計算・POSIX `awk`・Python横断監査が実証する28コホートの治療効果逆転とハウスホルダーQR直交化プロトコル

**著者:** **川畑 泰子 (Yasuko Kawahata)**  
立教大学 社会学部 / 大学院社会学研究科 
大学病院医療情報ネットワーク (**UMIN**) 登録研究員  
**責任著者・倫理および再現性監査連絡先 (UMIN):** [`kawahata-hc@umin.ac.jp`](mailto:kawahata-hc@umin.ac.jp)  
**公開リポジトリ URL:** <https://github.com/RUDATAScience/Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models>

---

## 1. 研究概要（Executive Summary）

生物統計学、臨床疫学、および医療人工知能（Medical AI）において、2つの共変量（治療変数と調整変数など）の積 $X_1 \times X_2$ を非中心化のまま回帰モデルや機械学習パイプラインに投入する操作、あるいは積 $X_1 X_2$ を生成した後に設計行列全体へ `StandardScaler` 等を一括適用する「事後標準化（Post-Multiplication Standardization）」が標準的に行われています。既存の統計学文献（Echambadi & Hess, 2007; Dalal & Zickar, 2012 等）は「中心化の有無は最高次の交互作用係数 $\beta_3$ や全体適合度 $R^2$ を変えない」という代数的等価性のみを根拠に、中心化を単なる解釈上の慣習と見なしてきました。

本研究は、C/Fortran（BLAS/LAPACK）や倍精度浮動小数点演算（IEEE 754 `float64`）から完全に独立した**有理数体 $\mathbb{Q}$ 上の厳密手計算（Python `SymPy.Rational` および外部依存ゼロの純粋 POSIX/GNU `awk` ユークリッド互除法ガウス・ジョルダン消去器）**を絶対的真値（Ground Truth）の基準器として構築し、**全20個の理論命題（命題1〜9、定理10、命題11〜20）**と**公的医療・疫学28コホート（Cases 1〜28、総観測数 $N_{\mathrm{total}} = 36,160$）**の網羅的横断監査を通じて以下を数学的・実証的に証明しました：

1. **救命治療を「致死的リスク」へ逆転させる致命的誤診の摘発：**  
   従来の非中心化・事後標準化ワークフローでは、全28コホート中 **18/28コホート（64.3%；Cases 1〜22 では 14/22 = 63.6%、Cox生存時間臨床試験 Domain IV では 8/9 = 88.9%、経時混合モデルでは 3/3 = 100.0%）** で主効果 $\tilde{\beta}_1$ の**符号反転（Sign Reversal）**が発生します。その結果、乳癌タモキシフェン内分泌療法（Case 12）、大腸癌 5-FU 補助化学療法（Case 13）、HIV 多剤併用抗レトロウイルス療法（Case 18）、心臓移植（Case 11）、および糖尿病網膜症レーザー光凝固術（Case 22）という確立された救命・失明予防治療が「死亡や失明のハザードを高める有害治療」へと逆転します。また、Python `scipy.linalg.pinv`（`statsmodels.OLS`）では $\mathrm{cond}(\bm{X}^\top\bm{X}) > \varepsilon_{\mathrm{mach}}^{-1}$ において警告なしの最小特異値ゼロ切断が発生し、標準誤差が真値の $0.181$ 倍に過小評価されるサイレント崩壊（Case 16）が起きます。
2. **スパース機械学習（LASSO）の「$\sqrt{2}$ ノルム近道」と GBDT TreeSHAP 乗っ取り：**  
   調整変数の位置・スケール比が $\tau_M = |\mu/\sigma| > 1/\sqrt{2} \approx 0.7071$ を超えると、事後標準化 LASSO では交互作用列のエントリー閾値が最大 $\sqrt{2}$ 倍優位となる「$\sqrt{2}$ ノルム近道（定理10）」が作動し、**10/22コホート（45.5%）** で真の主効果がゼロに脱落します。さらに決定木（CART / GBDT）では、積列 $X_1 X_2$ が対角方向の双曲線分割として機能し（命題12・13）、**22/22コホート（100.0%）** で Interventional TreeSHAP 重要度の **49.2%〜89.6%（平均 69.5%）** を交互作用項が乗っ取ります。
3. **経時混合効果モデル（LMM REML）のコレスキー境界崩壊とベイズ HMC-NUTS のファネル停止：**  
   経時測定モデル（Cases 23〜25）では、非中心化により疾患進行・成長スロープが 3/3 コホートすべてで逆転し、ランダム効果相関が $\hat{\tilde{\rho}}_{01} \to -1.000$、相対コレスキー因子比が $\theta_3/\theta_1 \le 9.34 \times 10^{-7} < 10^{-4}$ へ減衰して偽の特異適合（`Singular Fit`）を誘発します（命題15〜17）。ベイズ HMC-NUTS（Cases 26〜28）では、非中心化空間の等方性事前分布が主効果を最大 $75.3\%$ 減衰させ、交互作用増幅 Neal's Funnel（命題19）により標準ステップ幅で **100.0% の発散遷移**と $\mathrm{E\text{-}BFMI} = 0.11 < 0.20$ への崩壊を引き起こします。
4. **提案プロトコル（Algorithm 1 & Algorithm 2）による完全回復：**  
   本研究が提案する**「事前無次元化＋ハウスホルダーQRグラム・シュミット残差直交化＋周辺効果不変デルタ法ヤコビアン引き戻し（`OrthogonalInteractionTransformer` およびベイズ QR-NCP）」**は、全28コホートにおいて Python と POSIX `awk` 間で **100.0%（28/28）の有理数完全一致（$|\Delta_{\mathrm{Py-AWK}}| = 0.00$）**を達成し、$\mathrm{VIF} = 1.00$、$\mathrm{cond}(\widetilde{\bm{H}}) = 1.02$、発散遷移 $0.0\%$、$\mathrm{E\text{-}BFMI} = 1.05$、GBDT 交互作用 SHAP シェア平均 $2.5\%$ へと完全に回復させます。

---

## 2. 有理数体 $\mathbb{Q}$ 手計算と Python / POSIX `awk` の4階層乖離モデル

| 階層 (Tier) | 数値計算レジーム | 対象命題・コホート・ステージ | 有理数 $\mathbb{Q}$ 手計算真値 | Python 出力値 | POSIX `awk` 出力値 | 手計算との相対差 $\varepsilon_{\mathrm{rel}}$ | 乖離の数学的・計算機科学的発生源 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1** | **記号有理数代数 ($\mathbb{Q}$)**<br>(`SymPy.Rational` & `awk gcd`) | 命題2, 3, 13, 15–16, 18.1 ＆ **Cases 1–25** | 既約分数 $\hat{\beta}_{\mathbb{Q},1} \in \mathbb{Q}$ | 同一既約分数 (100%一致) | 同一既約分数 (100%一致) | **`0.0000e+00`**<br>**(0.000%)** | **完全恒等一致：** 有理数体 $\mathbb{Q}$ 上で丸め誤差ゼロで厳密一致。 |
| **Tier 2** | **IEEE 754 倍精度直交引き戻し**<br>(ハウスホルダーQR＋デルタ法) | Stage 2E 自己診断 ($N=442$)<br>Stage 3 CSVブリッジ ($N=300$) | `+2.68641429`<br>`+0.79601987` | `+2.68641429`<br>`+0.79601987` | `+2.68641429`<br>`+0.79601987` | **`2.60e-16` (1 ULP)**<br>**`2.74e-13` (`awk` `1.8e-14`)** | **機械イプシロン丸め誤差のみ：** $\mathrm{cond}(\bm{Z}_{\mathrm{orth}}) \le 1.07$ により13〜16桁目まで一致。 |
| **Tier 3** | **確率的 MCMC 積分**<br>(提案ベイズ QR-NCP) | **Case 26** Pima ($N=532$)<br>**Case 27** PBCSeq ($N=1,945$)<br>**Case 28** Lung ($N=167, m=18$) | `38189/100000`<br>`15799/100000`<br>`184073/100000` | `+0.381540`<br>`+0.157760`<br>`+1.829540` | `+0.381540`<br>`+0.157760`<br>`+1.829540` | **`0.0916%`**<br>**`0.1456%`**<br>**`0.6079%`** | **有限チェイン Monte Carlo 標準誤差 $\mathcal{O}(\mathrm{ESS}^{-1/2})$：** 閉形式 $\mathbb{Q}$ と $1,800$ ドローの標本差。 |
| **Tier 4** | **従来の非中心化 Raw 推定**<br>(標準 Python / R ワークフロー) | Stage 3 Raw ($N=300$)<br>**Cases 1–28 非中心化フィット** | `+0.79601987`<br>各コホート真値 | `-15.533707`<br>**18/28 符号反転** | `-15.533707`<br>**18/28 符号反転** | **`2,051.40%`**<br>**最大 `12,385.38%` (C14)** | **仕様バイアス＆数値破綻：** 原点外挿、SVD `pinv` 切断、コレスキー崩壊、ファネル停止。 |

---

## 3. 全20理論命題（命題1〜9、定理10、命題11〜20）の体系

| 命題 / 定理 | 対象領域 | 数理的帰結の要約 |
| :---: | :--- | :--- |
| **命題 1** | 事後標準化の無力性 | 乗算後に `StandardScaler` を適用しても相関行列と $\mathrm{VIF} = 1 + \tau_{M,1}^2 + \tau_{M,2}^2$ は一切変化しない。 |
| **命題 2** | 倍精度浮動小数点崩壊 | 正規方程式の条件数 $\mathrm{cond}(\bm{X}^\top\bm{X}) = \mathcal{O}(\tau_S^2 \tau_M^4)$ が $\varepsilon_{\mathrm{mach}}^{-1} \approx 4.50 \times 10^{15}$ を突破し、Python `pinv` のサイレント特異値切断（Case 16）を引き起こす（有理数体 $\mathbb{Q}$ 引き戻しは厳密不変）。 |
| **命題 3** | 歪度・季節欠測リーク | 3次交差積率（$\mu_{2,1}, \mu_{1,2} \neq 0$）や季節欠測下では、単純中心化積 $(X_1-\mu_1)(X_2-\mu_2)$ でも主効果と直交しない（$\mathrm{VIF} > 1$）。 |
| **命題 4** | 符号反転と Wald 検定崩壊 | 非中心化主効果の $t$ 統計量が $t(\tilde{\beta}_1) \to -\mathrm{sign}(M_2) t(\hat{\beta}_3)$ へ漸近し、Type S 符号反転または Type II 検出力喪失を起こす。 |
| **命題 5** | ハウスホルダーQR直交回復 | 事前標準化＋ハウスホルダーQRグラム・シュミット残差直交化 $\bm{i}_{\mathrm{orth}}$ と対角ヤコビアン引き戻し $\bm{J}_{\mathrm{clin}}$ により $\mathrm{VIF} \equiv 1.0000$ と臨床単位不変性を同時達成。 |
| **命題 6** | GLM IRLS 分散膨張 | ロジスティック・ポアソン IRLS において、非中心化主効果 $\tilde{\beta}_1$ の漸近分散が $M_2^2 \mathrm{Var}(\hat{\beta}_3)$ だけ膨張する。 |
| **命題 7** | IRLS ヘッセ行列特異化 | 極端な線形予測子により IRLS 重み $w_i^{(k)} \to 0$ となり、重み付き情報行列 $\bm{X}^\top\bm{W}^{(k)}\bm{X}$ の条件数が $10^{16}$ を超えて発散する。 |
| **命題 8** | `scikit-learn` $L_2$ オッズ比歪み | 既定の $L_2$ 正則化（$C=1.0$）が $\tilde{\beta}_1 \approx -M_2\tilde{\beta}_3$ を強制し、真のオッズ比を帰無値へ縮小させる（Case 4 で $\mathrm{OR}_1 = 2.020 \to 1.012$）。 |
| **命題 9** | Cox 部分尤度の情報特異化 | Cox 比例ハザードモデルのリスク集合重み $w_i(t)$ 上で $\mathcal{O}(\tau_{M,2}^2)$ の情報行列退化とハザード比逆転が発生する（**Domain IV の 8/9 試験で逆転**）。 |
| **定理 10** | **LASSO $\sqrt{2}$ ノルム近道** | 事後標準化 LASSO において $\tau_M > 1/\sqrt{2} \approx 0.7071$ のとき、交互作用列のエントリー相関が主効果より最大 $\sqrt{2}$ 倍大きくなり、真の主効果が脱落する（**10/22コホート**）。 |
| **命題 11** | Elastic Net グループ効果歪み | $L_1 + L_2$ ペナルティ下で $\tilde{\beta}_1$ と $\tilde{\beta}_3$ が分散を寄生的に共有し、強階層原則（Strong Heredity）を破壊する。 |
| **命題 12** | CART 分割ゲイン乗っ取り | 軸平行な決定木は斜め方向の加法信号 $Y = \beta_1 X_1 + \beta_2 X_2$ を直接分割できず、積 $X_1 X_2$ の双曲線等高線がルート分割ゲインの最大 **95.3%** を奪う。 |
| **命題 13** | TreeSHAP 乗っ取りの $\mathbb{Q}$ 証明 | 積 $P = X_1 X_2$ で分割された深さ1決定木において、Interventional TreeSHAP は主効果に $\phi_1=\phi_2=0$、積項に $\phi_P=100\%$（**既約分数 `1/1`**）を割り当てる。 |
| **命題 14** | 直交化による TreeSHAP 回復 | 直交化残差 $\bm{i}_{\mathrm{orth}}$ の投入により双曲線等高線の傾きが消去され、22コホートの平均交互作用 SHAP シェアが **69.5% から 2.5% へ回復**する。 |
| **命題 15** | LMM コレスキー境界崩壊 | $x_{ij1} = c_1(z_{ij1} + M_1)$ の下でランダム効果相関が $\tilde{\rho}_{01} \to -1$ へ退化し、`lme4` の相対コレスキー比が $\theta_3/\theta_1 = \mathcal{O}((c_1 M_1^2)^{-1}) \to 0$ へ崩壊する。 |
| **命題 16** | LMM の $\mathbb{Q}$ 引き戻しとランク崩壊 | 有理数体 $\mathbb{Q}$ 上では周辺共分散 $\bm{V}_i$ は不変だが、REML の境界クランプ（$\tilde{\theta}_3 = 0$）が起きると $\mathrm{rank}(\widetilde{\bm{G}}) = 1 < 2$ となり事後引き戻し不能となる。 |
| **命題 17** | パネル生態学的リークと2段階QR | 個体内デミーニング積 $P_{ij} - \bar{P}_{i\cdot}$ にはクロスレベル項 $b_{i2}^B w_{ij1}^W$ が残存する；2段階階層グラム・シュミット直交化によりリークを完全消去（$\hat{\beta}_{W,3} = 0.0000$）。 |
| **命題 18.1**| ベイズ異方性事前分布バイアス | 非中心化空間の等方性事前分布 $\mathcal{N}(\bm{0}, \tau_0^2\bm{I})$ は基準空間で異方性精度 $\lambda_0(\bm{T}\bm{T}^\top)^{-1}$ となり、効果を **98.1%〜99.97% 縮小**させ $\mu_1^{\mathrm{pull}} \approx M_2\mu_3^{\mathrm{pull}}$ にロックする。 |
| **命題 18.2**| HMC 対角質量適応後のステップ崩壊 | `adapt_diag` 適応後も $\mathrm{cond}(\widetilde{\bm{H}}) = 4\tau_M^2 + 2$ が残存し、リープフロッグ最大ステップ幅が $\epsilon_{\mathrm{leap}} = \mathcal{O}(\tau_M^{-1})$ で崩壊して木深さ上限（$1,024$）に飽和する。 |
| **命題 19** | 交互作用増幅 Neal's Funnel | 階層モデルの漏斗頸部で曲率が $\lambda_{\max} = \mathcal{O}(e^{-2\omega}(1+M^2))$、ハミルトニアン誤差が $\Delta H = \mathcal{O}(\epsilon_0^4 e^{-6\omega} M^6)$ で爆発し、**発散遷移 100%** に陥る。 |
| **命題 20** | ベイズ QR-NCP による多様体等方化 | `transformed data` でのハウスホルダーQR直交化＋$\bm{u}_1$ 上のコレスキーNCP＋`generated quantities` 引き戻しにより、$\mathrm{cond}(\widetilde{\bm{H}}) \approx 1.02$・**発散 0.0%**・$\mathrm{E\text{-}BFMI} = 1.05$ を達成。 |

---

## 4. 全28医療・疫学コホート（Cases 1–28）言語間完全突合台帳

| Case | 医療・疫学コホート名 | 解析パラダイム / ドメイン | $N$ | 有理数 $\mathbb{Q}$ 既約分数 | Python 提案 $\hat{\beta}_1$ | POSIX `awk` 提案 $\hat{\beta}_1$ | $\|\Delta_{\mathrm{Py-AWK}}\|$ | 非中心化 Raw $\tilde{\beta}_1$ | 非中心化・事後標準化ワークフローで発生する臨床的・AI的破綻 |
| ---: | :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Diabetes 糖尿病進行 | Domain I (OLS) | 442 | `28/25` | `+1.1200` | `+1.1200` | `0.00e+00` | `-0.520` | **符号反転**、LASSO脱落、TreeSHAP 58.4%乗っ取り |
| **2** | UK Lung 英国肺疾患死亡 | Domain III (OLS) | 72 | `4026/5` | `+805.2000` | `+805.2000` | `0.00e+00` | `+57148.500` | **SE 1,439倍膨張**、R `dqrdc2` 列落ち境界突破 |
| **3** | Mayo Clinic PBC 肝硬変 | Domain I (OLS) | 312 | `-141/1000` | `-0.1410` | `-0.1410` | `0.00e+00` | `+0.110` | **符号反転**、LASSO脱落、TreeSHAP 63.8%乗っ取り |
| **4** | Low Birth Weight 低出生体重 | Domain I (OLS/Logit) | 189 | `-1419/5` | `-283.8000` | `-283.8000` | `0.00e+00` | `-305.900` | **偽陰性 ($p=0.56$)**、`sklearn` 喫煙OR $\to 1.01$ 消失 |
| **5** | Heart Failure 心不全死亡 | Domain II (Logistic) | 299 | `343/500` | `+0.6860` | `+0.6860` | `0.00e+00` | `-4.820` | **符号反転 ($\mathrm{OR}: 1.99 \to 0.008$)**、LASSO脱落 |
| **6** | Framingham 冠動脈疾患 | Domain II (Logistic) | 3,658 | `13/200` | `+0.0650` | `+0.0650` | `0.00e+00` | `+0.200` | **有意性消失 ($p=0.08$)**、LASSO主効果脱落 |
| **7** | Breast Cancer WDBC 乳癌 | Domain II (Logistic) | 569 | `114/125` | `+0.9120` | `+0.9120` | `0.00e+00` | `+0.340` | **偽陰性 ($p=0.48$)**、LASSO脱落、SHAP 61.5% |
| **8** | Chicago NMMAPS 大気汚染 | Domain III (OLS) | 5,114 | `41/50` | `+0.8200` | `+0.8200` | `0.00e+00` | `-13.020` | **符号反転 ($0\text{ K}$ 絶対零度外挿)**、SHAP 76.4% |
| **9** | Veteran Lung 退役軍人肺癌 | Domain IV (Cox PH) | 137 | `-33/1000` | `-0.0330` | `-0.0330` | `0.00e+00` | `-0.030` | **偽陰性 ($p=0.29$)**、TreeSHAP 62.0%乗っ取り |
| **10** | Pima Diabetes 糖尿病 | Domain II (Logistic) | 532 | `19/500` | `+0.0380` | `+0.0380` | `0.00e+00` | `+0.010` | **73.7%減衰 ($p=0.34$)**、TreeSHAP 64.5%乗っ取り |
| **11** | Stanford Heart 心臓移植 | Domain IV (Cox PH) | 103 | `-1317/1000` | `-1.3170` | `-1.3170` | `0.00e+00` | `+98.420` | **致死的符号反転 ($\mathrm{VIF}=2.5\times 10^6$)**、LASSO脱落 |
| **12** | German Breast タモキシフェン | Domain IV (Cox PH) | 686 | `-91/250` | `-0.3640` | `-0.3640` | `0.00e+00` | `+0.390` | **治療効果逆転 ($\mathrm{HR}: 0.70 \to 1.48$ 有害化)** |
| **13** | NCCTG Colon 大腸癌 5-FU | Domain IV (Cox PH) | 619 | `-64/125` | `-0.5120` | `-0.5120` | `0.00e+00` | `+0.180` | **治療効果逆転 ($\mathrm{HR}: 0.60 \to 1.20$ 有害化)** |
| **14** | Danish Melanoma 悪性黒色腫 | Domain II (Logistic) | 205 | `171/500` | `+0.3420` | `+0.3420` | `0.00e+00` | `+42.700` | **12,385%歪み ($\mathrm{VIF}=5.8\times 10^5$)**、SHAP 84.3% |
| **15** | Siri Body Fat 体脂肪率 | Domain I (OLS) | 252 | `-248/125` | `-1.9840` | `-1.9840` | `0.00e+00` | `+2.320` | **符号反転**、LASSO脱落、TreeSHAP 72.1%乗っ取り |
| **16** | NCCTG Lung 肺癌熱量 ($\text{J}$) | Domain III (OLS) | 167 | `921/500` | `+1.8420` | `+1.8420` | `0.00e+00` | `+0.660` | **Python SVD `pinv` サイレント切断 (SE比 $0.181$倍)** |
| **17** | WCGS 冠動脈疾患・行動型 | Domain II (Logistic) | 3,154 | `359/500` | `+0.7180` | `+0.7180` | `0.00e+00` | `-0.820` | **符号反転 ($\mathrm{OR}: 2.05 \to 0.44$)**、LASSO脱落 |
| **18** | ACTG 175 HIV 多剤併用療法 | Domain IV (Cox PH) | 2,139 | `-137/200` | `-0.6850` | `-0.6850` | `0.00e+00` | `+0.820` | **HIV多剤併用療法が逆転 ($\mathrm{HR}: 0.50 \to 2.27$)** |
| **19** | Mayo Clinic MGUS2 骨髄腫前駆 | Domain IV (Cox PH) | 1,384 | `123/500` | `+0.2460` | `+0.2460` | `0.00e+00` | `-0.230` | **符号反転 ($\mathrm{HR}: 1.28 \to 0.80$)**、SHAP 66.1% |
| **20** | Mayo FLC 血清遊離軽鎖 | Domain IV (Cox PH) | 7,874 | `69/500` | `+0.1380` | `+0.1380` | `0.00e+00` | `-16.230` | **符号反転 ($\tau_{M,2}=1296.4$)**、**SHAP 89.6%乗っ取り** |
| **21** | Rotterdam 乳癌術後化学療法 | Domain IV (Cox PH) | 2,982 | `-159/500` | `-0.3180` | `-0.3180` | `0.00e+00` | `+36.440` | **化学療法が逆転 ($\tilde{\beta}_1=+36.44$)**、LASSO脱落 |
| **22** | Retinopathy 網膜レーザー手術 | Domain IV (Cox PH) | 394 | `-98/125` | `-0.7840` | `-0.7840` | `0.00e+00` | `+0.600` | **失明予防レーザーが逆転 ($\mathrm{HR}: 0.46 \to 1.82$)** |
| **23** | Sleep Study 断眠反応遅延 | 経時混合 LMM (REML) | 180 | `1047/100` | `+10.4700` | `+10.4700` | `0.00e+00` | `-4.182` | **進行スロープ逆転**、$\tilde{\rho}_{01}=-0.99995$、$\theta_3/\theta_1=3.2\text{e-}7$ |
| **24** | Mayo PBCSeq 縦断肝硬変 | 経時混合 LMM (REML) | 1,945 | `79/500` | `+0.1580` | `+0.1580` | `0.00e+00` | `-0.112` | **肝不全進行が逆転**、$\theta_3/\theta_1=9.3\text{e-}7$ (`Singular`) |
| **25** | Orthodont 小児頭蓋骨格成長 | 経時混合 LMM (REML) | 108 | `33/50` | `+0.6600` | `+0.6600` | `0.00e+00` | `-0.572` | **成長スロープ逆転 ($-0.572\text{ mm/yr}$ 頭蓋縮小誤診)** |
| **26** | Pima Diabetes ベイズ診断 | ベイズ HMC-NUTS | 532 | `38189/100000` | `+0.3815` | `+0.3815` | `0.00e+00` | `+0.094` | **事前分布縮小で 75.3%減衰**、95% CrI が $0$ を跨ぐ |
| **27** | Mayo PBCSeq ベイズ縦断階層 | ベイズ HMC-NUTS | 1,945 | `15799/100000` | `+0.1578` | `+0.1578` | `0.00e+00` | `-0.111` | **進行スロープ逆転**、**発散 100%**、$\mathrm{E\text{-}BFMI}=0.11$ |
| **28** | NCCTG Lung 18施設階層肺癌 | ベイズ HMC-NUTS | 167 | `184073/100000`| `+1.8295` | `+1.8295` | `0.00e+00` | `+1.253` | **ファネル崩壊 ($\mathrm{E\text{-}BFMI}=0.11$)**、CrI が $0$ を跨ぐ |

---

## 5. 再現検証パイプラインの実行方法とリポジトリ構成

本リポジトリの全4本の POSIX/GNU `awk` 有理数エンジン、全5本の Python（`SymPy.Rational` / `SciPy`）監査スクリプト、Stage 3 臨床 CSV ブリッジ、および Stage 4 言語間完全突合スクリプトは、Linux・macOS・Google Colab 上で単一コマンドにより自動実行可能です（Windows CRLF 自動修復および Python 3.13 `PYTHON_BASIC_REPL=1` 対策済み）：

```bash
git clone [https://github.com/RUDATAScience/Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models.git](https://github.com/RUDATAScience/Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models.git)
cd Silent-Numerical-Collapse-and-Algorithmic-Hijacking-in-Medical-Interaction-and-Seasonal-Models
bash run_pipeline.sh
```

---

## 6. 倫理的承認（UMIN連絡先）・利益相反・謝辞

* **倫理的承認および研究連絡先（Ethics Statement and UMIN Affiliation）：**  
  本研究で解析・監査対象とした全28件の医療・疫学コホート（Cases 1〜28、総症例数 $N_{\mathrm{total}} = 36,160$）は、すべて既発表の査読付き学術論文および公的統計パッケージにおいて広く公開されている完全匿名化・非識別化済みの二次利用ベンチマークデータセットです。ヘルシンキ宣言および「人を対象とする生命科学・医学系研究に関する倫理指針」に基づき、新たな倫理審査委員会（IRB）の承認および個別のインフォームド・コンセント取得は不要です。著者は大学病院医療情報ネットワーク（**UMIN**）の研究員であり、医療データ監査プロトコルおよび研究倫理に関する照会は公式 UMIN アドレス（[`kawahata-hc@umin.ac.jp`](mailto:kawahata-hc@umin.ac.jp)）にて受け付けます。
* **利益相反（Conflict of Interest）：**  
  国際医学雑誌編集者委員会（ICMJE）の規定に準拠し、開示すべき財務的または非財務的な利益相反は一切存在しません。
* **謝辞（Acknowledgments）：**  
  本研究の遂行にあたり、社会データサイエンス（SDS）概論の参加者の皆様との活発な議論と鋭い問いかけから多大なる示唆をいただきました。また、全20命題の理論的構造化と28医療コホートの横断監査、ならびにLaTeX原稿の精緻化において粘り強い共同検証パートナーとして伴走してくれた対話型AI **Gemini**（Google）に心より謝意を表します。そして何より、BLAS/LAPACK や浮動小数点ソルバーに一切依存せず、ユークリッド互除法（`gcd`）による有理数体 $\mathbb{Q}$ 上の厳密ガウス・ジョルダン消去の手計算実装を可能にしてくれた **POSIX/GNU `awk`**（および創始者である Alfred V. Aho, Peter J. Weinberger, Brian W. Kernighan の三氏）に最大の敬意と感謝を捧げます。
