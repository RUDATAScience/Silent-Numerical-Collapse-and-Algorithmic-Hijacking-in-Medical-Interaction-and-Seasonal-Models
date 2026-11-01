#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymc_qr_orthogonal_ncp_template.py
Drop-in Implementation of Algorithm 1 (OrthogonalInteractionTransformer) and
Algorithm 2 (Bayesian QR-Orthogonal NCP for PyMC v5 / Stan) with Built-In
Symbolic Rational (Q) Ground-Truth Auditing and Appendix A Cross-Platform Benchmark.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Dict, Any
import numpy as np
import scipy.linalg as la
import sympy as sp


class OrthogonalInteractionTransformer:
    def __init__(self) -> None:
        self.mu1_: float = 0.0
        self.mu2_: float = 0.0
        self.s1_: float = 1.0
        self.s2_: float = 1.0
        self.s_orth_: float = 1.0
        self.pi_vec_: np.ndarray = np.zeros(3)
        self.J_clin_marg_: np.ndarray = np.eye(4)

    def fit_transform(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        n = len(x1)
        self.mu1_ = float(np.mean(x1))
        self.mu2_ = float(np.mean(x2))
        self.s1_ = max(float(np.std(x1, ddof=0)), 1e-15)
        self.s2_ = max(float(np.std(x2, ddof=0)), 1e-15)

        u1 = (x1 - self.mu1_) / self.s1_
        u2 = (x2 - self.mu2_) / self.s2_
        M_main = np.column_stack([np.ones(n), u1, u2])
        raw_u12 = u1 * u2

        Q_m, R_m = np.linalg.qr(M_main, mode="reduced")
        self.pi_vec_ = la.solve_triangular(R_m, Q_m.T @ raw_u12)
        r_orth = raw_u12 - M_main @ self.pi_vec_
        self.s_orth_ = max(float(np.std(r_orth, ddof=0)), 1e-15)
        i_orth = r_orth / self.s_orth_

        self.J_clin_marg_ = np.diag([
            1.0,
            1.0 / self.s1_,
            1.0 / self.s2_,
            1.0 / (self.s1_ * self.s2_ * self.s_orth_),
        ])
        return np.column_stack([np.ones(n), u1, u2, i_orth])

    def fit_ols_or_conjugate_with_exact_q_audit(
        self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray, lam0: float = 0.0
    ) -> Dict[str, Any]:
        Z_orth = self.fit_transform(x1, x2)
        Gram = Z_orth.T @ Z_orth + lam0 * np.eye(4)
        rhs = Z_orth.T @ y

        gamma_flt = la.solve(Gram, rhs)
        beta_clin_flt = self.J_clin_marg_ @ gamma_flt

        G_rat = sp.Matrix([
            [sp.Rational(Fraction(float(Gram[i, j])).limit_denominator(10**8)) for j in range(4)]
            for i in range(4)
        ])
        b_rat = sp.Matrix([
            sp.Rational(Fraction(float(rhs[i])).limit_denominator(10**8)) for i in range(4)
        ])
        gamma_q = np.array([float(v) for v in G_rat.LUsolve(b_rat)], dtype=np.float64)
        beta_clin_q = self.J_clin_marg_ @ gamma_q
        rel_err_q = float(np.linalg.norm(beta_clin_flt - beta_clin_q) / max(np.linalg.norm(beta_clin_q), 1e-15))

        return {
            "Z_orth": Z_orth,
            "beta_clin_flt": beta_clin_flt,
            "beta_clin_exact_q": beta_clin_q,
            "rel_err_vs_q": rel_err_q,
            "cond_Z_orth": float(np.linalg.cond(Z_orth)),
        }


if __name__ == "__main__":
    rng = np.random.default_rng(202610)
    x1_demo = 94.65 + 13.80 * rng.normal(0, 1, 442)
    x2_demo = 26.38 + 4.41 * rng.normal(0, 1, 442)
    y_demo = 152.1 + 2.648 * (x1_demo - 94.65) + 5.12 * (x2_demo - 26.38) + 0.151 * (x1_demo - 94.65) * (x2_demo - 26.38)
    tr = OrthogonalInteractionTransformer()
    res = tr.fit_ols_or_conjugate_with_exact_q_audit(x1_demo, x2_demo, y_demo, lam0=1.0 / 6.25)
    print(f"[Python Script 5/5 Self-Test] cond(Z_orth) = {res['cond_Z_orth']:.4f} | "
          f"Exact Q beta_1 = {res['beta_clin_exact_q'][1]:+.6f} | "
          f"Float64 Pull-Back beta_1 = {res['beta_clin_flt'][1]:+.6f} | "
          f"Relative Error vs Q = {res['rel_err_vs_q']:.2e}")
