#!/usr/bin/env python
"""Quantify svd.c (textbook Golub-Reinsch) stability against LAPACK dgelsd.

Compiles svd.c standalone (via cpu_compat.h) and compares, over synthetic
matrices with controlled condition numbers and a duplicate-row case modeled on
the solver's degenerate-compset KKT systems:
  - relative error of each singular value vs LAPACK
  - min-norm lstsq solution difference vs numpy.linalg.lstsq (dgelsd)
  - loss of orthogonality in U/V
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU = os.path.join(ROOT, 'pycalphad', 'gpu')

HARNESS = r"""
#include "cpu_compat.h"
#include "svd.c"

extern "C" int svd_test(const double* A_in, int m, int n,
                        double* U, double* V, double* sv, double* x,
                        const double* b_in, double rcond, int use_jacobi) {
    static thread_local double Abuf[128*128], bbuf[128], superdiag[128];
    for (int i = 0; i < m*n; ++i) Abuf[i] = A_in[i];
    for (int i = 0; i < m; ++i) bbuf[i] = b_in[i];
    int rc = use_jacobi ? Jacobi_SVD(Abuf, m, n, U, sv, V)
                        : Singular_Value_Decomposition(Abuf, m, n, U, sv, V, superdiag);
    if (rc != 0) return rc;
    Singular_Value_Decomposition_Solve(U, sv, V, rcond, m, n, bbuf, x);
    return 0;
}
"""

src = os.path.join('/tmp', 'svd_harness.cpp')
lib_path = os.path.join('/tmp', 'svd_harness.so')
with open(src, 'w') as f:
    f.write(HARNESS)
r = subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-shared', '-fPIC',
                    f'-I{GPU}', '-o', lib_path, src], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-3000:])
    sys.exit(1)

lib = ctypes.CDLL(lib_path)
lib.svd_test.restype = ctypes.c_int
lib.svd_test.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                         ctypes.c_int]

USE_JACOBI = int(os.environ.get('SVD_JACOBI', '0'))


def ours(A, b, rcond=1e-16):
    m, n = A.shape
    U = np.zeros((m, n)); V = np.zeros((n, n)); sv = np.zeros(n); x = np.zeros(n)
    A_c = np.ascontiguousarray(A); b_c = np.ascontiguousarray(b)
    rc = lib.svd_test(A_c.ctypes.data, m, n, U.ctypes.data, V.ctypes.data,
                      sv.ctypes.data, x.ctypes.data, b_c.ctypes.data, rcond,
                      USE_JACOBI)
    return rc, np.sort(sv)[::-1], x, U, V


rng = np.random.default_rng(42)

print(f"{'case':<26} {'cond':>8} | {'ours vs TRUE':>13} {'LAPACK vs TRUE':>15} | "
      f"{'resid ours':>12} {'resid dgelsd':>13}")
for n in (6, 12, 24):
    for logk in (2, 6, 10, 13):
        Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
        Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s_true = np.logspace(0, -logk, n)
        A = (Q1 * s_true) @ Q2.T
        b = rng.standard_normal(n)
        rc, sv, x, U, V = ours(A, b)
        if rc != 0:
            print(f"n={n} 1e-{logk}: OUR SVD FAILED rc={rc}")
            continue
        sv_lapack = np.linalg.svd(A, compute_uv=False)
        # smallest singular value, relative error vs constructed ground truth
        ours_vs_true = abs(sv[-1] - s_true[-1]) / s_true[-1]
        lapack_vs_true = abs(sv_lapack[-1] - s_true[-1]) / s_true[-1]
        x_ref = np.linalg.lstsq(A, b, rcond=1e-16)[0]
        res_ours = np.linalg.norm(A @ x - b)
        res_ref = np.linalg.norm(A @ x_ref - b)
        print(f"n={n:<2} sigma 1..1e-{logk:<10} {10.0**logk:>8.0e} | {ours_vs_true:>13.3e} "
              f"{lapack_vs_true:>15.3e} | {res_ours:>12.3e} {res_ref:>13.3e}")

# Degenerate-compset pattern: two nearly identical rows (like duplicate BCC_B2
# compsets in the equilibrium KKT matrix), perturbation at 1e-6..1e-12.
print("\nDuplicate-row KKT-like case (row2 = row1 + eps*noise):")
for n in (6, 12):
    for eps in (1e-6, 1e-9, 1e-12):
        A = rng.standard_normal((n, n))
        A[1] = A[0] + eps * rng.standard_normal(n)
        b = rng.standard_normal(n)
        rc, sv, x, U, V = ours(A, b)
        sv_l = np.linalg.svd(A, compute_uv=False)
        x_ref = np.linalg.lstsq(A, b, rcond=1e-16)[0]
        dx = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        rel_small = abs(sv[-1] - sv_l[-1]) / sv_l[-1]
        print(f"  n={n:<2} eps={eps:.0e}: small-sv rel err {rel_small:.3e}  |dx|/|x| {dx:.3e}")
