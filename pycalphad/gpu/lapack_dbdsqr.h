/* lapack_dbdsqr.h
 *
 * Faithful C99 transliteration of LAPACK DBDSQR (dbdsqr.f, reference
 * LAPACK).  Numerical operation order matches the Fortran source
 * statement-for-statement; goal is bitwise-identical results given
 * bitwise-identical dependency routines.
 *
 * IMPORTANT: this header must be #include'd AFTER the sibling header that
 * provides the following `static` device functions (they are static there,
 * so no extern declarations are made here; we simply call them):
 *
 *   pyclap_lsame, pyclap_dlamch, pyclap_dlartg, pyclap_dlas2,
 *   pyclap_dlasv2, pyclap_dlasr, pyclap_drot, pyclap_dswap, pyclap_dscal
 *
 * Deviations from dbdsqr.f (and ONLY these):
 *   - Argument-validation boilerplate (XERBLA / INFO < 0 checks,
 *     Fortran lines 302-326) is dropped.
 *   - The DLASQ1 path (no singular vectors wanted, Fortran lines 336-346)
 *     is NOT transliterated: this port is always called with NCVT>0 and
 *     NRU>0, so that branch returns info = -999 to make misuse loud.
 *
 * GOTO label mapping (Fortran label -> C construct):
 *   10  -> for loop  (lower-to-upper bidiagonal rotation)
 *   20  -> for loop  (smax over d)
 *   30  -> for loop  (smax over e)
 *   40  -> for loop  (sminoa recurrence), GO TO 50 -> C goto L50
 *   50  -> C label L50
 *   60  -> C label L60 (top of main iteration loop)
 *   70  -> for loop  (find split), GO TO 80 -> C goto L80
 *   80  -> C label L80
 *   90  -> C label L90
 *   100 -> for loop  (forward convergence test), GO TO 60 -> C goto L60
 *   110 -> for loop  (backward convergence test), GO TO 60 -> C goto L60
 *   120 -> for loop  (zero-shift QR, top to bottom)
 *   130 -> for loop  (zero-shift QR, bottom to top)
 *   140 -> for loop  (shifted QR, top to bottom)
 *   150 -> for loop  (shifted QR, bottom to top)
 *   160 -> C label L160 (make singular values positive)
 *   170 -> for loop  (sign fix)
 *   180 -> for loop  (scan for smallest d)
 *   190 -> for loop  (insertion-sort outer)
 *   200 -> C label L200 (nonconvergence: count nonzero e)
 *   210 -> for loop  (count nonzero e)
 *   220 -> C label L220 (return)
 */

#ifndef PYCLAP_LAPACK_DBDSQR_H
#define PYCLAP_LAPACK_DBDSQR_H

#if !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(__device__)
#define __device__
#endif

#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__HIPCC__)
#if !defined(__CUDACC_RTC__) && !defined(__HIPCC_RTC__)
#include <math.h> /* fabs, sqrt, pow, copysign */
#endif
#endif

/* Tiny helpers mirroring Fortran intrinsic MAX/MIN/SIGN argument order.
 * Fortran MAX(a,b): larger of the two (first argument on ties).
 * Fortran SIGN(a,b): |a| with the sign of b; implemented via copysign to
 * match what gfortran emits for reference LAPACK (transfers -0.0 sign). */
__device__ static double pyclap_bdsqr_dmax(double a, double b)
{
    return (a >= b) ? a : b;
}

__device__ static double pyclap_bdsqr_dmin(double a, double b)
{
    return (a <= b) ? a : b;
}

__device__ static double pyclap_bdsqr_dsign(double a, double b)
{
    return copysign(a, b);
}

/* Column-major convention throughout: Fortran X(i,j) -> x[(i-1)+(j-1)*ldx].
 * WORK must have length >= 4*(n-1) (vectors are always requested here). */
__device__ static int pyclap_dbdsqr(char uplo, int n, int ncvt, int nru,
                                    int ncc, double* d, double* e,
                                    double* vt, int ldvt, double* u, int ldu,
                                    double* c, int ldc, double* work)
{
    /* .. Parameters .. */
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double NEGONE = -1.0;
    const double HNDRTH = 0.01;
    const double TEN = 10.0;
    const double HNDRD = 100.0;
    const double MEIGTH = -0.125;
    const int MAXITR = 6;

    /* .. Local Scalars .. */
    int lower, rotate;
    int i, idir, isub, iter, iterdivn, j, ll, lll, m;
    int maxitdivn, nm1, nm12, nm13, oldll, oldm;
    double abse, abss, cosl, cosr, cs, eps, f, g, h, mu;
    double oldcs, oldsn, r, shift, sigmn, sigmx, sinl;
    double sinr, sll, smax, smin, sminoa;
    double sn, thresh, tol, tolmul, unfl;
    int info;

    /* Silence maybe-uninitialized warnings; these are never read before
     * being assigned on any path the Fortran takes either. */
    oldsn = ZERO;
    ll = 0;
    sminoa = ZERO;

    info = 0;
    lower = pyclap_lsame(uplo, 'L');
    /* Argument-validation boilerplate (XERBLA / INFO<0, Fortran lines
     * 304-326) intentionally dropped per port conventions. */

    if (n == 0)
        return info;
    if (n == 1)
        goto L160; /* Fortran: IF( N.EQ.1 ) GO TO 160 */

    /* ROTATE is true if any singular vectors desired, false otherwise */
    rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);

    if (!rotate) {
        /* dlasq1 path: unreachable, vectors always requested in this port */
        return -999;
    }

    nm1 = n - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    idir = 0;

    /* Get machine constants */
    eps = pyclap_dlamch('E');  /* DLAMCH( 'Epsilon' ) */
    unfl = pyclap_dlamch('S'); /* DLAMCH( 'Safe minimum' ) */

    /* If matrix lower bidiagonal, rotate to be upper bidiagonal
     * by applying Givens rotations on the left */
    if (lower) {
        for (i = 1; i <= n - 1; i++) { /* DO 10 */
            pyclap_dlartg(d[i - 1], e[i - 1], &cs, &sn, &r);
            d[i - 1] = r;
            e[i - 1] = sn * d[i];   /* E( I ) = SN*D( I+1 ) */
            d[i] = cs * d[i];       /* D( I+1 ) = CS*D( I+1 ) */
            work[i - 1] = cs;       /* WORK( I ) */
            work[nm1 + i - 1] = sn; /* WORK( NM1+I ) */
        }

        /* Update singular vectors if desired */
        if (nru > 0)
            pyclap_dlasr('R', 'V', 'F', nru, n, &work[0], &work[n - 1], u,
                         ldu);
        if (ncc > 0)
            pyclap_dlasr('L', 'V', 'F', n, ncc, &work[0], &work[n - 1], c,
                         ldc);
    }

    /* Compute singular values to relative accuracy TOL
     * (By setting TOL to be negative, algorithm will compute
     * singular values to absolute accuracy ABS(TOL)*norm(input matrix)) */
    tolmul = pyclap_bdsqr_dmax(TEN, pyclap_bdsqr_dmin(HNDRD, pow(eps, MEIGTH)));
    tol = tolmul * eps;

    /* Compute approximate maximum, minimum singular values */
    smax = ZERO;
    for (i = 1; i <= n; i++) { /* DO 20 */
        smax = pyclap_bdsqr_dmax(smax, fabs(d[i - 1]));
    }
    for (i = 1; i <= n - 1; i++) { /* DO 30 */
        smax = pyclap_bdsqr_dmax(smax, fabs(e[i - 1]));
    }
    smin = ZERO;
    if (tol >= ZERO) {
        /* Relative accuracy desired */
        sminoa = fabs(d[0]);
        if (sminoa == ZERO)
            goto L50; /* GO TO 50 */
        mu = sminoa;
        for (i = 2; i <= n; i++) { /* DO 40 */
            mu = fabs(d[i - 1]) * (mu / (mu + fabs(e[i - 2])));
            sminoa = pyclap_bdsqr_dmin(sminoa, mu);
            if (sminoa == ZERO)
                goto L50; /* GO TO 50 */
        }
    L50: /* Fortran label 50 */
        sminoa = sminoa / sqrt((double)n);
        thresh = pyclap_bdsqr_dmax(tol * sminoa,
                                   MAXITR * (n * (n * unfl)));
    } else {
        /* Absolute accuracy desired */
        thresh = pyclap_bdsqr_dmax(fabs(tol) * smax,
                                   MAXITR * (n * (n * unfl)));
    }

    /* Prepare for main iteration loop for the singular values
     * (MAXIT is the maximum number of passes through the inner
     * loop permitted before nonconvergence signalled.) */
    maxitdivn = MAXITR * n;
    iterdivn = 0;
    iter = -1;
    oldll = -1;
    oldm = -1;

    /* M points to last element of unconverged part of matrix */
    m = n;

/* Begin main iteration loop */
L60: /* Fortran label 60 */

    /* Check for convergence or exceeding iteration count */
    if (m <= 1)
        goto L160; /* GO TO 160 */

    if (iter >= n) {
        iter = iter - n;
        iterdivn = iterdivn + 1;
        if (iterdivn >= maxitdivn)
            goto L200; /* GO TO 200 */
    }

    /* Find diagonal block of matrix to work on */
    if (tol < ZERO && fabs(d[m - 1]) <= thresh)
        d[m - 1] = ZERO;
    smax = fabs(d[m - 1]);
    for (lll = 1; lll <= m - 1; lll++) { /* DO 70 */
        ll = m - lll;
        abss = fabs(d[ll - 1]);
        abse = fabs(e[ll - 1]);
        if (tol < ZERO && abss <= thresh)
            d[ll - 1] = ZERO;
        if (abse <= thresh)
            goto L80; /* GO TO 80 */
        smax = pyclap_bdsqr_dmax(pyclap_bdsqr_dmax(smax, abss), abse);
    }
    ll = 0;
    goto L90; /* GO TO 90 */
L80: /* Fortran label 80 */
    e[ll - 1] = ZERO;

    /* Matrix splits since E(LL) = 0 */
    if (ll == m - 1) {
        /* Convergence of bottom singular value, return to top of loop */
        m = m - 1;
        goto L60; /* GO TO 60 */
    }
L90: /* Fortran label 90 */
    ll = ll + 1;

    /* E(LL) through E(M-1) are nonzero, E(LL-1) is zero */
    if (ll == m - 1) {
        /* 2 by 2 block, handle separately */
        pyclap_dlasv2(d[m - 2], e[m - 2], d[m - 1], &sigmn, &sigmx, &sinr,
                      &cosr, &sinl, &cosl);
        d[m - 2] = sigmx;
        e[m - 2] = ZERO;
        d[m - 1] = sigmn;

        /* Compute singular vectors, if desired */
        if (ncvt > 0)
            pyclap_drot(ncvt, &vt[m - 2], ldvt, &vt[m - 1], ldvt, cosr,
                        sinr); /* VT( M-1, 1 ), VT( M, 1 ) */
        if (nru > 0)
            pyclap_drot(nru, &u[(m - 2) * ldu], 1, &u[(m - 1) * ldu], 1,
                        cosl, sinl); /* U( 1, M-1 ), U( 1, M ) */
        if (ncc > 0)
            pyclap_drot(ncc, &c[m - 2], ldc, &c[m - 1], ldc, cosl,
                        sinl); /* C( M-1, 1 ), C( M, 1 ) */
        m = m - 2;
        goto L60; /* GO TO 60 */
    }

    /* If working on new submatrix, choose shift direction
     * (from larger end diagonal element towards smaller) */
    if (ll > oldm || m < oldll) {
        if (fabs(d[ll - 1]) >= fabs(d[m - 1])) {
            /* Chase bulge from top (big end) to bottom (small end) */
            idir = 1;
        } else {
            /* Chase bulge from bottom (big end) to top (small end) */
            idir = 2;
        }
    }

    /* Apply convergence tests */
    if (idir == 1) {
        /* Run convergence test in forward direction
         * First apply standard test to bottom of matrix */
        if (fabs(e[m - 2]) <= fabs(tol) * fabs(d[m - 1]) ||
            (tol < ZERO && fabs(e[m - 2]) <= thresh)) {
            e[m - 2] = ZERO;
            goto L60; /* GO TO 60 */
        }

        if (tol >= ZERO) {
            /* If relative accuracy desired,
             * apply convergence criterion forward */
            mu = fabs(d[ll - 1]);
            smin = mu;
            for (lll = ll; lll <= m - 1; lll++) { /* DO 100 */
                if (fabs(e[lll - 1]) <= tol * mu) {
                    e[lll - 1] = ZERO;
                    goto L60; /* GO TO 60 */
                }
                mu = fabs(d[lll]) * (mu / (mu + fabs(e[lll - 1])));
                smin = pyclap_bdsqr_dmin(smin, mu);
            }
        }

    } else {
        /* Run convergence test in backward direction
         * First apply standard test to top of matrix */
        if (fabs(e[ll - 1]) <= fabs(tol) * fabs(d[ll - 1]) ||
            (tol < ZERO && fabs(e[ll - 1]) <= thresh)) {
            e[ll - 1] = ZERO;
            goto L60; /* GO TO 60 */
        }

        if (tol >= ZERO) {
            /* If relative accuracy desired,
             * apply convergence criterion backward */
            mu = fabs(d[m - 1]);
            smin = mu;
            for (lll = m - 1; lll >= ll; lll--) { /* DO 110 */
                if (fabs(e[lll - 1]) <= tol * mu) {
                    e[lll - 1] = ZERO;
                    goto L60; /* GO TO 60 */
                }
                mu = fabs(d[lll - 1]) * (mu / (mu + fabs(e[lll - 1])));
                smin = pyclap_bdsqr_dmin(smin, mu);
            }
        }
    }
    oldll = ll;
    oldm = m;

    /* Compute shift.  First, test if shifting would ruin relative
     * accuracy, and if so set the shift to zero. */
    if (tol >= ZERO &&
        n * tol * (smin / smax) <= pyclap_bdsqr_dmax(eps, HNDRTH * tol)) {
        /* Use a zero shift to avoid loss of relative accuracy */
        shift = ZERO;
    } else {
        /* Compute the shift from 2-by-2 block at end of matrix */
        if (idir == 1) {
            sll = fabs(d[ll - 1]);
            pyclap_dlas2(d[m - 2], e[m - 2], d[m - 1], &shift, &r);
        } else {
            sll = fabs(d[m - 1]);
            pyclap_dlas2(d[ll - 1], e[ll - 1], d[ll], &shift, &r);
        }

        /* Test if shift negligible, and if so set to zero */
        if (sll > ZERO) {
            if ((shift / sll) * (shift / sll) < eps) /* ( SHIFT/SLL )**2 */
                shift = ZERO;
        }
    }

    /* Increment iteration count */
    iter = iter + m - ll;

    /* If SHIFT = 0, do simplified QR iteration */
    if (shift == ZERO) {
        if (idir == 1) {
            /* Chase bulge from top to bottom
             * Save cosines and sines for later singular vector updates */
            cs = ONE;
            oldcs = ONE;
            for (i = ll; i <= m - 1; i++) { /* DO 120 */
                pyclap_dlartg(d[i - 1] * cs, e[i - 1], &cs, &sn, &r);
                if (i > ll)
                    e[i - 2] = oldsn * r; /* E( I-1 ) = OLDSN*R */
                pyclap_dlartg(oldcs * r, d[i] * sn, &oldcs, &oldsn,
                              &d[i - 1]);
                work[i - ll] = cs;             /* WORK( I-LL+1 ) */
                work[i - ll + nm1] = sn;       /* WORK( I-LL+1+NM1 ) */
                work[i - ll + nm12] = oldcs;   /* WORK( I-LL+1+NM12 ) */
                work[i - ll + nm13] = oldsn;   /* WORK( I-LL+1+NM13 ) */
            }
            h = d[m - 1] * cs;
            d[m - 1] = h * oldcs;
            e[m - 2] = h * oldsn;

            /* Update singular vectors */
            if (ncvt > 0)
                pyclap_dlasr('L', 'V', 'F', m - ll + 1, ncvt, &work[0],
                             &work[n - 1], &vt[ll - 1], ldvt);
            if (nru > 0)
                pyclap_dlasr('R', 'V', 'F', nru, m - ll + 1, &work[nm12],
                             &work[nm13], &u[(ll - 1) * ldu], ldu);
            if (ncc > 0)
                pyclap_dlasr('L', 'V', 'F', m - ll + 1, ncc, &work[nm12],
                             &work[nm13], &c[ll - 1], ldc);

            /* Test convergence */
            if (fabs(e[m - 2]) <= thresh)
                e[m - 2] = ZERO;

        } else {
            /* Chase bulge from bottom to top
             * Save cosines and sines for later singular vector updates */
            cs = ONE;
            oldcs = ONE;
            for (i = m; i >= ll + 1; i--) { /* DO 130 */
                pyclap_dlartg(d[i - 1] * cs, e[i - 2], &cs, &sn, &r);
                if (i < m)
                    e[i - 1] = oldsn * r; /* E( I ) = OLDSN*R */
                pyclap_dlartg(oldcs * r, d[i - 2] * sn, &oldcs, &oldsn,
                              &d[i - 1]);
                work[i - ll - 1] = cs;              /* WORK( I-LL ) */
                work[i - ll + nm1 - 1] = -sn;       /* WORK( I-LL+NM1 ) */
                work[i - ll + nm12 - 1] = oldcs;    /* WORK( I-LL+NM12 ) */
                work[i - ll + nm13 - 1] = -oldsn;   /* WORK( I-LL+NM13 ) */
            }
            h = d[ll - 1] * cs;
            d[ll - 1] = h * oldcs;
            e[ll - 1] = h * oldsn;

            /* Update singular vectors */
            if (ncvt > 0)
                pyclap_dlasr('L', 'V', 'B', m - ll + 1, ncvt, &work[nm12],
                             &work[nm13], &vt[ll - 1], ldvt);
            if (nru > 0)
                pyclap_dlasr('R', 'V', 'B', nru, m - ll + 1, &work[0],
                             &work[n - 1], &u[(ll - 1) * ldu], ldu);
            if (ncc > 0)
                pyclap_dlasr('L', 'V', 'B', m - ll + 1, ncc, &work[0],
                             &work[n - 1], &c[ll - 1], ldc);

            /* Test convergence */
            if (fabs(e[ll - 1]) <= thresh)
                e[ll - 1] = ZERO;
        }
    } else {
        /* Use nonzero shift */
        if (idir == 1) {
            /* Chase bulge from top to bottom
             * Save cosines and sines for later singular vector updates */
            f = (fabs(d[ll - 1]) - shift) *
                (pyclap_bdsqr_dsign(ONE, d[ll - 1]) + shift / d[ll - 1]);
            g = e[ll - 1];
            for (i = ll; i <= m - 1; i++) { /* DO 140 */
                pyclap_dlartg(f, g, &cosr, &sinr, &r);
                if (i > ll)
                    e[i - 2] = r; /* E( I-1 ) = R */
                f = cosr * d[i - 1] + sinr * e[i - 1];
                e[i - 1] = cosr * e[i - 1] - sinr * d[i - 1];
                g = sinr * d[i];      /* G = SINR*D( I+1 ) */
                d[i] = cosr * d[i];   /* D( I+1 ) = COSR*D( I+1 ) */
                pyclap_dlartg(f, g, &cosl, &sinl, &r);
                d[i - 1] = r;
                f = cosl * e[i - 1] + sinl * d[i];
                d[i] = cosl * d[i] - sinl * e[i - 1];
                if (i < m - 1) {
                    g = sinl * e[i];    /* G = SINL*E( I+1 ) */
                    e[i] = cosl * e[i]; /* E( I+1 ) = COSL*E( I+1 ) */
                }
                work[i - ll] = cosr;           /* WORK( I-LL+1 ) */
                work[i - ll + nm1] = sinr;     /* WORK( I-LL+1+NM1 ) */
                work[i - ll + nm12] = cosl;    /* WORK( I-LL+1+NM12 ) */
                work[i - ll + nm13] = sinl;    /* WORK( I-LL+1+NM13 ) */
            }
            e[m - 2] = f; /* E( M-1 ) = F */

            /* Update singular vectors */
            if (ncvt > 0)
                pyclap_dlasr('L', 'V', 'F', m - ll + 1, ncvt, &work[0],
                             &work[n - 1], &vt[ll - 1], ldvt);
            if (nru > 0)
                pyclap_dlasr('R', 'V', 'F', nru, m - ll + 1, &work[nm12],
                             &work[nm13], &u[(ll - 1) * ldu], ldu);
            if (ncc > 0)
                pyclap_dlasr('L', 'V', 'F', m - ll + 1, ncc, &work[nm12],
                             &work[nm13], &c[ll - 1], ldc);

            /* Test convergence */
            if (fabs(e[m - 2]) <= thresh)
                e[m - 2] = ZERO;

        } else {
            /* Chase bulge from bottom to top
             * Save cosines and sines for later singular vector updates */
            f = (fabs(d[m - 1]) - shift) *
                (pyclap_bdsqr_dsign(ONE, d[m - 1]) + shift / d[m - 1]);
            g = e[m - 2]; /* G = E( M-1 ) */
            for (i = m; i >= ll + 1; i--) { /* DO 150 */
                pyclap_dlartg(f, g, &cosr, &sinr, &r);
                if (i < m)
                    e[i - 1] = r; /* E( I ) = R */
                f = cosr * d[i - 1] + sinr * e[i - 2];
                e[i - 2] = cosr * e[i - 2] - sinr * d[i - 1];
                g = sinr * d[i - 2];        /* G = SINR*D( I-1 ) */
                d[i - 2] = cosr * d[i - 2]; /* D( I-1 ) = COSR*D( I-1 ) */
                pyclap_dlartg(f, g, &cosl, &sinl, &r);
                d[i - 1] = r;
                f = cosl * e[i - 2] + sinl * d[i - 2];
                d[i - 2] = cosl * d[i - 2] - sinl * e[i - 2];
                if (i > ll + 1) {
                    g = sinl * e[i - 3];        /* G = SINL*E( I-2 ) */
                    e[i - 3] = cosl * e[i - 3]; /* E( I-2 ) = COSL*E( I-2 ) */
                }
                work[i - ll - 1] = cosr;            /* WORK( I-LL ) */
                work[i - ll + nm1 - 1] = -sinr;     /* WORK( I-LL+NM1 ) */
                work[i - ll + nm12 - 1] = cosl;     /* WORK( I-LL+NM12 ) */
                work[i - ll + nm13 - 1] = -sinl;    /* WORK( I-LL+NM13 ) */
            }
            e[ll - 1] = f; /* E( LL ) = F */

            /* Test convergence */
            if (fabs(e[ll - 1]) <= thresh)
                e[ll - 1] = ZERO;

            /* Update singular vectors if desired */
            if (ncvt > 0)
                pyclap_dlasr('L', 'V', 'B', m - ll + 1, ncvt, &work[nm12],
                             &work[nm13], &vt[ll - 1], ldvt);
            if (nru > 0)
                pyclap_dlasr('R', 'V', 'B', nru, m - ll + 1, &work[0],
                             &work[n - 1], &u[(ll - 1) * ldu], ldu);
            if (ncc > 0)
                pyclap_dlasr('L', 'V', 'B', m - ll + 1, ncc, &work[0],
                             &work[n - 1], &c[ll - 1], ldc);
        }
    }

    /* QR iteration finished, go back and check convergence */
    goto L60; /* GO TO 60 */

/* All singular values converged, so make them positive */
L160: /* Fortran label 160 */
    for (i = 1; i <= n; i++) { /* DO 170 */
        if (d[i - 1] == ZERO) {
            /* Avoid -ZERO */
            d[i - 1] = ZERO;
        }
        if (d[i - 1] < ZERO) {
            d[i - 1] = -d[i - 1];

            /* Change sign of singular vectors, if desired */
            if (ncvt > 0)
                pyclap_dscal(ncvt, NEGONE, &vt[i - 1], ldvt); /* VT( I, 1 ) */
        }
    }

    /* Sort the singular values into decreasing order (insertion sort on
     * singular values, but only one transposition per singular vector) */
    for (i = 1; i <= n - 1; i++) { /* DO 190 */
        /* Scan for smallest D(I) */
        isub = 1;
        smin = d[0]; /* SMIN = D( 1 ) */
        for (j = 2; j <= n + 1 - i; j++) { /* DO 180 */
            if (d[j - 1] <= smin) {
                isub = j;
                smin = d[j - 1];
            }
        }
        if (isub != n + 1 - i) {
            /* Swap singular values and vectors */
            d[isub - 1] = d[n - i]; /* D( ISUB ) = D( N+1-I ) */
            d[n - i] = smin;        /* D( N+1-I ) = SMIN */
            if (ncvt > 0)
                pyclap_dswap(ncvt, &vt[isub - 1], ldvt, &vt[n - i],
                             ldvt); /* VT( ISUB, 1 ), VT( N+1-I, 1 ) */
            if (nru > 0)
                pyclap_dswap(nru, &u[(isub - 1) * ldu], 1,
                             &u[(n - i) * ldu], 1); /* U( 1, ISUB ), U( 1, N+1-I ) */
            if (ncc > 0)
                pyclap_dswap(ncc, &c[isub - 1], ldc, &c[n - i],
                             ldc); /* C( ISUB, 1 ), C( N+1-I, 1 ) */
        }
    }
    goto L220; /* GO TO 220 */

/* Maximum number of iterations exceeded, failure to converge */
L200: /* Fortran label 200 */
    info = 0;
    for (i = 1; i <= n - 1; i++) { /* DO 210 */
        if (e[i - 1] != ZERO)
            info = info + 1;
    }
L220: /* Fortran label 220 */
    return info;
}

#endif /* PYCLAP_LAPACK_DBDSQR_H */
