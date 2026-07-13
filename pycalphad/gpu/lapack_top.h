/* lapack_top.h — top of the dgelsd transliteration chain.
 *
 * Faithful transliterations from Reference-LAPACK (master, fetched 2026-07-13)
 * of DLASDQ, DLALSD (base case) and DGELSD (square-matrix path), plus the
 * pycalphad reference-wrapper semantics (minimizer.pyx lstsq).
 *
 * Include order: lapack_leaf.h, lapack_dbdsqr.h, lapack_mid.h, THIS FILE.
 * All dependencies (pyclap_*) are `static` in those headers.
 *
 * Restrictions of this port (loud, not silent):
 *  - dlalsd: the divide-and-conquer section is not ported and is
 *    unreachable (dgelsd routes every n through the QR-iteration base
 *    case; above LAPACK's SMLSIZ=25 performance threshold this stays the
 *    correct LAPACK algorithm, merely not bitwise with a D&C build).
 *    PYCLAP_ERR_DC_UNPORTED remains as a defensive signal only.
 *  - dgelsd: square (m == n) only — the only shape the solver produces
 *    (construct_equilibrium_system raises on non-square). m < mnthr is
 *    implied by squareness (mnthr = int(1.6*n) > n), so the initial-QR
 *    branch is structurally unreachable and not ported.
 */

#define PYCLAP_ERR_DC_UNPORTED  (-1000)
#define PYCLAP_ERR_NOT_SQUARE   (-1001)

/* ------------------------------------------------------------------ */
/* DLASDQ (SRC/dlasdq.f) — SVD of a real bidiagonal matrix (with square
 * root extension), via implicit QR; here always reached with vectors. */
__device__ static int pyclap_dlasdq(char uplo, int sqre, int n, int ncvt,
                                    int nru, int ncc, double* d, double* e,
                                    double* vt, int ldvt, double* u, int ldu,
                                    double* c, int ldc, double* work)
{
    int info = 0;
    int iuplo = 0;
    if (pyclap_lsame(uplo, 'U')) iuplo = 1;
    if (pyclap_lsame(uplo, 'L')) iuplo = 2;
    /* argument validation dropped (XERBLA block); callers are internal */
    if (n == 0) return 0;

    int rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);
    int np1 = n + 1;
    int sqre1 = sqre;
    double cs, sn, r;
    int i;

    /* If matrix non-square upper bidiagonal, rotate to be lower bidiagonal. */
    if (iuplo == 1 && sqre1 == 1) {
        for (i = 1; i <= n - 1; ++i) {                       /* DO 10 */
            pyclap_dlartg(d[i-1], e[i-1], &cs, &sn, &r);
            d[i-1] = r;
            e[i-1] = sn * d[i];
            d[i]   = cs * d[i];
            if (rotate) { work[i-1] = cs; work[n+i-1] = sn; }
        }
        pyclap_dlartg(d[n-1], e[n-1], &cs, &sn, &r);
        d[n-1] = r;
        e[n-1] = 0.0;
        if (rotate) { work[n-1] = cs; work[n+n-1] = sn; }
        iuplo = 2;
        sqre1 = 0;
        if (ncvt > 0)
            pyclap_dlasr('L', 'V', 'F', np1, ncvt, &work[0], &work[np1-1],
                         vt, ldvt);
    }
    /* If matrix lower bidiagonal, rotate to be upper bidiagonal. */
    if (iuplo == 2) {
        for (i = 1; i <= n - 1; ++i) {                       /* DO 20 */
            pyclap_dlartg(d[i-1], e[i-1], &cs, &sn, &r);
            d[i-1] = r;
            e[i-1] = sn * d[i];
            d[i]   = cs * d[i];
            if (rotate) { work[i-1] = cs; work[n+i-1] = sn; }
        }
        if (sqre1 == 1) {
            pyclap_dlartg(d[n-1], e[n-1], &cs, &sn, &r);
            d[n-1] = r;
            if (rotate) { work[n-1] = cs; work[n+n-1] = sn; }
        }
        if (nru > 0) {
            if (sqre1 == 0)
                pyclap_dlasr('R', 'V', 'F', nru, n,   &work[0], &work[np1-1], u, ldu);
            else
                pyclap_dlasr('R', 'V', 'F', nru, np1, &work[0], &work[np1-1], u, ldu);
        }
        if (ncc > 0) {
            if (sqre1 == 0)
                pyclap_dlasr('L', 'V', 'F', n,   ncc, &work[0], &work[np1-1], c, ldc);
            else
                pyclap_dlasr('L', 'V', 'F', np1, ncc, &work[0], &work[np1-1], c, ldc);
        }
    }

    info = pyclap_dbdsqr('U', n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu,
                         c, ldc, work);
    if (info != 0) return info;

    /* Sort the singular values into ascending order (selection sort with
     * vector swaps).  DO 40 / DO 30. */
    for (i = 1; i <= n; ++i) {
        int isub = i;
        double smin = d[i-1];
        int j;
        for (j = i + 1; j <= n; ++j) {
            if (d[j-1] < smin) { isub = j; smin = d[j-1]; }
        }
        if (isub != i) {
            d[isub-1] = d[i-1];
            d[i-1] = smin;
            if (ncvt > 0) pyclap_dswap(ncvt, &vt[(isub-1)], ldvt, &vt[(i-1)], ldvt);
            if (nru  > 0) pyclap_dswap(nru,  &u[(isub-1)*ldu], 1, &u[(i-1)*ldu], 1);
            if (ncc  > 0) pyclap_dswap(ncc,  &c[(isub-1)], ldc, &c[(i-1)], ldc);
        }
    }
    return info;
}

/* ------------------------------------------------------------------ */
/* DLALSD (SRC/dlalsd.f) — minimum-norm solve of a bidiagonal LS problem.
 * Base case (n <= smlsiz) only; D&C section returns PYCLAP_ERR_DC_UNPORTED. */
__device__ static int pyclap_dlalsd(char uplo, int smlsiz, int n, int nrhs,
                                    double* d, double* e, double* b, int ldb,
                                    double rcond, int* rank, double* work,
                                    int* iwork)
{
    (void)iwork;  /* used only by the D&C section, unported */
    int info = 0;
    double eps = pyclap_dlamch('E');  /* 'Epsilon' */
    double rcnd;
    double cs, sn, r;
    int i, j;

    if (rcond <= 0.0 || rcond >= 1.0) rcnd = eps;
    else rcnd = rcond;

    *rank = 0;
    /* Quick return if possible. */
    if (n == 0) {
        return 0;
    } else if (n == 1) {
        if (d[0] == 0.0) {
            pyclap_dlaset('A', 1, nrhs, 0.0, 0.0, b, ldb);
        } else {
            *rank = 1;
            pyclap_dlascl('G', 0, 0, d[0], 1.0, 1, nrhs, b, ldb, &info);
            d[0] = fabs(d[0]);
        }
        return 0;
    }
    /* Rotate the matrix if it is lower bidiagonal. */
    if (pyclap_lsame(uplo, 'L')) {
        for (i = 1; i <= n - 1; ++i) {                        /* DO 10 */
            pyclap_dlartg(d[i-1], e[i-1], &cs, &sn, &r);
            d[i-1] = r;
            e[i-1] = sn * d[i];
            d[i]   = cs * d[i];
            if (nrhs == 1) {
                pyclap_drot(1, &b[i-1], 1, &b[i], 1, cs, sn);
            } else {
                work[i*2-2] = cs;
                work[i*2-1] = sn;
            }
        }
        if (nrhs > 1) {                                       /* DO 30 / DO 20 */
            for (i = 1; i <= nrhs; ++i) {
                for (j = 1; j <= n - 1; ++j) {
                    cs = work[j*2-2];
                    sn = work[j*2-1];
                    pyclap_drot(1, &b[(j-1)+(i-1)*ldb], 1, &b[j+(i-1)*ldb], 1, cs, sn);
                }
            }
        }
    }
    /* Scale. */
    {
        int nm1 = n - 1;
        double orgnrm = pyclap_dlanst('M', n, d, e);
        if (orgnrm == 0.0) {
            pyclap_dlaset('A', n, nrhs, 0.0, 0.0, b, ldb);
            return 0;
        }
        pyclap_dlascl('G', 0, 0, orgnrm, 1.0, n, 1, d, n, &info);
        pyclap_dlascl('G', 0, 0, orgnrm, 1.0, nm1, 1, e, nm1, &info);

        if (n <= smlsiz) {
            /* Solve the problem via QR-iteration SVD of the bidiagonal. */
            int nwork = 1 + n * n;             /* Fortran WORK(NWORK) */
            pyclap_dlaset('A', n, n, 0.0, 1.0, work, n);
            info = pyclap_dlasdq('U', 0, n, n, 0, nrhs, d, e, work, n,
                                 work, n, b, ldb, &work[nwork-1]);
            if (info != 0) return info;
            {
                double tol = rcnd * fabs(d[pyclap_idamax(n, d, 1) - 1]);
                for (i = 1; i <= n; ++i) {                    /* DO 40 */
                    if (d[i-1] <= tol) {
                        pyclap_dlaset('A', 1, nrhs, 0.0, 0.0, &b[i-1], ldb);
                    } else {
                        pyclap_dlascl('G', 0, 0, d[i-1], 1.0, 1, nrhs,
                                      &b[i-1], ldb, &info);
                        *rank = *rank + 1;
                    }
                }
                /* x = VT**T * (Sigma^-1 * Q**T * b): WORK holds VT (n x n). */
                pyclap_dgemm('T', 'N', n, nrhs, n, 1.0, work, n, b, ldb,
                             0.0, &work[nwork-1], n);
                pyclap_dlacpy('A', n, nrhs, &work[nwork-1], n, b, ldb);
                /* Unscale. */
                pyclap_dlascl('G', 0, 0, 1.0, orgnrm, n, 1, d, n, &info);
                pyclap_dlasrt('D', n, d, &info);
                pyclap_dlascl('G', 0, 0, orgnrm, 1.0, n, nrhs, b, ldb, &info);
            }
            return 0;
        }
    }
    /* Divide-and-conquer section (n > smlsiz): not ported. */
    return PYCLAP_ERR_DC_UNPORTED;
}

/* ------------------------------------------------------------------ */
/* DGELSD (SRC/dgelsd.f) — minimum-norm least squares via SVD (D&C).
 * Square path only (m == n): the m >= mnthr initial-QR branch is
 * structurally unreachable (mnthr = int(1.6*n) > n).
 *
 * A is column-major (lda >= m), overwritten. B (ldb >= m) holds the RHS on
 * entry and the solution on exit. S receives singular values.  WORK must
 * hold >= 3*n + max(2*n + n*n, wlalsd) doubles where
 * wlalsd = 9n + 2n*smlsiz + 8n + n + (smlsiz+1)^2 (nlvl=1 bound), i.e.
 * the caller passes a generous flat buffer. */
__device__ static int pyclap_dgelsd_sq(int m, int n, double* a, int lda,
                                       double* b, int ldb, double* s,
                                       double rcond, int* rank,
                                       double* work, int* iwork)
{
    int info = 0;
    if (m != n) return PYCLAP_ERR_NOT_SQUARE;
    /* ILAENV(9,'DGELSD') = 25 is LAPACK's PERFORMANCE threshold between the
     * QR-iteration base case and divide-and-conquer — not a validity limit.
     * For n <= 25 we take the identical path to LAPACK (bitwise); above it
     * we keep the (ported, bitwise-validated) base-case algorithm where the
     * library would switch to D&C for speed: correct LAPACK results, not
     * bitwise with a D&C build.  Real equilibrium systems are n ~ 4-15. */
    const int smlsiz = (n > 25) ? n : 25;
    int minmn = (m < n) ? m : n;
    if (minmn < 1) { *rank = 0; return 0; }
    const int nrhs = 1;             /* reference wrapper always NRHS=1 */

    /* Get machine parameters. */
    double eps    = pyclap_dlamch('P');
    double sfmin  = pyclap_dlamch('S');
    double smlnum = sfmin / eps;
    double bignum = 1.0 / smlnum;

    /* Scale A if max entry outside range [SMLNUM,BIGNUM]. */
    double anrm = pyclap_dlange('M', m, n, a, lda, work);
    int iascl = 0;
    if (anrm > 0.0 && anrm < smlnum) {
        pyclap_dlascl('G', 0, 0, anrm, smlnum, m, n, a, lda, &info);
        iascl = 1;
    } else if (anrm > bignum) {
        pyclap_dlascl('G', 0, 0, anrm, bignum, m, n, a, lda, &info);
        iascl = 2;
    } else if (anrm == 0.0) {
        /* Matrix all zero. Return zero solution. */
        pyclap_dlaset('F', (m > n ? m : n), nrhs, 0.0, 0.0, b, ldb);
        pyclap_dlaset('F', minmn, 1, 0.0, 0.0, s, 1);
        *rank = 0;
        return 0;                                    /* GO TO 10 */
    }

    /* Scale B if max entry outside range [SMLNUM,BIGNUM]. */
    double bnrm = pyclap_dlange('M', m, nrhs, b, ldb, work);
    int ibscl = 0;
    if (bnrm > 0.0 && bnrm < smlnum) {
        pyclap_dlascl('G', 0, 0, bnrm, smlnum, m, nrhs, b, ldb, &info);
        ibscl = 1;
    } else if (bnrm > bignum) {
        pyclap_dlascl('G', 0, 0, bnrm, bignum, m, nrhs, b, ldb, &info);
        ibscl = 2;
    }

    /* Path 1: m >= n, mm = m (no initial QR: m < mnthr for square). */
    {
        int mm = m;
        int ie    = 1;              /* Fortran workspace indices (1-based) */
        int itauq = ie + n;
        int itaup = itauq + n;
        int nwork = itaup + n;
        int lwork_rem;  /* LWORK - NWORK + 1 stand-in: give a big number */
        lwork_rem = 1 << 28;

        /* Bidiagonalize A: reduce to upper bidiagonal form. */
        pyclap_dgebrd(mm, n, a, lda, s, &work[ie-1], &work[itauq-1],
                      &work[itaup-1], &work[nwork-1], lwork_rem, &info);
        if (info != 0) return info;
        /* Multiply B by transpose of left bidiagonalizing vectors of R. */
        pyclap_dormbr('Q', 'L', 'T', mm, nrhs, n, a, lda,
                      &work[itauq-1], b, ldb, &work[nwork-1], lwork_rem, &info);
        if (info != 0) return info;
        /* Solve the bidiagonal least squares problem. */
        info = pyclap_dlalsd('U', smlsiz, n, nrhs, s, &work[ie-1], b, ldb,
                             rcond, rank, &work[nwork-1], iwork);
        if (info != 0) return info;                  /* incl. DC_UNPORTED */
        /* Multiply B by right bidiagonalizing vectors of R. */
        pyclap_dormbr('P', 'L', 'N', n, nrhs, n, a, lda,
                      &work[itaup-1], b, ldb, &work[nwork-1], lwork_rem, &info);
        if (info != 0) return info;
    }

    /* Undo scaling. */
    if (iascl == 1) {
        pyclap_dlascl('G', 0, 0, anrm, smlnum, n, nrhs, b, ldb, &info);
        pyclap_dlascl('G', 0, 0, smlnum, anrm, minmn, 1, s, minmn, &info);
    } else if (iascl == 2) {
        pyclap_dlascl('G', 0, 0, anrm, bignum, n, nrhs, b, ldb, &info);
        pyclap_dlascl('G', 0, 0, bignum, anrm, minmn, 1, s, minmn, &info);
    }
    if (ibscl == 1) {
        pyclap_dlascl('G', 0, 0, smlnum, bnrm, n, nrhs, b, ldb, &info);
    } else if (ibscl == 2) {
        pyclap_dlascl('G', 0, 0, bignum, bnrm, n, nrhs, b, ldb, &info);
    }
    return info;                                     /* label 10 */
}

/* ------------------------------------------------------------------ */
/* pycalphad reference-wrapper semantics (minimizer.pyx lstsq):
 *  - any NaN in A  -> x = 0 (no solve)
 *  - dgelsd info != 0 (incl. unported paths) -> x = -1e19 sentinels
 *  - rcond = 1e-16 (RELATIVE to largest singular value inside dlalsd)
 *
 * The kernel stores the equilibrium matrix ROW-major; the reference builds
 * it column-major (order='F').  a_colmajor is a caller-provided n*n
 * scratch buffer; the transpose-copy below converts layout, and because
 * the reference passes the SAME logical matrix, bitwise agreement is
 * preserved (no arithmetic in the copy).
 *
 * Returns 0 on success, or the dgelsd info (callers may fall back to the
 * legacy path on PYCLAP_ERR_* codes). b (length >= m) holds the solution
 * in its first n entries on success. */
__device__ static int pyclap_lstsq_pycalphad(const double* a_rowmajor, int m,
                                             int n, double* b, double rcond,
                                             double* a_colmajor, double* s,
                                             double* work, int* iwork)
{
    int i, j;
    int isfinite = 1;
    for (i = 0; i < m * n; ++i) {
        if (a_rowmajor[i] != a_rowmajor[i]) isfinite = 0;   /* NaN scrub */
    }
    if (!isfinite) {
        for (i = 0; i < n; ++i) b[i] = 0.0;
        return 0;
    }
    for (j = 0; j < n; ++j)
        for (i = 0; i < m; ++i)
            a_colmajor[i + j * m] = a_rowmajor[i * n + j];

    int rank = 0;
    int info = pyclap_dgelsd_sq(m, n, a_colmajor, m, b, m, s, rcond, &rank,
                                work, iwork);
    if (info != 0) {
        for (i = 0; i < n; ++i) b[i] = -1e19;
    }
    return info;
}

/* ==== LAPACK LU inversion chain (dgetrf/dgetri; folded, see lapack_lu
 * provenance in the transliteration report) ==== */
/*
 * lapack_lu.h -- faithful C99 transliteration of the reference LAPACK
 * LU-factorization / inversion chain (netlib master, fetched 2026), for
 * the pycalphad GPU/CPU-shared kernel backend.
 *
 * MUST be #include'd AFTER lapack_leaf.h and lapack_mid.h; calls
 * pyclap_lsame, pyclap_imax/imin, pyclap_disnan, pyclap_dlamch,
 * pyclap_idamax (1-based result), pyclap_dscal, pyclap_dswap,
 * pyclap_dger, pyclap_dgemv, pyclap_dtrmv, pyclap_dgemm, and the
 * pyclap_ilaenv_eq2/eq3 name-field helpers from there.
 *
 * Conventions (identical to lapack_leaf.h / lapack_mid.h):
 *   - Column-major layout preserved: A(I,J) -> a[(I-1) + (J-1)*lda].
 *   - Loop variables kept 1-based to mirror the Fortran bounds/direction
 *     exactly; indexing subtracts 1 at the access site.
 *   - Numerical operation ORDER matches the Fortran statement-for-statement
 *     (goal: bitwise-identical results). Only XERBLA/INFO<0 argument
 *     validation and the LWORK==-1 workspace-query branch were dropped
 *     (callers in this port never pass LWORK=-1 or invalid arguments).
 *   - Every GOTO converted to explicit control flow is documented with a
 *     comment naming the Fortran label. (This chain has no GOTOs other
 *     than the computed GO TO in ILAENV; loop labels are noted inline.)
 *   - Blocked code paths whose BLAS dependencies this port carries
 *     (DGETRF blocked, DGETRI blocked) are fully transliterated even
 *     where unreachable for this port's sizes; the one blocked path that
 *     would pull in DTRMM (DTRTRI blocked) is PROVABLY unreachable for
 *     n <= 64 and is replaced by a loud `*info = -999` guard (see the
 *     per-routine proof).
 *   - No dynamic allocation, no printf, C99 only. All functions are
 *     `__device__ static`.
 *
 * Vintage note: this LAPACK vintage's DGETRF routes its unblocked case
 * to the RECURSIVE DGETRF2 (dgetrf2.f, netlib master), NOT to the
 * classic level-2 DGETF2. Both are transliterated; DGETF2 is retained
 * because it is part of the published source set, but nothing in this
 * chain calls it. DGETRF2 recurses to depth <= ceil(log2(64)) + 1 for
 * this port's sizes (n <= 64); each frame holds a handful of scalars.
 *
 * Routines: pyclap_ilaenv_lu (local ILAENV subset), dlaswp, dtrsm,
 *           dgetf2, dgetrf2 (recursive), dgetrf, dtrti2, dtrtri,
 *           dgetri, and the pycalphad reference wrapper
 *           pyclap_invert_pycalphad (minimizer.pyx invert_matrix).
 */
#ifndef PYCLAP_LAPACK_LU_H
#define PYCLAP_LAPACK_LU_H

/* ------------------------------------------------------------------ */
/* ILAENV (LAPACK master 2026), local LU-chain subset: the sibling     */
/* pyclap_ilaenv in lapack_mid.h does not carry the TRF/TRI cases, so  */
/* the exact queries made by this chain are transliterated here:       */
/*   ISPEC=1 (NB, label 50):    GE+TRF -> 64 (S and D branches both    */
/*                              64), GE+TRI -> 64, TR+TRI -> 64;       */
/*                              default NB = 1.                        */
/*   ISPEC=2 (NBMIN, label 60): GE+TRI -> 2; default NBMIN = 2.        */
/* Verified against ilaenv.f lines 279-285 (GE TRF), 351-356 (GE TRI), */
/* 464-470 (TR TRI), 543-548 (ISPEC=2 GE TRI).                         */
/* NAME is assumed uppercase ASCII (this port passes string literals); */
/* the reference's lowercase/EBCDIC normalization (labels 10-40) is    */
/* dropped exactly as in pyclap_ilaenv.                                */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_ilaenv_lu(int ispec, const char* name,
                                       const char* opts, int n1, int n2,
                                       int n3, int n4)
{
    int nb, nbmin, sname, cname;
    char c1;
    const char *c2, *c3;
    (void)opts;   /* OPTS never participates in the TRF/TRI cases */
    (void)n1;
    (void)n2;
    (void)n3;
    (void)n4;

    /* Fortran: GO TO ( 10, 10, 10, 80, ... )ISPEC.  The LU chain only
       ever queries ISPEC = 1 and ISPEC = 2 (both route to label 10). */
    if (ispec != 1 && ispec != 2)
        return -1;                             /* invalid for this subset */

    /* label 10: ILAENV = 1 (folded into the defaults below). */
    c1 = name[0];
    sname = (c1 == 'S' || c1 == 'D');
    cname = (c1 == 'C' || c1 == 'Z');
    if (!(cname || sname))
        return 1;
    c2 = name + 1;      /* SUBNAM(2:3) */
    c3 = name + 3;      /* SUBNAM(4:6) */

    /* Fortran: GO TO ( 50, 60, 70 )ISPEC */
    if (ispec == 1) {
        /* label 50: ISPEC = 1: block size */
        nb = 1;
        if (pyclap_ilaenv_eq2(c2, "GE")) {
            if (pyclap_ilaenv_eq3(c3, "TRF")) {
                if (sname) {
                    nb = 64;
                } else {
                    nb = 64;
                }
            } else if (pyclap_ilaenv_eq3(c3, "TRI")) {
                if (sname) {
                    nb = 64;
                } else {
                    nb = 64;
                }
            }
        } else if (pyclap_ilaenv_eq2(c2, "TR")) {
            if (pyclap_ilaenv_eq3(c3, "TRI")) {
                if (sname) {
                    nb = 64;
                } else {
                    nb = 64;
                }
            }
        }
        return nb;
    }
    /* label 60: ISPEC = 2: minimum block size */
    nbmin = 2;
    if (pyclap_ilaenv_eq2(c2, "GE")) {
        if (pyclap_ilaenv_eq3(c3, "TRI")) {
            if (sname) {
                nbmin = 2;
            } else {
                nbmin = 2;
            }
        }
    }
    return nbmin;
}

/* ------------------------------------------------------------------ */
/* DLASWP (LAPACK master 2026): performs a series of row interchanges  */
/* on the matrix A; one interchange for each of rows K1 through K2.    */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlaswp(int n, double* a, int lda, int k1,
                                     int k2, const int* ipiv, int incx)
{
    int i, i1, i2, inc, ip, ix, ix0, j, k, n32;
    double temp;

    /* Interchange row I with row IPIV(K1+(I-K1)*abs(INCX)) for each of
       rows K1 through K2. */
    if (incx > 0) {
        ix0 = k1;
        i1 = k1;
        i2 = k2;
        inc = 1;
    } else if (incx < 0) {
        ix0 = k1 + (k1 - k2) * incx;
        i1 = k2;
        i2 = k1;
        inc = -1;
    } else {
        return;
    }

    n32 = (n / 32) * 32;
    if (n32 != 0) {
        for (j = 1; j <= n32; j += 32) {                     /* DO 30 */
            ix = ix0;
            /* DO 20 I = I1, I2, INC */
            for (i = i1; (inc > 0) ? (i <= i2) : (i >= i2); i += inc) {
                ip = ipiv[ix - 1];
                if (ip != i) {
                    for (k = j; k <= j + 31; k++) {          /* DO 10 */
                        temp = a[(i - 1) + (k - 1) * lda];
                        a[(i - 1) + (k - 1) * lda] =
                            a[(ip - 1) + (k - 1) * lda];
                        a[(ip - 1) + (k - 1) * lda] = temp;
                    }
                }
                ix = ix + incx;
            }
        }
    }
    if (n32 != n) {
        n32 = n32 + 1;
        ix = ix0;
        /* DO 50 I = I1, I2, INC */
        for (i = i1; (inc > 0) ? (i <= i2) : (i >= i2); i += inc) {
            ip = ipiv[ix - 1];
            if (ip != i) {
                for (k = n32; k <= n; k++) {                 /* DO 40 */
                    temp = a[(i - 1) + (k - 1) * lda];
                    a[(i - 1) + (k - 1) * lda] =
                        a[(ip - 1) + (k - 1) * lda];
                    a[(ip - 1) + (k - 1) * lda] = temp;
                }
            }
            ix = ix + incx;
        }
    }
}

/* ------------------------------------------------------------------ */
/* DTRSM (reference BLAS level 3, master 2026): solves                 */
/*   op( A )*X = alpha*B   or   X*op( A ) = alpha*B,                   */
/* X overwriting B.  ALL side/uplo/trans/diag cases transliterated     */
/* (the LU chain reaches 'L','L','N','U' from DGETRF2 and, on the      */
/* unreachable-but-kept blocked paths, 'L','L','N','U' from DGETRF and */
/* 'R','L','N','U' from DGETRI).  TRANSA = 'C' is treated identically  */
/* to 'T' exactly as in the reference (both fail the LSAME(TRANSA,'N') */
/* test).  Loop labels 10..360 are the reference's DO loops, preserved */
/* with identical nesting and accumulation order.                      */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dtrsm(char side, char uplo, char transa,
                                    char diag, int m, int n, double alpha,
                                    const double* a, int lda, double* b,
                                    int ldb)
{
    const double zero = 0.0;
    double temp;
    int i, j, k;
    int lside, nounit, upper;

    lside = pyclap_lsame(side, 'L');
    /* NROWA only feeds the dropped argument validation. */
    nounit = pyclap_lsame(diag, 'N');
    upper = pyclap_lsame(uplo, 'U');

    /* Quick return if possible. */
    if (m == 0 || n == 0) return;

    /* And when alpha.eq.zero. */
    if (alpha == zero) {
        for (j = 1; j <= n; j++) {                           /* DO 20 */
            for (i = 1; i <= m; i++) {                       /* DO 10 */
                b[(i - 1) + (j - 1) * ldb] = zero;
            }
        }
        return;
    }

    /* Start the operations. */
    if (lside) {
        if (pyclap_lsame(transa, 'N')) {
            /* Form  B := alpha*inv( A )*B. */
            if (upper) {
                for (j = 1; j <= n; j++) {                   /* DO 60 */
                    for (i = 1; i <= m; i++) {               /* DO 30 */
                        b[(i - 1) + (j - 1) * ldb] =
                            alpha * b[(i - 1) + (j - 1) * ldb];
                    }
                    for (k = m; k >= 1; k--) {               /* DO 50 */
                        if (nounit)
                            b[(k - 1) + (j - 1) * ldb] =
                                b[(k - 1) + (j - 1) * ldb] /
                                a[(k - 1) + (k - 1) * lda];
                        for (i = 1; i <= k - 1; i++) {       /* DO 40 */
                            b[(i - 1) + (j - 1) * ldb] =
                                b[(i - 1) + (j - 1) * ldb] -
                                b[(k - 1) + (j - 1) * ldb] *
                                    a[(i - 1) + (k - 1) * lda];
                        }
                    }
                }
            } else {
                for (j = 1; j <= n; j++) {                   /* DO 100 */
                    for (i = 1; i <= m; i++) {               /* DO 70 */
                        b[(i - 1) + (j - 1) * ldb] =
                            alpha * b[(i - 1) + (j - 1) * ldb];
                    }
                    for (k = 1; k <= m; k++) {               /* DO 90 */
                        if (nounit)
                            b[(k - 1) + (j - 1) * ldb] =
                                b[(k - 1) + (j - 1) * ldb] /
                                a[(k - 1) + (k - 1) * lda];
                        for (i = k + 1; i <= m; i++) {       /* DO 80 */
                            b[(i - 1) + (j - 1) * ldb] =
                                b[(i - 1) + (j - 1) * ldb] -
                                b[(k - 1) + (j - 1) * ldb] *
                                    a[(i - 1) + (k - 1) * lda];
                        }
                    }
                }
            }
        } else {
            /* Form  B := alpha*inv( A**T )*B. */
            if (upper) {
                for (j = 1; j <= n; j++) {                   /* DO 130 */
                    for (i = 1; i <= m; i++) {               /* DO 120 */
                        temp = alpha * b[(i - 1) + (j - 1) * ldb];
                        for (k = 1; k <= i - 1; k++) {       /* DO 110 */
                            temp = temp - a[(k - 1) + (i - 1) * lda] *
                                              b[(k - 1) + (j - 1) * ldb];
                        }
                        if (nounit)
                            temp = temp / a[(i - 1) + (i - 1) * lda];
                        b[(i - 1) + (j - 1) * ldb] = temp;
                    }
                }
            } else {
                for (j = 1; j <= n; j++) {                   /* DO 160 */
                    for (i = m; i >= 1; i--) {               /* DO 150 */
                        temp = alpha * b[(i - 1) + (j - 1) * ldb];
                        for (k = i + 1; k <= m; k++) {       /* DO 140 */
                            temp = temp - a[(k - 1) + (i - 1) * lda] *
                                              b[(k - 1) + (j - 1) * ldb];
                        }
                        if (nounit)
                            temp = temp / a[(i - 1) + (i - 1) * lda];
                        b[(i - 1) + (j - 1) * ldb] = temp;
                    }
                }
            }
        }
    } else {
        if (pyclap_lsame(transa, 'N')) {
            /* Form  B := alpha*B*inv( A ). */
            if (upper) {
                for (j = 1; j <= n; j++) {                   /* DO 210 */
                    for (i = 1; i <= m; i++) {               /* DO 170 */
                        b[(i - 1) + (j - 1) * ldb] =
                            alpha * b[(i - 1) + (j - 1) * ldb];
                    }
                    for (k = 1; k <= j - 1; k++) {           /* DO 190 */
                        for (i = 1; i <= m; i++) {           /* DO 180 */
                            b[(i - 1) + (j - 1) * ldb] =
                                b[(i - 1) + (j - 1) * ldb] -
                                a[(k - 1) + (j - 1) * lda] *
                                    b[(i - 1) + (k - 1) * ldb];
                        }
                    }
                    if (nounit) {
                        for (i = 1; i <= m; i++) {           /* DO 200 */
                            b[(i - 1) + (j - 1) * ldb] =
                                b[(i - 1) + (j - 1) * ldb] /
                                a[(j - 1) + (j - 1) * lda];
                        }
                    }
                }
            } else {
                for (j = n; j >= 1; j--) {                   /* DO 260 */
                    for (i = 1; i <= m; i++) {               /* DO 220 */
                        b[(i - 1) + (j - 1) * ldb] =
                            alpha * b[(i - 1) + (j - 1) * ldb];
                    }
                    for (k = j + 1; k <= n; k++) {           /* DO 240 */
                        for (i = 1; i <= m; i++) {           /* DO 230 */
                            b[(i - 1) + (j - 1) * ldb] =
                                b[(i - 1) + (j - 1) * ldb] -
                                a[(k - 1) + (j - 1) * lda] *
                                    b[(i - 1) + (k - 1) * ldb];
                        }
                    }
                    if (nounit) {
                        for (i = 1; i <= m; i++) {           /* DO 250 */
                            b[(i - 1) + (j - 1) * ldb] =
                                b[(i - 1) + (j - 1) * ldb] /
                                a[(j - 1) + (j - 1) * lda];
                        }
                    }
                }
            }
        } else {
            /* Form  B := alpha*B*inv( A**T ). */
            if (upper) {
                for (k = n; k >= 1; k--) {                   /* DO 310 */
                    if (nounit) {
                        for (i = 1; i <= m; i++) {           /* DO 270 */
                            b[(i - 1) + (k - 1) * ldb] =
                                b[(i - 1) + (k - 1) * ldb] /
                                a[(k - 1) + (k - 1) * lda];
                        }
                    }
                    for (j = 1; j <= k - 1; j++) {           /* DO 290 */
                        for (i = 1; i <= m; i++) {           /* DO 280 */
                            b[(i - 1) + (j - 1) * ldb] =
                                b[(i - 1) + (j - 1) * ldb] -
                                a[(j - 1) + (k - 1) * lda] *
                                    b[(i - 1) + (k - 1) * ldb];
                        }
                    }
                    for (i = 1; i <= m; i++) {               /* DO 300 */
                        b[(i - 1) + (k - 1) * ldb] =
                            alpha * b[(i - 1) + (k - 1) * ldb];
                    }
                }
            } else {
                for (k = 1; k <= n; k++) {                   /* DO 360 */
                    if (nounit) {
                        for (i = 1; i <= m; i++) {           /* DO 320 */
                            b[(i - 1) + (k - 1) * ldb] =
                                b[(i - 1) + (k - 1) * ldb] /
                                a[(k - 1) + (k - 1) * lda];
                        }
                    }
                    for (j = k + 1; j <= n; j++) {           /* DO 340 */
                        for (i = 1; i <= m; i++) {           /* DO 330 */
                            b[(i - 1) + (j - 1) * ldb] =
                                b[(i - 1) + (j - 1) * ldb] -
                                a[(j - 1) + (k - 1) * lda] *
                                    b[(i - 1) + (k - 1) * ldb];
                        }
                    }
                    for (i = 1; i <= m; i++) {               /* DO 350 */
                        b[(i - 1) + (k - 1) * ldb] =
                            alpha * b[(i - 1) + (k - 1) * ldb];
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* DGETF2 (LAPACK master 2026): LU factorization with partial pivoting */
/* of a general m-by-n matrix, right-looking level-2 BLAS version.     */
/* NOTE: nothing in this vintage's chain calls DGETF2 (DGETRF routes   */
/* its unblocked case to the recursive DGETRF2 below); it is           */
/* transliterated because it belongs to the published source set.      */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgetf2(int m, int n, double* a, int lda,
                                     int* ipiv, int* info)
{
    const double one = 1.0, zero = 0.0;
    double sfmin;
    int i, j, jp;

    *info = 0;
    /* (XERBLA argument validation dropped) */

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return;

    /* Compute machine safe minimum */
    sfmin = pyclap_dlamch('S');

    for (j = 1; j <= pyclap_imin(m, n); j++) {               /* DO 10 */
        /* Find pivot and test for singularity. */
        jp = j - 1 + pyclap_idamax(m - j + 1,
                                   &a[(j - 1) + (j - 1) * lda], 1);
        ipiv[j - 1] = jp;
        if (a[(jp - 1) + (j - 1) * lda] != zero) {
            /* Apply the interchange to columns 1:N. */
            if (jp != j)
                pyclap_dswap(n, &a[(j - 1)], lda, &a[(jp - 1)], lda);
            /* Compute elements J+1:M of J-th column. */
            if (j < m) {
                if (fabs(a[(j - 1) + (j - 1) * lda]) >= sfmin) {
                    pyclap_dscal(m - j, one / a[(j - 1) + (j - 1) * lda],
                                 &a[j + (j - 1) * lda], 1);
                } else {
                    for (i = 1; i <= m - j; i++) {           /* DO 20 */
                        a[(j + i - 1) + (j - 1) * lda] =
                            a[(j + i - 1) + (j - 1) * lda] /
                            a[(j - 1) + (j - 1) * lda];
                    }
                }
            }
        } else if (*info == 0) {
            *info = j;
        }
        if (j < pyclap_imin(m, n)) {
            /* Update trailing submatrix. */
            pyclap_dger(m - j, n - j, -one, &a[j + (j - 1) * lda], 1,
                        &a[(j - 1) + j * lda], lda, &a[j + j * lda],
                        lda);
        }
    }   /* 10 CONTINUE */
}

/* ------------------------------------------------------------------ */
/* DGETRF2 (LAPACK master 2026): LU factorization with partial         */
/* pivoting, recursive version.  RECURSIVE SUBROUTINE in the Fortran;  */
/* transliterated as a recursive C function (depth <= ~log2(n)+1;      */
/* n <= 64 in this port, each frame holds a handful of scalars).       */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgetrf2(int m, int n, double* a, int lda,
                                      int* ipiv, int* info)
{
    const double one = 1.0, zero = 0.0;
    double sfmin, temp;
    int i, iinfo, n1, n2;

    *info = 0;
    /* (XERBLA argument validation dropped) */

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return;

    if (m == 1) {
        /* Use unblocked code for one row case.
           Just need to handle IPIV and INFO. */
        ipiv[0] = 1;
        if (a[0] == zero)
            *info = 1;
    } else if (n == 1) {
        /* Use unblocked code for one column case. */
        /* Compute machine safe minimum */
        sfmin = pyclap_dlamch('S');
        /* Find pivot and test for singularity */
        i = pyclap_idamax(m, &a[0], 1);
        ipiv[0] = i;
        if (a[i - 1] != zero) {
            /* Apply the interchange */
            if (i != 1) {
                temp = a[0];
                a[0] = a[i - 1];
                a[i - 1] = temp;
            }
            /* Compute elements 2:M of the column */
            if (fabs(a[0]) >= sfmin) {
                pyclap_dscal(m - 1, one / a[0], &a[1], 1);
            } else {
                for (i = 1; i <= m - 1; i++) {               /* DO 10 */
                    a[i] = a[i] / a[0];      /* A(1+I,1) = A(1+I,1)/A(1,1) */
                }
            }
        } else {
            *info = 1;
        }
    } else {
        /* Use recursive code */
        n1 = pyclap_imin(m, n) / 2;
        n2 = n - n1;

        /*        [ A11 ]
           Factor [ --- ]
                  [ A21 ]   */
        pyclap_dgetrf2(m, n1, a, lda, ipiv, &iinfo);
        if (*info == 0 && iinfo > 0)
            *info = iinfo;

        /*                       [ A12 ]
           Apply interchanges to [ --- ]
                                 [ A22 ]   */
        pyclap_dlaswp(n2, &a[n1 * lda], lda, 1, n1, ipiv, 1);

        /* Solve A12 */
        pyclap_dtrsm('L', 'L', 'N', 'U', n1, n2, one, a, lda,
                     &a[n1 * lda], lda);

        /* Update A22 */
        pyclap_dgemm('N', 'N', m - n1, n2, n1, -one, &a[n1], lda,
                     &a[n1 * lda], lda, one, &a[n1 + n1 * lda], lda);

        /* Factor A22 */
        pyclap_dgetrf2(m - n1, n2, &a[n1 + n1 * lda], lda, &ipiv[n1],
                       &iinfo);

        /* Adjust INFO and the pivot indices */
        if (*info == 0 && iinfo > 0)
            *info = iinfo + n1;
        for (i = n1 + 1; i <= pyclap_imin(m, n); i++) {      /* DO 20 */
            ipiv[i - 1] = ipiv[i - 1] + n1;
        }

        /* Apply interchanges to A21 */
        pyclap_dlaswp(n1, &a[0], lda, n1 + 1, pyclap_imin(m, n), ipiv, 1);
    }
}

/* ------------------------------------------------------------------ */
/* DGETRF (LAPACK master 2026): LU factorization with partial          */
/* pivoting, right-looking level-3 BLAS driver.  The blocked branch is */
/* fully transliterated even though it is unreachable in this port:    */
/* NB = ILAENV(1,'DGETRF') = 64 and all callers have MIN(M,N) <= 64,   */
/* so NB >= MIN(M,N) always routes to the unblocked DGETRF2 call.      */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgetrf(int m, int n, double* a, int lda,
                                     int* ipiv, int* info)
{
    const double one = 1.0;
    int i, iinfo, j, jb, nb;

    *info = 0;
    /* (XERBLA argument validation dropped) */

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return;

    /* Determine the block size for this environment. */
    nb = pyclap_ilaenv_lu(1, "DGETRF", " ", m, n, -1, -1);
    if (nb <= 1 || nb >= pyclap_imin(m, n)) {
        /* Use unblocked code. */
        pyclap_dgetrf2(m, n, a, lda, ipiv, info);
    } else {
        /* Use blocked code.  (Unreachable for MIN(M,N) <= 64, see the
           routine comment; kept as a faithful transliteration.) */
        for (j = 1; j <= pyclap_imin(m, n); j += nb) {       /* DO 20 */
            jb = pyclap_imin(pyclap_imin(m, n) - j + 1, nb);

            /* Factor diagonal and subdiagonal blocks and test for exact
               singularity. */
            pyclap_dgetrf2(m - j + 1, jb, &a[(j - 1) + (j - 1) * lda],
                           lda, &ipiv[j - 1], &iinfo);

            /* Adjust INFO and the pivot indices. */
            if (*info == 0 && iinfo > 0)
                *info = iinfo + j - 1;
            for (i = j; i <= pyclap_imin(m, j + jb - 1); i++) { /* DO 10 */
                ipiv[i - 1] = j - 1 + ipiv[i - 1];
            }

            /* Apply interchanges to columns 1:J-1. */
            pyclap_dlaswp(j - 1, a, lda, j, j + jb - 1, ipiv, 1);

            if (j + jb <= n) {
                /* Apply interchanges to columns J+JB:N. */
                pyclap_dlaswp(n - j - jb + 1, &a[(j + jb - 1) * lda],
                              lda, j, j + jb - 1, ipiv, 1);

                /* Compute block row of U. */
                pyclap_dtrsm('L', 'L', 'N', 'U', jb, n - j - jb + 1, one,
                             &a[(j - 1) + (j - 1) * lda], lda,
                             &a[(j - 1) + (j + jb - 1) * lda], lda);
                if (j + jb <= m) {
                    /* Update trailing submatrix. */
                    pyclap_dgemm('N', 'N', m - j - jb + 1,
                                 n - j - jb + 1, jb, -one,
                                 &a[(j + jb - 1) + (j - 1) * lda], lda,
                                 &a[(j - 1) + (j + jb - 1) * lda], lda,
                                 one,
                                 &a[(j + jb - 1) + (j + jb - 1) * lda],
                                 lda);
                }
            }
        }   /* 20 CONTINUE */
    }
}

/* ------------------------------------------------------------------ */
/* DTRTI2 (LAPACK master 2026): inverse of a triangular matrix,        */
/* unblocked level-2 BLAS version.  Both UPLO branches transliterated  */
/* (the LU chain reaches 'U','N' via DTRTRI from DGETRI).              */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dtrti2(char uplo, char diag, int n,
                                     double* a, int lda, int* info)
{
    const double one = 1.0;
    double ajj;
    int j;
    int nounit, upper;

    *info = 0;
    upper = pyclap_lsame(uplo, 'U');
    nounit = pyclap_lsame(diag, 'N');
    /* (XERBLA argument validation dropped) */

    if (upper) {
        /* Compute inverse of upper triangular matrix. */
        for (j = 1; j <= n; j++) {                           /* DO 10 */
            if (nounit) {
                a[(j - 1) + (j - 1) * lda] =
                    one / a[(j - 1) + (j - 1) * lda];
                ajj = -a[(j - 1) + (j - 1) * lda];
            } else {
                ajj = -one;
            }
            /* Compute elements 1:j-1 of j-th column. */
            pyclap_dtrmv('U', 'N', diag, j - 1, a, lda,
                         &a[(j - 1) * lda], 1);
            pyclap_dscal(j - 1, ajj, &a[(j - 1) * lda], 1);
        }
    } else {
        /* Compute inverse of lower triangular matrix. */
        for (j = n; j >= 1; j--) {                           /* DO 20 */
            if (nounit) {
                a[(j - 1) + (j - 1) * lda] =
                    one / a[(j - 1) + (j - 1) * lda];
                ajj = -a[(j - 1) + (j - 1) * lda];
            } else {
                ajj = -one;
            }
            if (j < n) {
                /* Compute elements j+1:n of j-th column. */
                pyclap_dtrmv('L', 'N', diag, n - j, &a[j + j * lda],
                             lda, &a[j + (j - 1) * lda], 1);
                pyclap_dscal(n - j, ajj, &a[j + (j - 1) * lda], 1);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* DTRTRI (LAPACK master 2026): inverse of a triangular matrix,        */
/* blocked driver.                                                     */
/*                                                                     */
/* UNREACHABILITY PROOF for the blocked branch: reaching it requires   */
/* 1 < NB < N.  NB = ILAENV(1,'DTRTRI',UPLO//DIAG,...) = 64            */
/* unconditionally (ilaenv.f: C2='TR', C3='TRI' -> NB=64 for both the  */
/* SNAME and CNAME branches; OPTS is never consulted), so the blocked  */
/* branch requires N > 64.  Every caller in this port (pyclap_dgetri,  */
/* reached only from pyclap_invert_pycalphad) has N <= 64.  The body   */
/* would additionally require DTRMM, which this port does not carry,   */
/* so it is guarded with INFO = -999 (a value no reference LAPACK      */
/* routine ever returns) instead of being transliterated.              */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dtrtri(char uplo, char diag, int n,
                                     double* a, int lda, int* info)
{
    const double zero = 0.0;
    int nb;
    int nounit, upper;
    char opts[2];

    *info = 0;
    upper = pyclap_lsame(uplo, 'U');
    nounit = pyclap_lsame(diag, 'N');
    /* (XERBLA argument validation dropped.)  UPPER is only consulted
       inside the guarded blocked branch below (and the dropped
       validation); silence the unused warning. */
    (void)upper;

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Check for singularity if non-unit. */
    if (nounit) {
        /* Fortran: DO 10 INFO = 1, N -- INFO doubles as the loop index
           and is the returned singularity position on early exit. */
        for (*info = 1; *info <= n; (*info)++) {
            if (a[(*info - 1) + (*info - 1) * lda] == zero)
                return;
        }
        *info = 0;
    }

    /* Determine the block size for this environment.
       Fortran passes OPTS = UPLO // DIAG (never consulted for TR/TRI). */
    opts[0] = uplo;
    opts[1] = diag;
    nb = pyclap_ilaenv_lu(1, "DTRTRI", opts, n, -1, -1, -1);
    if (nb <= 1 || nb >= n) {
        /* Use unblocked code */
        pyclap_dtrti2(uplo, diag, n, a, lda, info);
    } else {
        /* Use blocked code: UNREACHABLE (see proof above; requires
           N > 64, and this port guarantees N <= 64).  Would call
           DTRMM/DTRSM/DTRTI2 per block column. */
        *info = -999;
        return;
    }
}

/* ------------------------------------------------------------------ */
/* DGETRI (LAPACK master 2026): inverse of a matrix from its DGETRF    */
/* LU factorization; inverts U then solves inv(A)*L = inv(U).          */
/*                                                                     */
/* Reachability with this port's call pattern (LWORK == N, N <= 64):   */
/*   NB = ILAENV(1,'DGETRI') = 64.  The workspace-reduction branch     */
/*   requires NB < N, i.e. N > 64: never taken, so NB stays 64 and     */
/*   IWS = N.  The blocked solve requires NBMIN <= NB < N: never       */
/*   taken either; the unblocked DGEMV-per-column loop always runs.    */
/*   Both branches are nevertheless fully transliterated (their BLAS   */
/*   dependencies -- DGEMM, DTRSM -- are carried by this port).        */
/* The LWORK = -1 workspace query and XERBLA validation are dropped    */
/* (callers never query; WORK(1) side effects are kept).               */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgetri(int n, double* a, int lda,
                                     const int* ipiv, double* work,
                                     int lwork, int* info)
{
    const double zero = 0.0, one = 1.0;
    int i, iws, j, jb, jj, jp, ldwork, lwkopt, nb, nbmin, nn;

    *info = 0;
    nb = pyclap_ilaenv_lu(1, "DGETRI", " ", n, -1, -1, -1);
    lwkopt = pyclap_imax(1, n * nb);
    work[0] = (double)lwkopt;    /* WORK( 1 ) = LWKOPT */

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Form inv(U).  If INFO > 0 from DTRTRI, then U is singular,
       and the inverse is not computed. */
    pyclap_dtrtri('U', 'N', n, a, lda, info);
    if (*info > 0)
        return;

    nbmin = 2;
    ldwork = n;
    if (nb > 1 && nb < n) {
        iws = pyclap_imax(ldwork * nb, 1);
        if (lwork < iws) {
            nb = lwork / ldwork;
            nbmin = pyclap_imax(2, pyclap_ilaenv_lu(2, "DGETRI", " ", n,
                                                    -1, -1, -1));
        }
    } else {
        iws = n;
    }

    /* Solve the equation inv(A)*L = inv(U) for inv(A). */
    if (nb < nbmin || nb >= n) {
        /* Use unblocked code. */
        for (j = n; j >= 1; j--) {                           /* DO 20 */
            /* Copy current column of L to WORK and replace with zeros. */
            for (i = j + 1; i <= n; i++) {                   /* DO 10 */
                work[i - 1] = a[(i - 1) + (j - 1) * lda];
                a[(i - 1) + (j - 1) * lda] = zero;
            }
            /* Compute current column of inv(A). */
            if (j < n)
                pyclap_dgemv('N', n, n - j, -one, &a[j * lda], lda,
                             &work[j], 1, one, &a[(j - 1) * lda], 1);
        }
    } else {
        /* Use blocked code.  (Unreachable with LWORK == N and N <= 64,
           see the routine comment; kept as a faithful transliteration.) */
        nn = ((n - 1) / nb) * nb + 1;
        for (j = nn; j >= 1; j -= nb) {                      /* DO 50 */
            jb = pyclap_imin(nb, n - j + 1);

            /* Copy current block column of L to WORK and replace with
               zeros. */
            for (jj = j; jj <= j + jb - 1; jj++) {           /* DO 40 */
                for (i = jj + 1; i <= n; i++) {              /* DO 30 */
                    work[(i - 1) + (jj - j) * ldwork] =
                        a[(i - 1) + (jj - 1) * lda];
                    a[(i - 1) + (jj - 1) * lda] = zero;
                }
            }

            /* Compute current block column of inv(A). */
            if (j + jb <= n)
                pyclap_dgemm('N', 'N', n, jb, n - j - jb + 1, -one,
                             &a[(j + jb - 1) * lda], lda,
                             &work[j + jb - 1], ldwork, one,
                             &a[(j - 1) * lda], lda);
            pyclap_dtrsm('R', 'L', 'N', 'U', n, jb, one, &work[j - 1],
                         ldwork, &a[(j - 1) * lda], lda);
        }   /* 50 CONTINUE */
    }

    /* Apply column interchanges. */
    for (j = n - 1; j >= 1; j--) {                           /* DO 60 */
        jp = ipiv[j - 1];
        if (jp != j)
            pyclap_dswap(n, &a[(j - 1) * lda], 1, &a[(jp - 1) * lda], 1);
    }

    work[0] = (double)iws;       /* WORK( 1 ) = IWS */
}

/* ------------------------------------------------------------------ */
/* pycalphad reference wrapper (minimizer.pyx invert_matrix): NaN      */
/* scrub -> all zeros; dgetrf+dgetri with lwork=n; info != 0 -> all    */
/* -1e19.  The caller passes its buffer directly (the reference passes */
/* a C-ordered numpy array straight to Fortran, so this chain sees the */
/* same bytes the reference's LAPACK sees; no transpose).  ipiv and    */
/* work are caller scratch (>= n ints / >= n doubles).                 */
/*                                                                     */
/* Faithful to the Cython source, DGETRI is called UNCONDITIONALLY     */
/* after DGETRF (not gated on info == 0), with the single info         */
/* variable overwritten by the second call -- when DGETRF reports a    */
/* singular U, DGETRI's DTRTRI diagonal scan rediscovers the zero      */
/* pivot and returns info > 0 without touching A, so the final buffer  */
/* is all -1e19 either way.                                            */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_invert_pycalphad(double* a, int n, int* ipiv,
                                               double* work)
{
    int info = 0;
    int i;
    int isfinite = 1;

    for (i = 0; i < n * n; i++) {
        if (pyclap_disnan(a[i]))
            isfinite = 0;
    }

    if (!isfinite) {
        for (i = 0; i < n * n; i++)
            a[i] = 0.0;
    } else {
        pyclap_dgetrf(n, n, a, n, ipiv, &info);
        pyclap_dgetri(n, a, n, ipiv, work, n, &info);
    }

    if (info != 0) {
        for (i = 0; i < n * n; i++)
            a[i] = -1e19;
    }
}

#endif /* PYCLAP_LAPACK_LU_H */
