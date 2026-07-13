/*
 * lapack_mid.h -- faithful C99 transliteration of reference LAPACK/BLAS
 * mid-level routines (netlib master, fetched 2026), for the pycalphad
 * GPU/CPU-shared kernel backend.
 *
 * MUST be #include'd AFTER lapack_leaf.h; calls pyclap_lsame,
 * pyclap_imax/imin, pyclap_dlassq, pyclap_disnan, pyclap_dlarfg,
 * pyclap_dgemv, pyclap_dger, pyclap_daxpy, pyclap_dscal,
 * pyclap_iladlr, pyclap_iladlc from there.
 *
 * Conventions (identical to lapack_leaf.h):
 *   - Column-major layout preserved: A(I,J) -> a[(I-1) + (J-1)*lda].
 *   - Loop variables kept 1-based to mirror the Fortran bounds/direction
 *     exactly; indexing subtracts 1 at the access site.
 *   - Numerical operation ORDER matches the Fortran statement-for-statement
 *     (goal: bitwise-identical results). Only XERBLA/INFO<0 argument
 *     validation and LWORK==-1 workspace-query branches were dropped
 *     (callers in this port never pass LWORK=-1); LWORK parameters are
 *     kept because the blocked drivers' NB reduction logic reads them.
 *   - Every GOTO converted to explicit control flow is documented with a
 *     comment naming the Fortran label.
 *   - Blocked code paths that are PROVABLY unreachable for this port's
 *     matrix sizes (all dims <= 32; see per-routine proofs) are replaced
 *     by loud `*info = -999` guards instead of pulling in DLABRD/DLARFT/
 *     DLARFB/DTRMM. If a guard ever fires, the caller sees INFO=-999,
 *     which no reference LAPACK routine ever returns.
 *   - No dynamic allocation, no printf, C99 only. All functions are
 *     `__device__ static`.
 *
 * Routines: ilaenv (subset), dlanst, dlasrt, dlarf1f, dgemm, dgebd2,
 *           dgebrd, dgeqr2, dgelq2, dgeqrf, dgelqf, dorm2r, dorml2,
 *           dormqr, dormlq, dormbr.
 */
#ifndef PYCLAP_LAPACK_MID_H
#define PYCLAP_LAPACK_MID_H

/* ------------------------------------------------------------------ */
/* ILAENV helpers: 2- and 3-character name-field comparisons           */
/* (Fortran CHARACTER*2 / CHARACTER*3 .EQ. tests).                     */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_ilaenv_eq2(const char* s, const char* t)
{
    return s[0] == t[0] && s[1] == t[1];
}

__device__ static int pyclap_ilaenv_eq3(const char* s, const char* t)
{
    return s[0] == t[0] && s[1] == t[1] && s[2] == t[2];
}

/* The C4 membership test that ilaenv.f repeats for the OR/UN families:
   C4.EQ.'QR' .OR. 'RQ' .OR. 'LQ' .OR. 'QL' .OR. 'HR' .OR. 'TR' .OR. 'BR' */
__device__ static int pyclap_ilaenv_c4gen(const char* c4)
{
    return pyclap_ilaenv_eq2(c4, "QR") || pyclap_ilaenv_eq2(c4, "RQ") ||
           pyclap_ilaenv_eq2(c4, "LQ") || pyclap_ilaenv_eq2(c4, "QL") ||
           pyclap_ilaenv_eq2(c4, "HR") || pyclap_ilaenv_eq2(c4, "TR") ||
           pyclap_ilaenv_eq2(c4, "BR");
}

/* ------------------------------------------------------------------ */
/* ILAENV (LAPACK master 2026): is called from the LAPACK routines to  */
/* choose problem-dependent parameters for the local environment.      */
/*                                                                     */
/* Transliterated SUBSET: exactly the (ISPEC, NAME) queries made by    */
/* this port's callers (DGEBRD, DGEQRF, DGELQF, DORMQR, DORMLQ,        */
/* DORMBR, DGELSD/DLALSD):                                             */
/*   ISPEC=1 (NB, label 50):   GE+{QRF,RQF,LQF,QLF} -> 32,             */
/*                             GE+BRD -> 32, OR+{G,M}{QR,RQ,LQ,QL,HR,  */
/*                             TR,BR} -> 32; default 1.                */
/*   ISPEC=2 (NBMIN, label 60): GE+{QRF,RQF,LQF,QLF} -> 2, GE+BRD -> 2,*/
/*                             OR+{G,M}{...} -> 2; default 2.          */
/*   ISPEC=3 (NX, label 70):   GE+{QRF,RQF,LQF,QLF} -> 128,            */
/*                             GE+BRD -> 128, OR+G{...} -> 128;        */
/*                             default 0 (OR+M gets the default).      */
/*   ISPEC=4 -> 6 (label 80), ISPEC=5 -> 2 (label 90),                 */
/*   ISPEC=6 -> INT(REAL(MIN(N1,N2))*1.6E0) (label 100, MNTHR),        */
/*   ISPEC=7 -> 1 (label 110), ISPEC=8 -> 50 (label 120),              */
/*   ISPEC=9 -> 25 (label 130, SMLSIZ),                                */
/*   ISPEC=10/11 -> 1 (labels 140/150; IEEECK = 1 on IEEE hardware     */
/*   with NaN/infinity support, which this port assumes).              */
/*   Other ISPEC values -> -1 (out-of-range computed GOTO).            */
/* Other GE C3 cases in the reference (TRF, QR , LQ , HRD, TRI, QP3RK) */
/* and the PO/SY/HE/UN/GB/PB/TR/LA/ST/GG families are never queried by */
/* this port and are omitted; their omission cannot change any queried */
/* value because name dispatch is exact-match.                         */
/*                                                                     */
/* NAME is assumed uppercase ASCII (this port passes string literals); */
/* the reference's lowercase/EBCDIC normalization (labels 10-40) and   */
/* the ISPEC=12..17 IPARMQ forwarding (label 160) are dropped.         */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_ilaenv(int ispec, const char* name,
                                    const char* opts, int n1, int n2,
                                    int n3, int n4)
{
    int nb, nbmin, nx, sname, cname;
    char c1;
    const char *c2, *c3, *c4;
    (void)opts;   /* OPTS never participates in the transliterated cases */
    (void)n3;
    (void)n4;

    /* Fortran: GO TO ( 10, 10, 10, 80, 90, 100, 110, 120, 130, 140,
                        150, 160, ... )ISPEC */
    switch (ispec) {
    case 1:
    case 2:
    case 3:
        break;                                 /* label 10 */
    case 4:
        return 6;                              /* label 80 */
    case 5:
        return 2;                              /* label 90 */
    case 6:
        /* label 100: ILAENV = INT( REAL( MIN( N1, N2 ) )*1.6E0 )
           (single-precision product, truncated) */
        return (int)((float)pyclap_imin(n1, n2) * 1.6f);
    case 7:
        return 1;                              /* label 110 */
    case 8:
        return 50;                             /* label 120 */
    case 9:
        return 25;                             /* label 130 */
    case 10:
        return 1;                              /* label 140: IEEECK(1,..) */
    case 11:
        return 1;                              /* label 150: IEEECK(0,..) */
    default:
        return -1;                             /* invalid ISPEC */
    }

    /* label 10: ILAENV = 1; convert NAME to upper case (dropped: input is
       already uppercase ASCII here). */
    c1 = name[0];
    sname = (c1 == 'S' || c1 == 'D');
    cname = (c1 == 'C' || c1 == 'Z');
    if (!(cname || sname))
        return 1;
    c2 = name + 1;      /* SUBNAM(2:3) */
    c3 = name + 3;      /* SUBNAM(4:6) */
    c4 = c3 + 1;        /* C3(2:3)     */
    /* TWOSTAGE only affects SY/HE TRF, never queried here. */

    /* Fortran: GO TO ( 50, 60, 70 )ISPEC */
    if (ispec == 1) {
        /* label 50: ISPEC = 1: block size */
        nb = 1;
        if (pyclap_ilaenv_eq2(c2, "GE")) {
            if (pyclap_ilaenv_eq3(c3, "QRF") ||
                pyclap_ilaenv_eq3(c3, "RQF") ||
                pyclap_ilaenv_eq3(c3, "LQF") ||
                pyclap_ilaenv_eq3(c3, "QLF")) {
                if (sname) {
                    nb = 32;
                } else {
                    nb = 32;
                }
            } else if (pyclap_ilaenv_eq3(c3, "BRD")) {
                if (sname) {
                    nb = 32;
                } else {
                    nb = 32;
                }
            }
            /* other GE C3 cases omitted (never queried by this port) */
        } else if (sname && pyclap_ilaenv_eq2(c2, "OR")) {
            if (c3[0] == 'G') {
                if (pyclap_ilaenv_c4gen(c4)) {
                    nb = 32;
                }
            } else if (c3[0] == 'M') {
                if (pyclap_ilaenv_c4gen(c4)) {
                    nb = 32;
                }
            }
        }
        /* other families omitted (never queried by this port) */
        return nb;
    }
    if (ispec == 2) {
        /* label 60: ISPEC = 2: minimum block size */
        nbmin = 2;
        if (pyclap_ilaenv_eq2(c2, "GE")) {
            if (pyclap_ilaenv_eq3(c3, "QRF") ||
                pyclap_ilaenv_eq3(c3, "RQF") ||
                pyclap_ilaenv_eq3(c3, "LQF") ||
                pyclap_ilaenv_eq3(c3, "QLF")) {
                if (sname) {
                    nbmin = 2;
                } else {
                    nbmin = 2;
                }
            } else if (pyclap_ilaenv_eq3(c3, "BRD")) {
                if (sname) {
                    nbmin = 2;
                } else {
                    nbmin = 2;
                }
            }
        } else if (sname && pyclap_ilaenv_eq2(c2, "OR")) {
            if (c3[0] == 'G') {
                if (pyclap_ilaenv_c4gen(c4)) {
                    nbmin = 2;
                }
            } else if (c3[0] == 'M') {
                if (pyclap_ilaenv_c4gen(c4)) {
                    nbmin = 2;
                }
            }
        }
        return nbmin;
    }
    /* label 70: ISPEC = 3: crossover point */
    nx = 0;
    if (pyclap_ilaenv_eq2(c2, "GE")) {
        if (pyclap_ilaenv_eq3(c3, "QRF") ||
            pyclap_ilaenv_eq3(c3, "RQF") ||
            pyclap_ilaenv_eq3(c3, "LQF") ||
            pyclap_ilaenv_eq3(c3, "QLF")) {
            if (sname) {
                nx = 128;
            } else {
                nx = 128;
            }
        } else if (pyclap_ilaenv_eq3(c3, "BRD")) {
            if (sname) {
                nx = 128;
            } else {
                nx = 128;
            }
        }
    } else if (sname && pyclap_ilaenv_eq2(c2, "OR")) {
        if (c3[0] == 'G') {
            if (pyclap_ilaenv_c4gen(c4)) {
                nx = 128;
            }
        }
        /* OR+M... takes the NX = 0 default, as in the reference */
    }
    return nx;
}

/* ------------------------------------------------------------------ */
/* DLANST (LAPACK master 2026): returns the value of the one norm, or  */
/* the Frobenius norm, or the infinity norm, or the element of largest */
/* absolute value of a real symmetric tridiagonal matrix A.            */
/* No validation in the reference; none dropped.                       */
/* ------------------------------------------------------------------ */
__device__ static double pyclap_dlanst(char norm, int n,
                                       const double* d, const double* e)
{
    const double one = 1.0, zero = 0.0;
    int i;
    double anorm, scale, sum;

    anorm = zero;   /* C-only init: the Fortran leaves ANORM undefined for
                       an unrecognized NORM; every caller passes a valid
                       NORM, so this cannot change results. */
    if (n <= 0) {
        anorm = zero;
    } else if (pyclap_lsame(norm, 'M')) {
        /* Find max(abs(A(i,j))). */
        anorm = fabs(d[n-1]);
        for (i = 1; i <= n - 1; i++) {          /* DO 10 */
            sum = fabs(d[i-1]);
            if (anorm < sum || pyclap_disnan(sum)) anorm = sum;
            sum = fabs(e[i-1]);
            if (anorm < sum || pyclap_disnan(sum)) anorm = sum;
        }
    } else if (pyclap_lsame(norm, 'O') || norm == '1' ||
               pyclap_lsame(norm, 'I')) {
        /* Find norm1(A). */
        if (n == 1) {
            anorm = fabs(d[0]);
        } else {
            anorm = fabs(d[0]) + fabs(e[0]);
            sum = fabs(e[n-2]) + fabs(d[n-1]);
            if (anorm < sum || pyclap_disnan(sum)) anorm = sum;
            for (i = 2; i <= n - 1; i++) {      /* DO 20 */
                sum = fabs(d[i-1]) + fabs(e[i-1]) + fabs(e[i-2]);
                if (anorm < sum || pyclap_disnan(sum)) anorm = sum;
            }
        }
    } else if (pyclap_lsame(norm, 'F') || pyclap_lsame(norm, 'E')) {
        /* Find normF(A). */
        scale = zero;
        sum = one;
        if (n > 1) {
            pyclap_dlassq(n-1, e, 1, &scale, &sum);
            sum = 2*sum;
        }
        pyclap_dlassq(n, d, 1, &scale, &sum);
        anorm = scale*sqrt(sum);
    }
    return anorm;
}

/* ------------------------------------------------------------------ */
/* DLASRT (LAPACK master 2026): sorts the numbers in D in increasing   */
/* order (if ID = 'I') or in decreasing order (if ID = 'D').           */
/* Quick Sort, reverting to Insertion sort on arrays of size <= 20     */
/* (SELECT = 20). STACK(2,32) kept as a flat int[2*32], column-major.  */
/* Dropped: XERBLA call. The DIR validity check is kept (it selects    */
/* the comparison direction); an invalid ID returns INFO = -1 with D   */
/* untouched, matching the reference's early return.                   */
/* GOTO map: 10 = segment-processing loop (bottom test STKPNT>0);      */
/* 20/30 and 40/50 = insertion-sort inner/outer loops (GO TO 30 / 50   */
/* = break out of the inner J loop); 60 and 90 = partition swap loops  */
/* (infinite for(;;), exited when I >= J); 70/100 = J-descend          */
/* do-whiles; 80/110 = I-ascend do-whiles.                             */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlasrt(char id, int n, double* d, int* info)
{
    const int select_ = 20;                     /* SELECT parameter */
    int dir, endd, i, j, start, stkpnt;
    double d1, d2, d3, dmnmx, tmp;
    int stack[2*32];                            /* STACK( 2, 32 ) */

    /* Test the input parameters. */
    *info = 0;
    dir = -1;
    if (pyclap_lsame(id, 'D')) {
        dir = 0;
    } else if (pyclap_lsame(id, 'I')) {
        dir = 1;
    }
    if (dir == -1) {
        *info = -1;
        return;
    }
    /* (N < 0 check dropped with XERBLA) */

    /* Quick return if possible */
    if (n <= 1)
        return;

    stkpnt = 1;
    stack[0 + 0*2] = 1;                         /* STACK( 1, 1 ) = 1 */
    stack[1 + 0*2] = n;                         /* STACK( 2, 1 ) = N */
    do {                                        /* label 10 */
        start = stack[0 + (stkpnt-1)*2];
        endd  = stack[1 + (stkpnt-1)*2];
        stkpnt = stkpnt - 1;
        if (endd - start <= select_ && endd - start > 0) {
            /* Do Insertion sort on D( START:ENDD ) */
            if (dir == 0) {
                /* Sort into decreasing order */
                for (i = start + 1; i <= endd; i++) {       /* DO 30 */
                    for (j = i; j >= start + 1; j--) {      /* DO 20 */
                        if (d[j-1] > d[j-2]) {
                            dmnmx = d[j-1];
                            d[j-1] = d[j-2];
                            d[j-2] = dmnmx;
                        } else {
                            break;              /* GO TO 30 */
                        }
                    }
                }
            } else {
                /* Sort into increasing order */
                for (i = start + 1; i <= endd; i++) {       /* DO 50 */
                    for (j = i; j >= start + 1; j--) {      /* DO 40 */
                        if (d[j-1] < d[j-2]) {
                            dmnmx = d[j-1];
                            d[j-1] = d[j-2];
                            d[j-2] = dmnmx;
                        } else {
                            break;              /* GO TO 50 */
                        }
                    }
                }
            }
        } else if (endd - start > select_) {
            /* Partition D( START:ENDD ) and stack parts, largest one first.
               Choose partition entry as median of 3 */
            d1 = d[start-1];
            d2 = d[endd-1];
            i = (start + endd) / 2;
            d3 = d[i-1];
            if (d1 < d2) {
                if (d3 < d1) {
                    dmnmx = d1;
                } else if (d3 < d2) {
                    dmnmx = d3;
                } else {
                    dmnmx = d2;
                }
            } else {
                if (d3 < d2) {
                    dmnmx = d2;
                } else if (d3 < d1) {
                    dmnmx = d3;
                } else {
                    dmnmx = d1;
                }
            }
            if (dir == 0) {
                /* Sort into decreasing order */
                i = start - 1;
                j = endd + 1;
                for (;;) {                      /* label 60 */
                    do {                        /* label 70 */
                        j = j - 1;
                    } while (d[j-1] < dmnmx);
                    do {                        /* label 80 */
                        i = i + 1;
                    } while (d[i-1] > dmnmx);
                    if (i < j) {
                        tmp = d[i-1];
                        d[i-1] = d[j-1];
                        d[j-1] = tmp;
                        /* GO TO 60 */
                    } else {
                        break;
                    }
                }
                if (j - start > endd - j - 1) {
                    stkpnt = stkpnt + 1;
                    stack[0 + (stkpnt-1)*2] = start;
                    stack[1 + (stkpnt-1)*2] = j;
                    stkpnt = stkpnt + 1;
                    stack[0 + (stkpnt-1)*2] = j + 1;
                    stack[1 + (stkpnt-1)*2] = endd;
                } else {
                    stkpnt = stkpnt + 1;
                    stack[0 + (stkpnt-1)*2] = j + 1;
                    stack[1 + (stkpnt-1)*2] = endd;
                    stkpnt = stkpnt + 1;
                    stack[0 + (stkpnt-1)*2] = start;
                    stack[1 + (stkpnt-1)*2] = j;
                }
            } else {
                /* Sort into increasing order */
                i = start - 1;
                j = endd + 1;
                for (;;) {                      /* label 90 */
                    do {                        /* label 100 */
                        j = j - 1;
                    } while (d[j-1] > dmnmx);
                    do {                        /* label 110 */
                        i = i + 1;
                    } while (d[i-1] < dmnmx);
                    if (i < j) {
                        tmp = d[i-1];
                        d[i-1] = d[j-1];
                        d[j-1] = tmp;
                        /* GO TO 90 */
                    } else {
                        break;
                    }
                }
                if (j - start > endd - j - 1) {
                    stkpnt = stkpnt + 1;
                    stack[0 + (stkpnt-1)*2] = start;
                    stack[1 + (stkpnt-1)*2] = j;
                    stkpnt = stkpnt + 1;
                    stack[0 + (stkpnt-1)*2] = j + 1;
                    stack[1 + (stkpnt-1)*2] = endd;
                } else {
                    stkpnt = stkpnt + 1;
                    stack[0 + (stkpnt-1)*2] = j + 1;
                    stack[1 + (stkpnt-1)*2] = endd;
                    stkpnt = stkpnt + 1;
                    stack[0 + (stkpnt-1)*2] = start;
                    stack[1 + (stkpnt-1)*2] = j;
                }
            }
        }
    } while (stkpnt > 0);                       /* IF( STKPNT.GT.0 ) GO TO 10 */
}

/* ------------------------------------------------------------------ */
/* DLARF1F (LAPACK master 2026): applies an elementary reflector H to  */
/* a real m by n matrix C, from either the left or the right, with     */
/* V(1) assumed to be 1 and not referenced.                            */
/*   H = I - tau * v * v**T                                            */
/* WORK: dimension N (SIDE='L') or M (SIDE='R'). No validation in the  */
/* reference; none dropped. (The classic pyclap_dlarf in lapack_leaf.h */
/* is a different routine: it reads V(1) from memory.)                 */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlarf1f(char side, int m, int n,
                                      const double* v, int incv, double tau,
                                      double* c, int ldc, double* work)
{
    const double one = 1.0, zero = 0.0;
    int applyleft;
    int i, lastv, lastc;

    i = 1;   /* C-only init: referenced only via branches where it has
                been set (tau != 0 guarantees lastc-scan ran) */
    applyleft = pyclap_lsame(side, 'L');
    lastv = 1;
    lastc = 0;
    if (tau != zero) {
        /* Set up variables for scanning V.  LASTV begins pointing to the
           end of V. */
        if (applyleft) {
            lastv = m;
        } else {
            lastv = n;
        }
        if (incv > 0) {
            i = 1 + (lastv - 1) * incv;
        } else {
            i = 1;
        }
        /* Look for the last non-zero row in V.  Since we are assuming that
           V(1) = 1, and it is not stored, we shouldn't access it. */
        while (lastv > 1 && v[i-1] == zero) {
            lastv = lastv - 1;
            i = i - incv;
        }
        if (applyleft) {
            /* Scan for the last non-zero column in C(1:lastv,:). */
            lastc = pyclap_iladlc(lastv, n, c, ldc);
        } else {
            /* Scan for the last non-zero row in C(:,1:lastv). */
            lastc = pyclap_iladlr(m, lastv, c, ldc);
        }
        /* Set index for V. If INCV < 0, then I points to the end of V.
           For INCV > 0, set I to point to V(2) */
        if (incv > 0) {
            i = 1 + incv;
        }
    }
    if (lastc == 0) {
        return;
    }
    if (applyleft) {
        /* Form  H * C */
        /* Check if lastv = 1. This means v = 1, so we just need to compute
           C := HC = (1-tau)C. */
        if (lastv == 1) {
            /* C(1,1:lastc) := ( 1 - tau ) * C(1,1:lastc) */
            pyclap_dscal(lastc, one - tau, c, ldc);
        } else {
            /* w(1:lastc,1) := C(2:lastv,1:lastc)**T * v(2:lastv,1) */
            pyclap_dgemv('T', lastv-1, lastc, one, &c[1 + 0*ldc], ldc,
                         &v[i-1], incv, zero, work, 1);
            /* w(1:lastc,1) += C(1,1:lastc)**T * v(1,1) = C(1,1:lastc)**T */
            pyclap_daxpy(lastc, one, c, ldc, work, 1);
            /* C(1, 1:lastc) := C(...) - tau * v(1,1) * w(1:lastc,1)**T */
            pyclap_daxpy(lastc, -tau, work, 1, c, ldc);
            /* C(2:lastv,1:lastc) := C(...) - tau*v(2:lastv,1)*w(1:lastc,1)**T */
            pyclap_dger(lastv-1, lastc, -tau, &v[i-1], incv, work, 1,
                        &c[1 + 0*ldc], ldc);
        }
    } else {
        /* Form  C * H */
        /* Check if lastv = 1. This means v = 1, so we just need to compute
           C := CH = C(1-tau). */
        if (lastv == 1) {
            /* C(1:lastc,1) := ( 1 - tau ) * C(1:lastc,1) */
            pyclap_dscal(lastc, one - tau, c, 1);
        } else {
            /* w(1:lastc,1) := C(1:lastc,2:lastv) * v(2:lastv,1) */
            pyclap_dgemv('N', lastc, lastv-1, one, &c[0 + 1*ldc], ldc,
                         &v[i-1], incv, zero, work, 1);
            /* w(1:lastc,1) += C(1:lastc,1) v(1,1) = C(1:lastc,1) */
            pyclap_daxpy(lastc, one, c, 1, work, 1);
            /* C(1:lastc,1) := C(...) - tau * w(1:lastc,1) * v(1,1)**T */
            pyclap_daxpy(lastc, -tau, work, 1, c, 1);
            /* C(1:lastc,2:lastv) := C(...) - tau*w(1:lastc,1)*v(2:lastv)**T */
            pyclap_dger(lastc, lastv-1, -tau, work, 1, &v[i-1], incv,
                        &c[0 + 1*ldc], ldc);
        }
    }
}

/* ------------------------------------------------------------------ */
/* DGEMM (reference BLAS, netlib master 2026): performs one of the     */
/* matrix-matrix operations C := alpha*op( A )*op( B ) + beta*C,       */
/* where op( X ) is one of op( X ) = X or op( X ) = X**T.              */
/* Dropped: INFO parameter checks + XERBLA. 'C' is accepted for either */
/* TRANS argument exactly as in the reference (treated as 'T').        */
/* Loop labels 10..200 are the reference's nested DO loops, preserved  */
/* with identical nesting and accumulation order.                      */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgemm(char transa, char transb, int m, int n,
                                    int k, double alpha, const double* a,
                                    int lda, const double* b, int ldb,
                                    double beta, double* c, int ldc)
{
    const double one = 1.0, zero = 0.0;
    double temp;
    int i, j, l;
    int nota, notb;

    nota = pyclap_lsame(transa, 'N');
    notb = pyclap_lsame(transb, 'N');
    /* NROWA/NROWB only feed the dropped validation. */

    /* Quick return if possible. */
    if ((m == 0) || (n == 0) ||
        (((alpha == zero) || (k == 0)) && (beta == one)))
        return;

    /* And if alpha.eq.zero. */
    if (alpha == zero) {
        if (beta == zero) {
            for (j = 1; j <= n; j++) {          /* DO 20 */
                for (i = 1; i <= m; i++) {      /* DO 10 */
                    c[(i-1) + (j-1)*ldc] = zero;
                }
            }
        } else {
            for (j = 1; j <= n; j++) {          /* DO 40 */
                for (i = 1; i <= m; i++) {      /* DO 30 */
                    c[(i-1) + (j-1)*ldc] = beta*c[(i-1) + (j-1)*ldc];
                }
            }
        }
        return;
    }

    /* Start the operations. */
    if (notb) {
        if (nota) {
            /* Form  C := alpha*A*B + beta*C. */
            for (j = 1; j <= n; j++) {          /* DO 90 */
                if (beta == zero) {
                    for (i = 1; i <= m; i++) {  /* DO 50 */
                        c[(i-1) + (j-1)*ldc] = zero;
                    }
                } else if (beta != one) {
                    for (i = 1; i <= m; i++) {  /* DO 60 */
                        c[(i-1) + (j-1)*ldc] = beta*c[(i-1) + (j-1)*ldc];
                    }
                }
                for (l = 1; l <= k; l++) {      /* DO 80 */
                    temp = alpha*b[(l-1) + (j-1)*ldb];
                    for (i = 1; i <= m; i++) {  /* DO 70 */
                        c[(i-1) + (j-1)*ldc] = c[(i-1) + (j-1)*ldc] +
                                               temp*a[(i-1) + (l-1)*lda];
                    }
                }
            }
        } else {
            /* Form  C := alpha*A**T*B + beta*C */
            for (j = 1; j <= n; j++) {          /* DO 120 */
                for (i = 1; i <= m; i++) {      /* DO 110 */
                    temp = zero;
                    for (l = 1; l <= k; l++) {  /* DO 100 */
                        temp = temp + a[(l-1) + (i-1)*lda]*
                                      b[(l-1) + (j-1)*ldb];
                    }
                    if (beta == zero) {
                        c[(i-1) + (j-1)*ldc] = alpha*temp;
                    } else {
                        c[(i-1) + (j-1)*ldc] = alpha*temp +
                                               beta*c[(i-1) + (j-1)*ldc];
                    }
                }
            }
        }
    } else {
        if (nota) {
            /* Form  C := alpha*A*B**T + beta*C */
            for (j = 1; j <= n; j++) {          /* DO 170 */
                if (beta == zero) {
                    for (i = 1; i <= m; i++) {  /* DO 130 */
                        c[(i-1) + (j-1)*ldc] = zero;
                    }
                } else if (beta != one) {
                    for (i = 1; i <= m; i++) {  /* DO 140 */
                        c[(i-1) + (j-1)*ldc] = beta*c[(i-1) + (j-1)*ldc];
                    }
                }
                for (l = 1; l <= k; l++) {      /* DO 160 */
                    temp = alpha*b[(j-1) + (l-1)*ldb];
                    for (i = 1; i <= m; i++) {  /* DO 150 */
                        c[(i-1) + (j-1)*ldc] = c[(i-1) + (j-1)*ldc] +
                                               temp*a[(i-1) + (l-1)*lda];
                    }
                }
            }
        } else {
            /* Form  C := alpha*A**T*B**T + beta*C */
            for (j = 1; j <= n; j++) {          /* DO 200 */
                for (i = 1; i <= m; i++) {      /* DO 190 */
                    temp = zero;
                    for (l = 1; l <= k; l++) {  /* DO 180 */
                        temp = temp + a[(l-1) + (i-1)*lda]*
                                      b[(j-1) + (l-1)*ldb];
                    }
                    if (beta == zero) {
                        c[(i-1) + (j-1)*ldc] = alpha*temp;
                    } else {
                        c[(i-1) + (j-1)*ldc] = alpha*temp +
                                               beta*c[(i-1) + (j-1)*ldc];
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* DGEBD2 (LAPACK master 2026): reduces a real general m by n matrix A */
/* to upper or lower bidiagonal form B by an orthogonal transformation */
/* Q**T * A * P = B. WORK: dimension (max(M,N)).                       */
/* Dropped: INFO<0 parameter checks + XERBLA (INFO is always set 0).   */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgebd2(int m, int n, double* a, int lda,
                                     double* d, double* e, double* tauq,
                                     double* taup, double* work, int* info)
{
    const double zero = 0.0;
    int i;

    *info = 0;

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i = 1; i <= n; i++) {              /* DO 10 */
            /* Generate elementary reflector H(i) to annihilate A(i+1:m,i) */
            pyclap_dlarfg(m-i+1, &a[(i-1) + (i-1)*lda],
                          &a[(pyclap_imin(i+1, m)-1) + (i-1)*lda], 1,
                          &tauq[i-1]);
            d[i-1] = a[(i-1) + (i-1)*lda];
            /* Apply H(i) to A(i:m,i+1:n) from the left */
            if (i < n)
                pyclap_dlarf1f('L', m-i+1, n-i, &a[(i-1) + (i-1)*lda], 1,
                               tauq[i-1], &a[(i-1) + i*lda], lda, work);
            if (i < n) {
                /* Generate elementary reflector G(i) to annihilate
                   A(i,i+2:n) */
                pyclap_dlarfg(n-i, &a[(i-1) + i*lda],
                              &a[(i-1) + (pyclap_imin(i+2, n)-1)*lda],
                              lda, &taup[i-1]);
                e[i-1] = a[(i-1) + i*lda];
                /* Apply G(i) to A(i+1:m,i+1:n) from the right */
                pyclap_dlarf1f('R', m-i, n-i, &a[(i-1) + i*lda], lda,
                               taup[i-1], &a[i + i*lda], lda, work);
            } else {
                taup[i-1] = zero;
            }
        }
    } else {
        /* Reduce to lower bidiagonal form */
        for (i = 1; i <= m; i++) {              /* DO 20 */
            /* Generate elementary reflector G(i) to annihilate A(i,i+1:n) */
            pyclap_dlarfg(n-i+1, &a[(i-1) + (i-1)*lda],
                          &a[(i-1) + (pyclap_imin(i+1, n)-1)*lda], lda,
                          &taup[i-1]);
            d[i-1] = a[(i-1) + (i-1)*lda];
            /* Apply G(i) to A(i+1:m,i:n) from the right */
            if (i < m)
                pyclap_dlarf1f('R', m-i, n-i+1, &a[(i-1) + (i-1)*lda], lda,
                               taup[i-1], &a[i + (i-1)*lda], lda, work);
            if (i < m) {
                /* Generate elementary reflector H(i) to annihilate
                   A(i+2:m,i) */
                pyclap_dlarfg(m-i, &a[i + (i-1)*lda],
                              &a[(pyclap_imin(i+2, m)-1) + (i-1)*lda], 1,
                              &tauq[i-1]);
                e[i-1] = a[i + (i-1)*lda];
                /* Apply H(i) to A(i+1:m,i+1:n) from the left */
                pyclap_dlarf1f('L', m-i, n-i, &a[i + (i-1)*lda], 1,
                               tauq[i-1], &a[i + i*lda], lda, work);
            } else {
                tauq[i-1] = zero;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* DGEBRD (LAPACK master 2026): reduces a general real M-by-N matrix A */
/* to upper or lower bidiagonal form B by an orthogonal transformation */
/* Q**T * A * P = B. Blocked driver over DLABRD/DGEMM with DGEBD2 tail.*/
/* Dropped: INFO<0 checks (incl. the LWORK<LWKMIN check), XERBLA, and  */
/* the LQUERY branch (callers never pass LWORK = -1).                  */
/*                                                                     */
/* BLOCKED PATH GUARDED (-999) -- unreachability proof from the ilaenv */
/* constants (this port's dimensions satisfy M,N <= 32 by compile-time */
/* array sizing; the proof below holds for all M,N <= 128):            */
/*   The DO 30 loop runs iff MINMN - NX >= 1, i.e. NX < MINMN.         */
/*   - If NB <= 1 or NB >= MINMN, the ELSE branch sets NX = MINMN:     */
/*     zero trips.                                                     */
/*   - Otherwise NB = ILAENV(1,'DGEBRD') = 32 (so MINMN > 32 already   */
/*     required) and NX = MAX(NB, ILAENV(3,'DGEBRD')) = MAX(32,128)    */
/*     = 128. NX is never reduced afterwards (the LWORK branch can     */
/*     only set NX = MINMN, again giving zero trips). Hence trips      */
/*     require MINMN > 128 -- impossible for M,N <= 32.                */
/* Consequently DLABRD and the two trailing-update DGEMM calls are     */
/* never needed and are not transliterated. The loop leaves I = 1      */
/* after zero trips, exactly like the Fortran DO, so the DGEBD2 tail   */
/* call operates on the full matrix.                                   */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgebrd(int m, int n, double* a, int lda,
                                     double* d, double* e, double* tauq,
                                     double* taup, double* work, int lwork,
                                     int* info)
{
    int i, iinfo, ldwrkx, ldwrky, lwkopt, minmn, nb, nbmin, nx, ws;

    /* Test the input parameters */
    *info = 0;
    minmn = pyclap_imin(m, n);
    nb = 1;   /* C-only init: the Fortran leaves NB undefined when
                 MINMN == 0, but then returns before using it */
    if (minmn == 0) {
        lwkopt = 1;
    } else {
        nb = pyclap_imax(1, pyclap_ilaenv(1, "DGEBRD", " ", m, n, -1, -1));
        lwkopt = (m + n)*nb;
    }
    work[0] = (double)lwkopt;
    /* (parameter checks, XERBLA, and LQUERY early return dropped) */

    /* Quick return if possible */
    if (minmn == 0) {
        work[0] = 1;
        return;
    }

    ws = pyclap_imax(m, n);
    ldwrkx = m;
    ldwrky = n;
    (void)ldwrkx;   /* referenced only by the guarded blocked path */
    (void)ldwrky;

    if (nb > 1 && nb < minmn) {
        /* Set the crossover point NX. */
        nx = pyclap_imax(nb, pyclap_ilaenv(3, "DGEBRD", " ", m, n, -1, -1));
        /* Determine when to switch from blocked to unblocked code. */
        if (nx < minmn) {
            ws = lwkopt;
            if (lwork < ws) {
                /* Not enough work space for the optimal NB, consider using
                   a smaller block size. */
                nbmin = pyclap_ilaenv(2, "DGEBRD", " ", m, n, -1, -1);
                if (lwork >= (m + n)*nbmin) {
                    nb = lwork / (m + n);
                } else {
                    nb = 1;
                    nx = minmn;
                }
            }
        }
    } else {
        nx = minmn;
    }

    for (i = 1; i <= minmn - nx; i += nb) {     /* DO 30 (guarded) */
        /* UNREACHABLE for this port (requires MINMN > 128; see proof in
           the routine header). The reference body reduces rows/columns
           i:i+nb-1 via DLABRD and updates the trailing submatrix with two
           DGEMMs. If this guard ever fires, sizes exceeded the port's
           design envelope. */
        *info = -999;
        return;
    }

    /* Use unblocked code to reduce the remainder of the matrix.
       After a zero-trip DO loop the Fortran leaves I = 1; the C for-loop
       above does the same. */
    pyclap_dgebd2(m-i+1, n-i+1, &a[(i-1) + (i-1)*lda], lda, &d[i-1],
                  &e[i-1], &tauq[i-1], &taup[i-1], work, &iinfo);
    work[0] = (double)ws;
}

/* ------------------------------------------------------------------ */
/* DGEQR2 (LAPACK master 2026): computes a QR factorization of a real  */
/* m by n matrix A: A = Q * R (unblocked). WORK: dimension (N).        */
/* Dropped: INFO<0 parameter checks + XERBLA (INFO is always set 0).   */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgeqr2(int m, int n, double* a, int lda,
                                     double* tau, double* work, int* info)
{
    int i, k;

    *info = 0;

    k = pyclap_imin(m, n);

    for (i = 1; i <= k; i++) {                  /* DO 10 */
        /* Generate elementary reflector H(i) to annihilate A(i+1:m,i) */
        pyclap_dlarfg(m-i+1, &a[(i-1) + (i-1)*lda],
                      &a[(pyclap_imin(i+1, m)-1) + (i-1)*lda], 1,
                      &tau[i-1]);
        if (i < n) {
            /* Apply H(i) to A(i:m,i+1:n) from the left */
            pyclap_dlarf1f('L', m-i+1, n-i, &a[(i-1) + (i-1)*lda], 1,
                           tau[i-1], &a[(i-1) + i*lda], lda, work);
        }
    }
}

/* ------------------------------------------------------------------ */
/* DGELQ2 (LAPACK master 2026): computes an LQ factorization of a real */
/* m by n matrix A: A = L * Q (unblocked). WORK: dimension (M).        */
/* Dropped: INFO<0 parameter checks + XERBLA (INFO is always set 0).   */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgelq2(int m, int n, double* a, int lda,
                                     double* tau, double* work, int* info)
{
    int i, k;

    *info = 0;

    k = pyclap_imin(m, n);

    for (i = 1; i <= k; i++) {                  /* DO 10 */
        /* Generate elementary reflector H(i) to annihilate A(i,i+1:n) */
        pyclap_dlarfg(n-i+1, &a[(i-1) + (i-1)*lda],
                      &a[(i-1) + (pyclap_imin(i+1, n)-1)*lda], lda,
                      &tau[i-1]);
        if (i < m) {
            /* Apply H(i) to A(i+1:m,i:n) from the right */
            pyclap_dlarf1f('R', m-i, n-i+1, &a[(i-1) + (i-1)*lda], lda,
                           tau[i-1], &a[i + (i-1)*lda], lda, work);
        }
    }
}

/* ------------------------------------------------------------------ */
/* DGEQRF (LAPACK master 2026): computes a QR factorization of a real  */
/* M-by-N matrix A = Q * R (blocked driver over DGEQR2/DLARFT/DLARFB). */
/* Dropped: INFO<0 checks (incl. LWORK check), XERBLA, and the LQUERY  */
/* branch with its LWKOPT computation (callers never pass LWORK = -1). */
/*                                                                     */
/* BLOCKED PATH GUARDED (-999) -- unreachability proof:                */
/*   The blocked branch requires NB >= NBMIN .AND. NB < K .AND. NX < K.*/
/*   NB = ILAENV(1,'DGEQRF') = 32; NB is only ever reduced inside the  */
/*   IF( NB.GT.1 .AND. NB.LT.K ) block, so NB < K requires K > 32 --   */
/*   impossible for this port (K = MIN(M,N) <= 32). (For 32 < K <= 128 */
/*   the reference would also take the unblocked tail whenever         */
/*   NX = 128 >= K.) Hence DLARFT/DLARFB are never needed. The ELSE    */
/*   branch sets I = 1 exactly as the Fortran does.                    */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgeqrf(int m, int n, double* a, int lda,
                                     double* tau, double* work, int lwork,
                                     int* info)
{
    int i, iinfo, iws, k, ldwork, nb, nbmin, nx;

    /* Test the input arguments */
    k = pyclap_imin(m, n);
    *info = 0;
    nb = pyclap_ilaenv(1, "DGEQRF", " ", m, n, -1, -1);
    /* (parameter checks, XERBLA, and LQUERY early return dropped) */

    /* Quick return if possible */
    if (k == 0) {
        work[0] = 1;
        return;
    }

    nbmin = 2;
    nx = 0;
    iws = n;
    if (nb > 1 && nb < k) {
        /* Determine when to cross over from blocked to unblocked code. */
        nx = pyclap_imax(0, pyclap_ilaenv(3, "DGEQRF", " ", m, n, -1, -1));
        if (nx < k) {
            /* Determine if workspace is large enough for blocked code. */
            ldwork = n;
            iws = ldwork*nb;
            if (lwork < iws) {
                /* Not enough workspace to use optimal NB:  reduce NB and
                   determine the minimum value of NB. */
                nb = lwork / ldwork;
                nbmin = pyclap_imax(2, pyclap_ilaenv(2, "DGEQRF", " ",
                                                     m, n, -1, -1));
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* UNREACHABLE for this port (requires K > 32; see proof in the
           routine header). The reference body (DO 10 loop) factors
           blocks with DGEQR2 and applies H**T via DLARFT/DLARFB. */
        *info = -999;
        return;
    } else {
        i = 1;
    }

    /* Use unblocked code to factor the last or only block. */
    if (i <= k)
        pyclap_dgeqr2(m-i+1, n-i+1, &a[(i-1) + (i-1)*lda], lda, &tau[i-1],
                      work, &iinfo);

    work[0] = (double)iws;
}

/* ------------------------------------------------------------------ */
/* DGELQF (LAPACK master 2026): computes an LQ factorization of a real */
/* M-by-N matrix A = L * Q (blocked driver over DGELQ2/DLARFT/DLARFB). */
/* Dropped: INFO<0 checks (incl. LWORK check), XERBLA, and the LQUERY  */
/* branch with its LWKOPT computation (callers never pass LWORK = -1). */
/*                                                                     */
/* BLOCKED PATH GUARDED (-999): identical argument to DGEQRF above     */
/* with NB = ILAENV(1,'DGELQF') = 32 and NX = 128 -- the blocked       */
/* branch requires K = MIN(M,N) > 32, impossible for this port.        */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgelqf(int m, int n, double* a, int lda,
                                     double* tau, double* work, int lwork,
                                     int* info)
{
    int i, iinfo, iws, k, ldwork, nb, nbmin, nx;

    /* Test the input arguments */
    *info = 0;
    k = pyclap_imin(m, n);
    nb = pyclap_ilaenv(1, "DGELQF", " ", m, n, -1, -1);
    /* (parameter checks, XERBLA, and LQUERY early return dropped) */

    /* Quick return if possible */
    if (k == 0) {
        work[0] = 1;
        return;
    }

    nbmin = 2;
    nx = 0;
    iws = m;
    if (nb > 1 && nb < k) {
        /* Determine when to cross over from blocked to unblocked code. */
        nx = pyclap_imax(0, pyclap_ilaenv(3, "DGELQF", " ", m, n, -1, -1));
        if (nx < k) {
            /* Determine if workspace is large enough for blocked code. */
            ldwork = m;
            iws = ldwork*nb;
            if (lwork < iws) {
                /* Not enough workspace to use optimal NB:  reduce NB and
                   determine the minimum value of NB. */
                nb = lwork / ldwork;
                nbmin = pyclap_imax(2, pyclap_ilaenv(2, "DGELQF", " ",
                                                     m, n, -1, -1));
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* UNREACHABLE for this port (requires K > 32; see proof in the
           DGEQRF header). The reference body (DO 10 loop) factors blocks
           with DGELQ2 and applies H via DLARFT/DLARFB. */
        *info = -999;
        return;
    } else {
        i = 1;
    }

    /* Use unblocked code to factor the last or only block. */
    if (i <= k)
        pyclap_dgelq2(m-i+1, n-i+1, &a[(i-1) + (i-1)*lda], lda, &tau[i-1],
                      work, &iinfo);

    work[0] = (double)iws;
}

/* ------------------------------------------------------------------ */
/* DORM2R (LAPACK master 2026): overwrites the general real m by n     */
/* matrix C with Q*C, Q**T*C, C*Q or C*Q**T (unblocked), where Q is    */
/* defined as the product of k elementary reflectors from DGEQRF.      */
/* WORK: dimension (N) if SIDE='L', (M) if SIDE='R'.                   */
/* Dropped: INFO<0 parameter checks + XERBLA (INFO is always set 0;    */
/* the NQ local fed only the dropped LDA check and is omitted).        */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dorm2r(char side, char trans, int m, int n,
                                     int k, const double* a, int lda,
                                     const double* tau, double* c, int ldc,
                                     double* work, int* info)
{
    int left, notran;
    int i, i1, i2, i3, ic, jc, mi, ni;

    *info = 0;
    left = pyclap_lsame(side, 'L');
    notran = pyclap_lsame(trans, 'N');
    /* (parameter checks and XERBLA dropped) */

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0)
        return;

    if ((left && !notran) || (!left && notran)) {
        i1 = 1;
        i2 = k;
        i3 = 1;
    } else {
        i1 = k;
        i2 = 1;
        i3 = -1;
    }

    mi = 0;   /* C-only inits: the Fortran leaves the non-selected pair */
    ni = 0;   /* undefined until the loop body assigns it */
    ic = 0;
    jc = 0;
    if (left) {
        ni = n;
        jc = 1;
    } else {
        mi = m;
        ic = 1;
    }

    for (i = i1; (i3 > 0) ? (i <= i2) : (i >= i2); i += i3) {  /* DO 10 */
        if (left) {
            /* H(i) is applied to C(i:m,1:n) */
            mi = m - i + 1;
            ic = i;
        } else {
            /* H(i) is applied to C(1:m,i:n) */
            ni = n - i + 1;
            jc = i;
        }
        /* Apply H(i) */
        pyclap_dlarf1f(side, mi, ni, &a[(i-1) + (i-1)*lda], 1, tau[i-1],
                       &c[(ic-1) + (jc-1)*ldc], ldc, work);
    }
}

/* ------------------------------------------------------------------ */
/* DORML2 (LAPACK master 2026): overwrites the general real m by n     */
/* matrix C with Q*C, Q**T*C, C*Q or C*Q**T (unblocked), where Q is    */
/* defined as the product of k elementary reflectors from DGELQF.      */
/* WORK: dimension (N) if SIDE='L', (M) if SIDE='R'.                   */
/* Dropped: INFO<0 parameter checks + XERBLA (INFO is always set 0;    */
/* the NQ local fed only the dropped K range check and is omitted).    */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dorml2(char side, char trans, int m, int n,
                                     int k, const double* a, int lda,
                                     const double* tau, double* c, int ldc,
                                     double* work, int* info)
{
    int left, notran;
    int i, i1, i2, i3, ic, jc, mi, ni;

    *info = 0;
    left = pyclap_lsame(side, 'L');
    notran = pyclap_lsame(trans, 'N');
    /* (parameter checks and XERBLA dropped) */

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0)
        return;

    if ((left && notran) || (!left && !notran)) {
        i1 = 1;
        i2 = k;
        i3 = 1;
    } else {
        i1 = k;
        i2 = 1;
        i3 = -1;
    }

    mi = 0;   /* C-only inits (see DORM2R) */
    ni = 0;
    ic = 0;
    jc = 0;
    if (left) {
        ni = n;
        jc = 1;
    } else {
        mi = m;
        ic = 1;
    }

    for (i = i1; (i3 > 0) ? (i <= i2) : (i >= i2); i += i3) {  /* DO 10 */
        if (left) {
            /* H(i) is applied to C(i:m,1:n) */
            mi = m - i + 1;
            ic = i;
        } else {
            /* H(i) is applied to C(1:m,i:n) */
            ni = n - i + 1;
            jc = i;
        }
        /* Apply H(i) */
        pyclap_dlarf1f(side, mi, ni, &a[(i-1) + (i-1)*lda], lda, tau[i-1],
                       &c[(ic-1) + (jc-1)*ldc], ldc, work);
    }
}

/* ------------------------------------------------------------------ */
/* DORMQR (LAPACK master 2026): overwrites the general real M-by-N     */
/* matrix C with Q*C, Q**T*C, C*Q or C*Q**T, where Q is defined as the */
/* product of k elementary reflectors from DGEQRF (blocked driver).    */
/* Dropped: INFO<0 checks (incl. LWORK check), XERBLA, LQUERY branch.  */
/* The TSIZE parameter of the reference is unused there and omitted.   */
/*                                                                     */
/* BLOCKED PATH GUARDED (-999) -- unreachability proof:                */
/*   NB = MIN(NBMAX=64, ILAENV(1,'DORMQR')=32) = 32. NB is only        */
/*   reduced inside IF( NB.GT.1 .AND. NB.LT.K ), so the blocked branch */
/*   condition .NOT.( NB.LT.NBMIN .OR. NB.GE.K ) requires NB < K,      */
/*   i.e. K > 32 -- impossible for this port (K <= 32). Hence          */
/*   DLARFT/DLARFB are never needed.                                   */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dormqr(char side, char trans, int m, int n,
                                     int k, const double* a, int lda,
                                     const double* tau, double* c, int ldc,
                                     double* work, int lwork, int* info)
{
    const int nbmax = 64;
    const int ldt = nbmax + 1;                  /* LDT = NBMAX+1 */
    int left, notran;
    int iinfo, ldwork, lwkopt, nb, nbmin, nw;
    char opts2[3];

    *info = 0;
    left = pyclap_lsame(side, 'L');
    notran = pyclap_lsame(trans, 'N');
    (void)notran;   /* feeds only the guarded blocked path (I1/I2/I3) and
                       the dropped validation */

    /* NQ is the order of Q and NW is the minimum dimension of WORK
       (NQ fed only the dropped LDA check and is omitted) */
    if (left) {
        nw = pyclap_imax(1, n);
    } else {
        nw = pyclap_imax(1, m);
    }
    /* (parameter checks and XERBLA dropped) */

    /* Determine the block size (Fortran: OPTS = SIDE // TRANS) */
    opts2[0] = side;
    opts2[1] = trans;
    opts2[2] = '\0';
    nb = pyclap_imin(nbmax, pyclap_ilaenv(1, "DORMQR", opts2, m, n, k, -1));
    lwkopt = nw*nb + ldt*nb;
    work[0] = (double)lwkopt;
    /* (LQUERY early return dropped) */

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        work[0] = 1;
        return;
    }

    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < k) {
        if (lwork < lwkopt) {
            nb = lwork / (ldwork + ldt);
            nbmin = pyclap_imax(2, pyclap_ilaenv(2, "DORMQR", opts2,
                                                 m, n, k, -1));
        }
    }

    if (nb < nbmin || nb >= k) {
        /* Use unblocked code */
        pyclap_dorm2r(side, trans, m, n, k, a, lda, tau, c, ldc, work,
                      &iinfo);
    } else {
        /* UNREACHABLE for this port (requires K > 32; see proof in the
           routine header). The reference body (DO 10 loop) forms block
           reflectors with DLARFT and applies them with DLARFB. */
        *info = -999;
        return;
    }
    work[0] = (double)lwkopt;
}

/* ------------------------------------------------------------------ */
/* DORMLQ (LAPACK master 2026): overwrites the general real M-by-N     */
/* matrix C with Q*C, Q**T*C, C*Q or C*Q**T, where Q is defined as the */
/* product of k elementary reflectors from DGELQF (blocked driver).    */
/* Dropped: INFO<0 checks (incl. LWORK check), XERBLA, LQUERY branch.  */
/* The TSIZE parameter of the reference is unused there and omitted.   */
/* The TRANST local of the reference is used only by the guarded       */
/* blocked path (DLARFB call) and is omitted.                          */
/*                                                                     */
/* BLOCKED PATH GUARDED (-999): identical argument to DORMQR above     */
/* with NB = MIN(64, ILAENV(1,'DORMLQ')=32) = 32; the blocked branch   */
/* requires K > 32, impossible for this port.                          */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dormlq(char side, char trans, int m, int n,
                                     int k, const double* a, int lda,
                                     const double* tau, double* c, int ldc,
                                     double* work, int lwork, int* info)
{
    const int nbmax = 64;
    const int ldt = nbmax + 1;                  /* LDT = NBMAX+1 */
    int left, notran;
    int iinfo, ldwork, lwkopt, nb, nbmin, nw;
    char opts2[3];

    *info = 0;
    left = pyclap_lsame(side, 'L');
    notran = pyclap_lsame(trans, 'N');
    (void)notran;   /* feeds only the guarded blocked path (TRANST) and
                       the dropped validation */

    /* NQ is the order of Q and NW is the minimum dimension of WORK
       (NQ fed only the guarded DLARFT call and dropped checks) */
    if (left) {
        nw = pyclap_imax(1, n);
    } else {
        nw = pyclap_imax(1, m);
    }
    /* (parameter checks and XERBLA dropped) */

    /* Determine the block size (Fortran: OPTS = SIDE // TRANS) */
    opts2[0] = side;
    opts2[1] = trans;
    opts2[2] = '\0';
    nb = pyclap_imin(nbmax, pyclap_ilaenv(1, "DORMLQ", opts2, m, n, k, -1));
    lwkopt = nw*nb + ldt*nb;
    work[0] = (double)lwkopt;
    /* (LQUERY early return dropped) */

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        work[0] = 1;
        return;
    }

    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < k) {
        if (lwork < lwkopt) {
            nb = lwork / (ldwork + ldt);
            nbmin = pyclap_imax(2, pyclap_ilaenv(2, "DORMLQ", opts2,
                                                 m, n, k, -1));
        }
    }

    if (nb < nbmin || nb >= k) {
        /* Use unblocked code */
        pyclap_dorml2(side, trans, m, n, k, a, lda, tau, c, ldc, work,
                      &iinfo);
    } else {
        /* UNREACHABLE for this port (requires K > 32; see proof in the
           DORMQR header). The reference body (DO 10 loop) forms block
           reflectors with DLARFT and applies them with DLARFB. */
        *info = -999;
        return;
    }
    work[0] = (double)lwkopt;
}

/* ------------------------------------------------------------------ */
/* DORMBR (LAPACK master 2026): If VECT = 'Q', overwrites the general  */
/* real M-by-N matrix C with Q*C, Q**T*C, C*Q or C*Q**T; if VECT='P',  */
/* with P*C, P**T*C, C*P or C*P**T, where Q and P**T are the           */
/* orthogonal matrices determined by DGEBRD. Complete transliteration  */
/* (routing to DORMQR/DORMLQ, including the NQ<K index offsets).       */
/* Dropped: INFO<0 checks (incl. LWORK check), XERBLA, LQUERY branch.  */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dormbr(char vect, char side, char trans,
                                     int m, int n, int k, const double* a,
                                     int lda, const double* tau, double* c,
                                     int ldc, double* work, int lwork,
                                     int* info)
{
    int applyq, left, notran;
    char transt;
    int i1, i2, iinfo, lwkopt, mi, ni, nb, nq, nw;
    char opts2[3];

    *info = 0;
    applyq = pyclap_lsame(vect, 'Q');
    left = pyclap_lsame(side, 'L');
    notran = pyclap_lsame(trans, 'N');

    /* NQ is the order of Q or P and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = pyclap_imax(1, n);
    } else {
        nq = n;
        nw = pyclap_imax(1, m);
    }
    (void)nw;   /* NW feeds only the dropped LWORK check */
    /* (parameter checks and XERBLA dropped) */

    /* Fortran: OPTS = SIDE // TRANS in the ILAENV calls below */
    opts2[0] = side;
    opts2[1] = trans;
    opts2[2] = '\0';
    if (applyq) {
        if (left) {
            nb = pyclap_ilaenv(1, "DORMQR", opts2, m-1, n, m-1, -1);
        } else {
            nb = pyclap_ilaenv(1, "DORMQR", opts2, m, n-1, n-1, -1);
        }
    } else {
        if (left) {
            nb = pyclap_ilaenv(1, "DORMLQ", opts2, m-1, n, m-1, -1);
        } else {
            nb = pyclap_ilaenv(1, "DORMLQ", opts2, m, n-1, n-1, -1);
        }
    }
    lwkopt = nw*nb;
    work[0] = (double)lwkopt;
    /* (LQUERY early return dropped) */

    /* Quick return if possible */
    work[0] = 1;
    if (m == 0 || n == 0)
        return;

    if (applyq) {
        /* Apply Q */
        if (nq >= k) {
            /* Q was determined by a call to DGEBRD with nq >= k */
            pyclap_dormqr(side, trans, m, n, k, a, lda, tau, c, ldc,
                          work, lwork, &iinfo);
        } else if (nq > 1) {
            /* Q was determined by a call to DGEBRD with nq < k */
            if (left) {
                mi = m - 1;
                ni = n;
                i1 = 2;
                i2 = 1;
            } else {
                mi = m;
                ni = n - 1;
                i1 = 1;
                i2 = 2;
            }
            pyclap_dormqr(side, trans, mi, ni, nq-1, &a[1 + 0*lda], lda,
                          tau, &c[(i1-1) + (i2-1)*ldc], ldc, work, lwork,
                          &iinfo);
        }
    } else {
        /* Apply P */
        if (notran) {
            transt = 'T';
        } else {
            transt = 'N';
        }
        if (nq > k) {
            /* P was determined by a call to DGEBRD with nq > k */
            pyclap_dormlq(side, transt, m, n, k, a, lda, tau, c, ldc,
                          work, lwork, &iinfo);
        } else if (nq > 1) {
            /* P was determined by a call to DGEBRD with nq <= k */
            if (left) {
                mi = m - 1;
                ni = n;
                i1 = 2;
                i2 = 1;
            } else {
                mi = m;
                ni = n - 1;
                i1 = 1;
                i2 = 2;
            }
            pyclap_dormlq(side, transt, mi, ni, nq-1, &a[0 + 1*lda], lda,
                          tau, &c[(i1-1) + (i2-1)*ldc], ldc, work, lwork,
                          &iinfo);
        }
    }
    work[0] = (double)lwkopt;
}

#endif /* PYCLAP_LAPACK_MID_H */
