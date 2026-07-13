/*
 * lapack_leaf.h -- faithful C99 transliteration of reference LAPACK/BLAS
 * leaf routines (netlib master, fetched 2026), for the pycalphad
 * GPU/CPU-shared kernel backend.
 *
 * Conventions:
 *   - Column-major layout preserved: A(I,J) -> a[(I-1) + (J-1)*lda].
 *   - Loop variables kept 1-based to mirror the Fortran bounds/direction
 *     exactly; indexing subtracts 1 at the access site.
 *   - Numerical operation ORDER matches the Fortran statement-for-statement
 *     (goal: bitwise-identical results). Only XERBLA/INFO<0 argument
 *     validation was dropped.
 *   - Every GOTO converted to explicit control flow is documented with a
 *     comment naming the Fortran label.
 *   - Fortran SIGN(a,b) is translated as copysign(a,b) (gfortran lowers
 *     SIGN to copysign for reals, honoring the sign bit of zero).
 *   - No dynamic allocation, no printf, C99 only. All functions are
 *     `__device__ static`; the consuming header defines __device__ away
 *     for CPU builds.
 */
#ifndef PYCLAP_LAPACK_LEAF_H
#define PYCLAP_LAPACK_LEAF_H

#if !defined(__CUDACC_RTC__) && !defined(__HIPRTC__)
#if !defined(__CUDACC_RTC__) && !defined(__HIPCC_RTC__)
#include <math.h>
#endif
#if !defined(__CUDACC_RTC__) && !defined(__HIPCC_RTC__)
#include <float.h>
#endif
#endif

#ifndef __device__
#define __device__
#endif

/* <float.h> fallbacks for RTC compilers without hosted headers.
   Values are the IEEE-754 binary64 constants, hex-exact. */
#ifndef DBL_EPSILON
#define DBL_EPSILON 0x1p-52
#endif
#ifndef DBL_MIN
#define DBL_MIN 0x1p-1022
#endif
#ifndef DBL_MAX
#define DBL_MAX 0x1.fffffffffffffp+1023
#endif
#ifndef DBL_MANT_DIG
#define DBL_MANT_DIG 53
#endif
#ifndef DBL_MIN_EXP
#define DBL_MIN_EXP (-1021)
#endif
#ifndef DBL_MAX_EXP
#define DBL_MAX_EXP 1024
#endif

/* ------------------------------------------------------------------ */
/* Small integer helpers (Fortran MAX/MIN intrinsics on INTEGER).      */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_imax(int a, int b) { return a > b ? a : b; }
__device__ static int pyclap_imin(int a, int b) { return a < b ? a : b; }

/* ------------------------------------------------------------------ */
/* LSAME (LAPACK master 2026): tests if CA is the same letter as CB    */
/* regardless of case. ASCII branch of the Fortran reference.          */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_lsame(char ca, char cb)
{
    int inta, intb;
    if (ca == cb) return 1;
    inta = (int)(unsigned char)ca;
    intb = (int)(unsigned char)cb;
    /* ASCII: lower-case letters are upper-case + 32 */
    if (inta >= 97 && inta <= 122) inta = inta - 32;
    if (intb >= 97 && intb <= 122) intb = intb - 32;
    return inta == intb;
}

/* ------------------------------------------------------------------ */
/* DLAISNAN (LAPACK master 2026): tests DIN1 and DIN2 for inequality;  */
/* NaN check without optimizer interference.                           */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_dlaisnan(double din1, double din2)
{
    return din1 != din2;
}

/* ------------------------------------------------------------------ */
/* DISNAN (LAPACK master 2026): returns .TRUE. if its argument is NaN. */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_disnan(double din)
{
    return pyclap_dlaisnan(din, din);
}

/* ------------------------------------------------------------------ */
/* DLAMCH (LAPACK master 2026): determines double precision machine    */
/* parameters. Implemented from <float.h> constants exactly as the     */
/* Fortran computes them from EPSILON/TINY/HUGE/RADIX/DIGITS/          */
/* MINEXPONENT/MAXEXPONENT (rounding assumed, so eps = EPSILON*0.5).   */
/* ------------------------------------------------------------------ */
__device__ static double pyclap_dlamch(char cmach)
{
    const double one = 1.0, zero = 0.0;
    double rnd, eps, sfmin, small_, rmach;
    /* Assume rounding, not chopping. Always. */
    rnd = one;
    if (one == rnd) {
        eps = DBL_EPSILON * 0.5;    /* EPSILON(ZERO) * 0.5 */
    } else {
        eps = DBL_EPSILON;
    }
    if (pyclap_lsame(cmach, 'E')) {
        rmach = eps;
    } else if (pyclap_lsame(cmach, 'S')) {
        sfmin = DBL_MIN;            /* TINY(ZERO) */
        small_ = one / DBL_MAX;     /* ONE / HUGE(ZERO) */
        if (small_ >= sfmin) {
            /* Use SMALL plus a bit, to avoid the possibility of rounding
               causing overflow when computing 1/sfmin. */
            sfmin = small_ * (one + eps);
        }
        rmach = sfmin;
    } else if (pyclap_lsame(cmach, 'B')) {
        rmach = 2.0;                        /* RADIX(ZERO) */
    } else if (pyclap_lsame(cmach, 'P')) {
        rmach = eps * 2.0;                  /* EPS * RADIX(ZERO) */
    } else if (pyclap_lsame(cmach, 'N')) {
        rmach = (double)DBL_MANT_DIG;       /* DIGITS(ZERO) = 53 */
    } else if (pyclap_lsame(cmach, 'R')) {
        rmach = rnd;
    } else if (pyclap_lsame(cmach, 'M')) {
        rmach = (double)DBL_MIN_EXP;        /* MINEXPONENT(ZERO) = -1021 */
    } else if (pyclap_lsame(cmach, 'U')) {
        rmach = DBL_MIN;                    /* TINY(ZERO) */
    } else if (pyclap_lsame(cmach, 'L')) {
        rmach = (double)DBL_MAX_EXP;        /* MAXEXPONENT(ZERO) = 1024 */
    } else if (pyclap_lsame(cmach, 'O')) {
        rmach = DBL_MAX;                    /* HUGE(ZERO) */
    } else {
        rmach = zero;
    }
    return rmach;
}

/* ================================================================== */
/* BLAS level 1                                                        */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* DCOPY (reference BLAS, LAPACK master 2026): copies a vector, x, to  */
/* a vector, y; uses unrolled loops for increments equal to 1.         */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dcopy(int n, const double* dx, int incx,
                                    double* dy, int incy)
{
    int i, ix, iy, m, mp1;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        /* clean-up loop */
        m = n % 7;
        if (m != 0) {
            for (i = 1; i <= m; i++) {
                dy[i-1] = dx[i-1];
            }
            if (n < 7) return;
        }
        mp1 = m + 1;
        for (i = mp1; i <= n; i += 7) {
            dy[i-1] = dx[i-1];
            dy[i]   = dx[i];
            dy[i+1] = dx[i+1];
            dy[i+2] = dx[i+2];
            dy[i+3] = dx[i+3];
            dy[i+4] = dx[i+4];
            dy[i+5] = dx[i+5];
        }
    } else {
        /* unequal increments or equal increments not equal to 1 */
        ix = 1;
        iy = 1;
        if (incx < 0) ix = (-n + 1) * incx + 1;
        if (incy < 0) iy = (-n + 1) * incy + 1;
        for (i = 1; i <= n; i++) {
            dy[iy-1] = dx[ix-1];
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

/* ------------------------------------------------------------------ */
/* DSCAL (reference BLAS, LAPACK master 2026): scales a vector by a    */
/* constant; uses unrolled loops for increment equal to 1.             */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dscal(int n, double da, double* dx, int incx)
{
    const double one = 1.0;
    int i, m, mp1, nincx;
    if (n <= 0 || incx <= 0 || da == one) return;
    if (incx == 1) {
        /* clean-up loop */
        m = n % 5;
        if (m != 0) {
            for (i = 1; i <= m; i++) {
                dx[i-1] = da * dx[i-1];
            }
            if (n < 5) return;
        }
        mp1 = m + 1;
        for (i = mp1; i <= n; i += 5) {
            dx[i-1] = da * dx[i-1];
            dx[i]   = da * dx[i];
            dx[i+1] = da * dx[i+1];
            dx[i+2] = da * dx[i+2];
            dx[i+3] = da * dx[i+3];
        }
    } else {
        /* increment not equal to 1 */
        nincx = n * incx;
        for (i = 1; i <= nincx; i += incx) {
            dx[i-1] = da * dx[i-1];
        }
    }
}

/* ------------------------------------------------------------------ */
/* DSWAP (reference BLAS, LAPACK master 2026): interchanges two        */
/* vectors; uses unrolled loops for increments equal to 1.             */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dswap(int n, double* dx, int incx,
                                    double* dy, int incy)
{
    double dtemp;
    int i, ix, iy, m, mp1;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        /* clean-up loop */
        m = n % 3;
        if (m != 0) {
            for (i = 1; i <= m; i++) {
                dtemp = dx[i-1];
                dx[i-1] = dy[i-1];
                dy[i-1] = dtemp;
            }
            if (n < 3) return;
        }
        mp1 = m + 1;
        for (i = mp1; i <= n; i += 3) {
            dtemp = dx[i-1];
            dx[i-1] = dy[i-1];
            dy[i-1] = dtemp;
            dtemp = dx[i];
            dx[i] = dy[i];
            dy[i] = dtemp;
            dtemp = dx[i+1];
            dx[i+1] = dy[i+1];
            dy[i+1] = dtemp;
        }
    } else {
        /* unequal increments or equal increments not equal to 1 */
        ix = 1;
        iy = 1;
        if (incx < 0) ix = (-n + 1) * incx + 1;
        if (incy < 0) iy = (-n + 1) * incy + 1;
        for (i = 1; i <= n; i++) {
            dtemp = dx[ix-1];
            dx[ix-1] = dy[iy-1];
            dy[iy-1] = dtemp;
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

/* ------------------------------------------------------------------ */
/* DAXPY (reference BLAS, LAPACK master 2026): constant times a vector */
/* plus a vector; uses unrolled loops for increments equal to one.     */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_daxpy(int n, double da, const double* dx,
                                    int incx, double* dy, int incy)
{
    int i, ix, iy, m, mp1;
    if (n <= 0) return;
    if (da == 0.0) return;
    if (incx == 1 && incy == 1) {
        /* clean-up loop */
        m = n % 4;
        if (m != 0) {
            for (i = 1; i <= m; i++) {
                dy[i-1] = dy[i-1] + da * dx[i-1];
            }
        }
        if (n < 4) return;   /* NOTE: outside the m!=0 block, as in Fortran */
        mp1 = m + 1;
        for (i = mp1; i <= n; i += 4) {
            dy[i-1] = dy[i-1] + da * dx[i-1];
            dy[i]   = dy[i]   + da * dx[i];
            dy[i+1] = dy[i+1] + da * dx[i+1];
            dy[i+2] = dy[i+2] + da * dx[i+2];
        }
    } else {
        /* unequal increments or equal increments not equal to 1 */
        ix = 1;
        iy = 1;
        if (incx < 0) ix = (-n + 1) * incx + 1;
        if (incy < 0) iy = (-n + 1) * incy + 1;
        for (i = 1; i <= n; i++) {
            dy[iy-1] = dy[iy-1] + da * dx[ix-1];
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

/* ------------------------------------------------------------------ */
/* DDOT (reference BLAS, LAPACK master 2026): forms the dot product of */
/* two vectors; uses unrolled loops for increments equal to one.       */
/* ------------------------------------------------------------------ */
__device__ static double pyclap_ddot(int n, const double* dx, int incx,
                                     const double* dy, int incy)
{
    double dtemp;
    int i, ix, iy, m, mp1;
    dtemp = 0.0;
    if (n <= 0) return 0.0;
    if (incx == 1 && incy == 1) {
        /* clean-up loop */
        m = n % 5;
        if (m != 0) {
            for (i = 1; i <= m; i++) {
                dtemp = dtemp + dx[i-1] * dy[i-1];
            }
            if (n < 5) {
                return dtemp;
            }
        }
        mp1 = m + 1;
        for (i = mp1; i <= n; i += 5) {
            dtemp = dtemp + dx[i-1] * dy[i-1] + dx[i] * dy[i] +
                    dx[i+1] * dy[i+1] + dx[i+2] * dy[i+2] + dx[i+3] * dy[i+3];
        }
    } else {
        /* unequal increments or equal increments not equal to 1 */
        ix = 1;
        iy = 1;
        if (incx < 0) ix = (-n + 1) * incx + 1;
        if (incy < 0) iy = (-n + 1) * incy + 1;
        for (i = 1; i <= n; i++) {
            dtemp = dtemp + dx[ix-1] * dy[iy-1];
            ix = ix + incx;
            iy = iy + incy;
        }
    }
    return dtemp;
}

/* ------------------------------------------------------------------ */
/* DNRM2 (reference BLAS dnrm2.f90, LAPACK master 2026): returns the   */
/* euclidean norm of a vector: DNRM2 := sqrt( x'*x ). Blue's scaled    */
/* three-accumulator algorithm (Anderson 2017, Algorithm 978).         */
/* Constants (IEEE binary64, hex-exact):                               */
/*   tsml = 2**ceiling((minexp-1)/2)        = 2**-511                  */
/*   tbig = 2**floor((maxexp-digits+1)/2)   = 2**486                   */
/*   ssml = 2**(-floor((minexp-digits)/2))  = 2**537                   */
/*   sbig = 2**(-ceiling((maxexp+digits-1)/2)) = 2**-538               */
/* ------------------------------------------------------------------ */
__device__ static double pyclap_dnrm2(int n, const double* x, int incx)
{
    const double zero = 0.0;
    const double one  = 1.0;
    const double maxN = DBL_MAX;            /* huge(0.0_wp) */
    const double tsml = 0x1p-511;
    const double tbig = 0x1p+486;
    const double ssml = 0x1p+537;
    const double sbig = 0x1p-538;
    int i, ix;
    int notbig;
    double abig, amed, asml, ax, scl, sumsq, ymax, ymin;

    /* Quick return if possible */
    if (n <= 0) return zero;

    scl = one;
    sumsq = zero;

    /* Compute the sum of squares in 3 accumulators:
          abig -- sums of squares scaled down to avoid overflow
          asml -- sums of squares scaled up to avoid underflow
          amed -- sums of squares that do not require scaling         */
    notbig = 1;
    asml = zero;
    amed = zero;
    abig = zero;
    ix = 1;
    if (incx < 0) ix = 1 - (n - 1) * incx;
    for (i = 1; i <= n; i++) {
        ax = fabs(x[ix-1]);
        if (ax > tbig) {
            abig = abig + (ax * sbig) * (ax * sbig);
            notbig = 0;
        } else if (ax < tsml) {
            if (notbig) asml = asml + (ax * ssml) * (ax * ssml);
        } else {
            amed = amed + ax * ax;
        }
        ix = ix + incx;
    }
    /* Combine abig and amed or amed and asml if more than one
       accumulator was used. */
    if (abig > zero) {
        /* Combine abig and amed if abig > 0. */
        if ((amed > zero) || (amed > maxN) || (amed != amed)) {
            abig = abig + (amed * sbig) * sbig;
        }
        scl = one / sbig;
        sumsq = abig;
    } else if (asml > zero) {
        /* Combine amed and asml if asml > 0. */
        if ((amed > zero) || (amed > maxN) || (amed != amed)) {
            amed = sqrt(amed);
            asml = sqrt(asml) / ssml;
            if (asml > amed) {
                ymin = amed;
                ymax = asml;
            } else {
                ymin = asml;
                ymax = amed;
            }
            scl = one;
            sumsq = (ymax * ymax) * (one + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = one / ssml;
            sumsq = asml;
        }
    } else {
        /* Otherwise all values are mid-range */
        scl = one;
        sumsq = amed;
    }
    return scl * sqrt(sumsq);
}

/* ------------------------------------------------------------------ */
/* IDAMAX (reference BLAS, LAPACK master 2026): finds the index of the */
/* first element having maximum absolute value.                        */
/* NOTE: returns the Fortran 1-based index (0 on quick return).        */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_idamax(int n, const double* dx, int incx)
{
    double dmax;
    int i, ix, ret;
    ret = 0;
    if (n < 1 || incx <= 0) return ret;
    ret = 1;
    if (n == 1) return ret;
    if (incx == 1) {
        /* increment equal to 1 */
        dmax = fabs(dx[0]);
        for (i = 2; i <= n; i++) {
            if (fabs(dx[i-1]) > dmax) {
                ret = i;
                dmax = fabs(dx[i-1]);
            }
        }
    } else {
        /* increment not equal to 1 */
        ix = 1;
        dmax = fabs(dx[0]);
        ix = ix + incx;
        for (i = 2; i <= n; i++) {
            if (fabs(dx[ix-1]) > dmax) {
                ret = i;
                dmax = fabs(dx[ix-1]);
            }
            ix = ix + incx;
        }
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/* DROT (reference BLAS, LAPACK master 2026): applies a plane rotation.*/
/* ------------------------------------------------------------------ */
__device__ static void pyclap_drot(int n, double* dx, int incx,
                                   double* dy, int incy, double c, double s)
{
    double dtemp;
    int i, ix, iy;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        /* both increments equal to 1 */
        for (i = 1; i <= n; i++) {
            dtemp = c * dx[i-1] + s * dy[i-1];
            dy[i-1] = c * dy[i-1] - s * dx[i-1];
            dx[i-1] = dtemp;
        }
    } else {
        /* unequal increments or equal increments not equal to 1 */
        ix = 1;
        iy = 1;
        if (incx < 0) ix = (-n + 1) * incx + 1;
        if (incy < 0) iy = (-n + 1) * incy + 1;
        for (i = 1; i <= n; i++) {
            dtemp = c * dx[ix-1] + s * dy[iy-1];
            dy[iy-1] = c * dy[iy-1] - s * dx[ix-1];
            dx[ix-1] = dtemp;
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

/* ================================================================== */
/* BLAS level 2                                                        */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* DGEMV (reference BLAS, LAPACK master 2026): performs                */
/* y := alpha*A*x + beta*y  or  y := alpha*A**T*x + beta*y.            */
/* (Argument validation / XERBLA dropped.)                             */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dgemv(char trans, int m, int n, double alpha,
                                    const double* a, int lda,
                                    const double* x, int incx, double beta,
                                    double* y, int incy)
{
    const double one = 1.0, zero = 0.0;
    double temp;
    int i, ix, iy, j, jx, jy, kx, ky, lenx, leny;

    /* Quick return if possible. */
    if ((m == 0) || (n == 0) ||
        ((alpha == zero) && (beta == one))) return;

    /* Set LENX and LENY, the lengths of the vectors x and y, and set
       up the start points in X and Y. */
    if (pyclap_lsame(trans, 'N')) {
        lenx = n;
        leny = m;
    } else {
        lenx = m;
        leny = n;
    }
    if (incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (lenx - 1) * incx;
    }
    if (incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (leny - 1) * incy;
    }

    /* Start the operations. First form  y := beta*y. */
    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) {
                for (i = 1; i <= leny; i++) {          /* loop 10 */
                    y[i-1] = zero;
                }
            } else {
                for (i = 1; i <= leny; i++) {          /* loop 20 */
                    y[i-1] = beta * y[i-1];
                }
            }
        } else {
            iy = ky;
            if (beta == zero) {
                for (i = 1; i <= leny; i++) {          /* loop 30 */
                    y[iy-1] = zero;
                    iy = iy + incy;
                }
            } else {
                for (i = 1; i <= leny; i++) {          /* loop 40 */
                    y[iy-1] = beta * y[iy-1];
                    iy = iy + incy;
                }
            }
        }
    }
    if (alpha == zero) return;
    if (pyclap_lsame(trans, 'N')) {
        /* Form  y := alpha*A*x + y. */
        jx = kx;
        if (incy == 1) {
            for (j = 1; j <= n; j++) {                 /* loop 60 */
                temp = alpha * x[jx-1];
                for (i = 1; i <= m; i++) {             /* loop 50 */
                    y[i-1] = y[i-1] + temp * a[(i-1) + (j-1)*lda];
                }
                jx = jx + incx;
            }
        } else {
            for (j = 1; j <= n; j++) {                 /* loop 80 */
                temp = alpha * x[jx-1];
                iy = ky;
                for (i = 1; i <= m; i++) {             /* loop 70 */
                    y[iy-1] = y[iy-1] + temp * a[(i-1) + (j-1)*lda];
                    iy = iy + incy;
                }
                jx = jx + incx;
            }
        }
    } else {
        /* Form  y := alpha*A**T*x + y. */
        jy = ky;
        if (incx == 1) {
            for (j = 1; j <= n; j++) {                 /* loop 100 */
                temp = zero;
                for (i = 1; i <= m; i++) {             /* loop 90 */
                    temp = temp + a[(i-1) + (j-1)*lda] * x[i-1];
                }
                y[jy-1] = y[jy-1] + alpha * temp;
                jy = jy + incy;
            }
        } else {
            for (j = 1; j <= n; j++) {                 /* loop 120 */
                temp = zero;
                ix = kx;
                for (i = 1; i <= m; i++) {             /* loop 110 */
                    temp = temp + a[(i-1) + (j-1)*lda] * x[ix-1];
                    ix = ix + incx;
                }
                y[jy-1] = y[jy-1] + alpha * temp;
                jy = jy + incy;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* DGER (reference BLAS, LAPACK master 2026): performs the rank 1      */
/* operation  A := alpha*x*y**T + A.                                   */
/* (Argument validation / XERBLA dropped.)                             */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dger(int m, int n, double alpha,
                                   const double* x, int incx,
                                   const double* y, int incy,
                                   double* a, int lda)
{
    const double zero = 0.0;
    double temp;
    int i, ix, j, jy, kx;

    /* Quick return if possible. */
    if ((m == 0) || (n == 0) || (alpha == zero)) return;

    if (incy > 0) {
        jy = 1;
    } else {
        jy = 1 - (n - 1) * incy;
    }
    if (incx == 1) {
        for (j = 1; j <= n; j++) {                     /* loop 20 */
            if (y[jy-1] != zero) {
                temp = alpha * y[jy-1];
                for (i = 1; i <= m; i++) {             /* loop 10 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] + x[i-1] * temp;
                }
            }
            jy = jy + incy;
        }
    } else {
        if (incx > 0) {
            kx = 1;
        } else {
            kx = 1 - (m - 1) * incx;
        }
        for (j = 1; j <= n; j++) {                     /* loop 40 */
            if (y[jy-1] != zero) {
                temp = alpha * y[jy-1];
                ix = kx;
                for (i = 1; i <= m; i++) {             /* loop 30 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] + x[ix-1] * temp;
                    ix = ix + incx;
                }
            }
            jy = jy + incy;
        }
    }
}

/* ------------------------------------------------------------------ */
/* DTRMV (reference BLAS, LAPACK master 2026): performs                */
/* x := A*x or x := A**T*x, A an n-by-n (unit/non-unit) triangular     */
/* matrix. (Argument validation / XERBLA dropped.)                     */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dtrmv(char uplo, char trans, char diag, int n,
                                    const double* a, int lda,
                                    double* x, int incx)
{
    double temp;
    int i, ix, j, jx, kx;
    int nounit;

    /* Quick return if possible. */
    if (n == 0) return;

    nounit = pyclap_lsame(diag, 'N');

    /* Set up the start point in X if the increment is not unity. This
       will be (N-1)*INCX too small for descending loops.
       (kx is only referenced when incx != 1, as in the Fortran, where it
       is otherwise left undefined; initialized here to silence warnings.) */
    kx = 1;
    if (incx <= 0) {
        kx = 1 - (n - 1) * incx;
    } else if (incx != 1) {
        kx = 1;
    }

    if (pyclap_lsame(trans, 'N')) {
        /* Form  x := A*x. */
        if (pyclap_lsame(uplo, 'U')) {
            if (incx == 1) {
                for (j = 1; j <= n; j++) {             /* loop 20 */
                    temp = x[j-1];
                    for (i = 1; i <= j - 1; i++) {     /* loop 10 */
                        x[i-1] = x[i-1] + temp * a[(i-1) + (j-1)*lda];
                    }
                    if (nounit) x[j-1] = x[j-1] * a[(j-1) + (j-1)*lda];
                }
            } else {
                jx = kx;
                for (j = 1; j <= n; j++) {             /* loop 40 */
                    temp = x[jx-1];
                    ix = kx;
                    for (i = 1; i <= j - 1; i++) {     /* loop 30 */
                        x[ix-1] = x[ix-1] + temp * a[(i-1) + (j-1)*lda];
                        ix = ix + incx;
                    }
                    if (nounit) x[jx-1] = x[jx-1] * a[(j-1) + (j-1)*lda];
                    jx = jx + incx;
                }
            }
        } else {
            if (incx == 1) {
                for (j = n; j >= 1; j--) {             /* loop 60 */
                    temp = x[j-1];
                    for (i = n; i >= j + 1; i--) {     /* loop 50 */
                        x[i-1] = x[i-1] + temp * a[(i-1) + (j-1)*lda];
                    }
                    if (nounit) x[j-1] = x[j-1] * a[(j-1) + (j-1)*lda];
                }
            } else {
                kx = kx + (n - 1) * incx;
                jx = kx;
                for (j = n; j >= 1; j--) {             /* loop 80 */
                    temp = x[jx-1];
                    ix = kx;
                    for (i = n; i >= j + 1; i--) {     /* loop 70 */
                        x[ix-1] = x[ix-1] + temp * a[(i-1) + (j-1)*lda];
                        ix = ix - incx;
                    }
                    if (nounit) x[jx-1] = x[jx-1] * a[(j-1) + (j-1)*lda];
                    jx = jx - incx;
                }
            }
        }
    } else {
        /* Form  x := A**T*x. */
        if (pyclap_lsame(uplo, 'U')) {
            if (incx == 1) {
                for (j = n; j >= 1; j--) {             /* loop 100 */
                    temp = x[j-1];
                    if (nounit) temp = temp * a[(j-1) + (j-1)*lda];
                    for (i = j - 1; i >= 1; i--) {     /* loop 90 */
                        temp = temp + a[(i-1) + (j-1)*lda] * x[i-1];
                    }
                    x[j-1] = temp;
                }
            } else {
                jx = kx + (n - 1) * incx;
                for (j = n; j >= 1; j--) {             /* loop 120 */
                    temp = x[jx-1];
                    ix = jx;
                    if (nounit) temp = temp * a[(j-1) + (j-1)*lda];
                    for (i = j - 1; i >= 1; i--) {     /* loop 110 */
                        ix = ix - incx;
                        temp = temp + a[(i-1) + (j-1)*lda] * x[ix-1];
                    }
                    x[jx-1] = temp;
                    jx = jx - incx;
                }
            }
        } else {
            if (incx == 1) {
                for (j = 1; j <= n; j++) {             /* loop 140 */
                    temp = x[j-1];
                    if (nounit) temp = temp * a[(j-1) + (j-1)*lda];
                    for (i = j + 1; i <= n; i++) {     /* loop 130 */
                        temp = temp + a[(i-1) + (j-1)*lda] * x[i-1];
                    }
                    x[j-1] = temp;
                }
            } else {
                jx = kx;
                for (j = 1; j <= n; j++) {             /* loop 160 */
                    temp = x[jx-1];
                    ix = jx;
                    if (nounit) temp = temp * a[(j-1) + (j-1)*lda];
                    for (i = j + 1; i <= n; i++) {     /* loop 150 */
                        ix = ix + incx;
                        temp = temp + a[(i-1) + (j-1)*lda] * x[ix-1];
                    }
                    x[jx-1] = temp;
                    jx = jx + incx;
                }
            }
        }
    }
}

/* ================================================================== */
/* LAPACK auxiliary routines                                           */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* DLAPY2 (LAPACK master 2026): returns sqrt(x**2+y**2), taking care   */
/* not to cause unnecessary overflow / underflow.                      */
/* ------------------------------------------------------------------ */
__device__ static double pyclap_dlapy2(double x, double y)
{
    const double zero = 0.0;
    const double one = 1.0;
    double w, xabs, yabs, z, hugeval;
    double result = 0.0;    /* function result variable */
    int x_is_nan, y_is_nan;

    x_is_nan = pyclap_disnan(x);
    y_is_nan = pyclap_disnan(y);
    if (x_is_nan) result = x;
    if (y_is_nan) result = y;
    hugeval = pyclap_dlamch('O');   /* 'Overflow' */

    if (!(x_is_nan || y_is_nan)) {
        xabs = fabs(x);
        yabs = fabs(y);
        w = fmax(xabs, yabs);
        z = fmin(xabs, yabs);
        if (z == zero || w > hugeval) {
            result = w;
        } else {
            result = w * sqrt(one + (z / w) * (z / w));
        }
    }
    return result;
}

/* ------------------------------------------------------------------ */
/* DLASSQ (LAPACK dlassq.f90, master 2026): updates a sum of squares   */
/* represented in scaled form:                                         */
/*   (scale_out**2)*sumsq_out = x(1)**2 + ... + x(n)**2                */
/*                              + (scale**2)*sumsq.                    */
/* Blue's scaling constants from la_constants.f90 (IEEE binary64,      */
/* hex-exact): tsml=2**-511, tbig=2**486, ssml=2**537, sbig=2**-538.   */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlassq(int n, const double* x, int incx,
                                     double* scale, double* sumsq)
{
    const double zero = 0.0;
    const double one  = 1.0;
    const double tsml = 0x1p-511;
    const double tbig = 0x1p+486;
    const double ssml = 0x1p+537;
    const double sbig = 0x1p-538;
    int i, ix;
    int notbig;
    double abig, amed, asml, ax, ymax, ymin;

    /* Quick return if possible */
    if (pyclap_disnan(*scale) || pyclap_disnan(*sumsq)) return;
    if (*sumsq == zero) *scale = one;
    if (*scale == zero) {
        *scale = one;
        *sumsq = zero;
    }
    if (n <= 0) {
        return;
    }

    /* Compute the sum of squares in 3 accumulators:
          abig -- sums of squares scaled down to avoid overflow
          asml -- sums of squares scaled up to avoid underflow
          amed -- sums of squares that do not require scaling         */
    notbig = 1;
    asml = zero;
    amed = zero;
    abig = zero;
    ix = 1;
    if (incx < 0) ix = 1 - (n - 1) * incx;
    for (i = 1; i <= n; i++) {
        ax = fabs(x[ix-1]);
        if (ax > tbig) {
            abig = abig + (ax * sbig) * (ax * sbig);
            notbig = 0;
        } else if (ax < tsml) {
            if (notbig) asml = asml + (ax * ssml) * (ax * ssml);
        } else {
            amed = amed + ax * ax;
        }
        ix = ix + incx;
    }
    /* Put the existing sum of squares into one of the accumulators */
    if (*sumsq > zero) {
        ax = *scale * sqrt(*sumsq);
        if (ax > tbig) {
            if (*scale > one) {
                *scale = *scale * sbig;
                abig = abig + *scale * (*scale * *sumsq);
            } else {
                /* sumsq > tbig^2 => (sbig * (sbig * sumsq)) is representable */
                abig = abig + *scale * (*scale * (sbig * (sbig * *sumsq)));
            }
        } else if (ax < tsml) {
            if (notbig) {
                if (*scale < one) {
                    *scale = *scale * ssml;
                    asml = asml + *scale * (*scale * *sumsq);
                } else {
                    /* sumsq < tsml^2 => (ssml * (ssml * sumsq)) is representable */
                    asml = asml + *scale * (*scale * (ssml * (ssml * *sumsq)));
                }
            }
        } else {
            amed = amed + *scale * (*scale * *sumsq);
        }
    }
    /* Combine abig and amed or amed and asml if more than one
       accumulator was used. */
    if (abig > zero) {
        /* Combine abig and amed if abig > 0. */
        if (amed > zero || pyclap_disnan(amed)) {
            abig = abig + (amed * sbig) * sbig;
        }
        *scale = one / sbig;
        *sumsq = abig;
    } else if (asml > zero) {
        /* Combine amed and asml if asml > 0. */
        if (amed > zero || pyclap_disnan(amed)) {
            amed = sqrt(amed);
            asml = sqrt(asml) / ssml;
            if (asml > amed) {
                ymin = amed;
                ymax = asml;
            } else {
                ymin = asml;
                ymax = amed;
            }
            *scale = one;
            *sumsq = (ymax * ymax) * (one + (ymin / ymax) * (ymin / ymax));
        } else {
            *scale = one / ssml;
            *sumsq = asml;
        }
    } else {
        /* Otherwise all values are mid-range or zero */
        *scale = one;
        *sumsq = amed;
    }
}

/* ------------------------------------------------------------------ */
/* DLANGE (LAPACK master 2026): returns the value of the one norm, or  */
/* the Frobenius norm, or the infinity norm, or the element of largest */
/* absolute value of a real matrix A.                                  */
/* WORK is used only for NORM = 'I' (dimension >= M), as in Fortran.   */
/* ------------------------------------------------------------------ */
__device__ static double pyclap_dlange(char norm, int m, int n,
                                       const double* a, int lda, double* work)
{
    const double one = 1.0, zero = 0.0;
    int i, j;
    double scale, sum, value, temp;

    value = zero;   /* Fortran leaves VALUE undefined for invalid NORM */
    if (pyclap_imin(m, n) == 0) {
        value = zero;
    } else if (pyclap_lsame(norm, 'M')) {
        /* Find max(abs(A(i,j))). */
        value = zero;
        for (j = 1; j <= n; j++) {                     /* loop 20 */
            for (i = 1; i <= m; i++) {                 /* loop 10 */
                temp = fabs(a[(i-1) + (j-1)*lda]);
                if (value < temp || pyclap_disnan(temp)) value = temp;
            }
        }
    } else if (pyclap_lsame(norm, 'O') || norm == '1') {
        /* Find norm1(A). */
        value = zero;
        for (j = 1; j <= n; j++) {                     /* loop 40 */
            sum = zero;
            for (i = 1; i <= m; i++) {                 /* loop 30 */
                sum = sum + fabs(a[(i-1) + (j-1)*lda]);
            }
            if (value < sum || pyclap_disnan(sum)) value = sum;
        }
    } else if (pyclap_lsame(norm, 'I')) {
        /* Find normI(A). */
        for (i = 1; i <= m; i++) {                     /* loop 50 */
            work[i-1] = zero;
        }
        for (j = 1; j <= n; j++) {                     /* loop 70 */
            for (i = 1; i <= m; i++) {                 /* loop 60 */
                work[i-1] = work[i-1] + fabs(a[(i-1) + (j-1)*lda]);
            }
        }
        value = zero;
        for (i = 1; i <= m; i++) {                     /* loop 80 */
            temp = work[i-1];
            if (value < temp || pyclap_disnan(temp)) value = temp;
        }
    } else if (pyclap_lsame(norm, 'F') || pyclap_lsame(norm, 'E')) {
        /* Find normF(A). */
        scale = zero;
        sum = one;
        for (j = 1; j <= n; j++) {                     /* loop 90 */
            pyclap_dlassq(m, &a[0 + (j-1)*lda], 1, &scale, &sum);
        }
        value = scale * sqrt(sum);
    }
    return value;
}

/* ------------------------------------------------------------------ */
/* DLASCL (LAPACK master 2026): multiplies the M by N real matrix A by */
/* the real scalar CTO/CFROM, done without over/underflow as long as   */
/* the final result CTO*A(I,J)/CFROM does not over/underflow.          */
/* (Argument validation / XERBLA dropped; TYPE still selects the       */
/* storage layout, and an unrecognized TYPE returns with no action,    */
/* mirroring the Fortran error path minus the XERBLA call.)            */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlascl(char type, int kl, int ku,
                                     double cfrom, double cto, int m, int n,
                                     double* a, int lda, int* info)
{
    const double zero = 0.0, one = 1.0;
    int done;
    int i, itype, j, k1, k2, k3, k4;
    double bignum, cfrom1, cfromc, cto1, ctoc, mul, smlnum;

    *info = 0;

    if (pyclap_lsame(type, 'G')) {
        itype = 0;
    } else if (pyclap_lsame(type, 'L')) {
        itype = 1;
    } else if (pyclap_lsame(type, 'U')) {
        itype = 2;
    } else if (pyclap_lsame(type, 'H')) {
        itype = 3;
    } else if (pyclap_lsame(type, 'B')) {
        itype = 4;
    } else if (pyclap_lsame(type, 'Q')) {
        itype = 5;
    } else if (pyclap_lsame(type, 'Z')) {
        itype = 6;
    } else {
        itype = -1;
    }
    if (itype == -1) {
        /* Fortran would XERBLA and return; validation dropped. */
        *info = -1;
        return;
    }

    /* Quick return if possible */
    if (n == 0 || m == 0) return;

    /* Get machine parameters */
    smlnum = pyclap_dlamch('S');
    bignum = one / smlnum;

    cfromc = cfrom;
    ctoc = cto;

    for (;;) {          /* Fortran label 10: iterative-scaling loop */
        cfrom1 = cfromc * smlnum;
        if (cfrom1 == cfromc) {
            /* CFROMC is an inf.  Multiply by a correctly signed zero for
               finite CTOC, or a NaN if CTOC is infinite. */
            mul = ctoc / cfromc;
            done = 1;
            cto1 = ctoc;
            (void)cto1;
        } else {
            cto1 = ctoc / bignum;
            if (cto1 == ctoc) {
                /* CTOC is either 0 or an inf.  In both cases, CTOC itself
                   serves as the correct multiplication factor. */
                mul = ctoc;
                done = 1;
                cfromc = one;
            } else if (fabs(cfrom1) > fabs(ctoc) && ctoc != zero) {
                mul = smlnum;
                done = 0;
                cfromc = cfrom1;
            } else if (fabs(cto1) > fabs(cfromc)) {
                mul = bignum;
                done = 0;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = 1;
                if (mul == one) return;
            }
        }

        if (itype == 0) {
            /* Full matrix */
            for (j = 1; j <= n; j++) {                             /* loop 30 */
                for (i = 1; i <= m; i++) {                         /* loop 20 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] * mul;
                }
            }
        } else if (itype == 1) {
            /* Lower triangular matrix */
            for (j = 1; j <= pyclap_imin(m, n); j++) {             /* loop 50 */
                for (i = j; i <= m; i++) {                         /* loop 40 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] * mul;
                }
            }
        } else if (itype == 2) {
            /* Upper triangular matrix */
            for (j = 1; j <= n; j++) {                             /* loop 70 */
                for (i = 1; i <= pyclap_imin(j, m); i++) {         /* loop 60 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] * mul;
                }
            }
        } else if (itype == 3) {
            /* Upper Hessenberg matrix */
            for (j = 1; j <= n; j++) {                             /* loop 90 */
                for (i = 1; i <= pyclap_imin(j + 1, m); i++) {     /* loop 80 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] * mul;
                }
            }
        } else if (itype == 4) {
            /* Lower half of a symmetric band matrix */
            k3 = kl + 1;
            k4 = n + 1;
            for (j = 1; j <= n; j++) {                             /* loop 110 */
                for (i = 1; i <= pyclap_imin(k3, k4 - j); i++) {   /* loop 100 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] * mul;
                }
            }
        } else if (itype == 5) {
            /* Upper half of a symmetric band matrix */
            k1 = ku + 2;
            k3 = ku + 1;
            for (j = 1; j <= n; j++) {                             /* loop 130 */
                for (i = pyclap_imax(k1 - j, 1); i <= k3; i++) {   /* loop 120 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] * mul;
                }
            }
        } else if (itype == 6) {
            /* Band matrix */
            k1 = kl + ku + 2;
            k2 = kl + 1;
            k3 = 2 * kl + ku + 1;
            k4 = kl + ku + 1 + m;
            for (j = 1; j <= n; j++) {                             /* loop 150 */
                for (i = pyclap_imax(k1 - j, k2);
                     i <= pyclap_imin(k3, k4 - j); i++) {          /* loop 140 */
                    a[(i-1) + (j-1)*lda] = a[(i-1) + (j-1)*lda] * mul;
                }
            }
        }

        if (done) break;   /* Fortran: IF( .NOT.DONE ) GO TO 10 */
    }
}

/* ------------------------------------------------------------------ */
/* DLASET (LAPACK master 2026): initializes the off-diagonal elements  */
/* of a matrix to ALPHA and the diagonal elements to BETA.             */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlaset(char uplo, int m, int n, double alpha,
                                     double beta, double* a, int lda)
{
    int i, j;
    if (pyclap_lsame(uplo, 'U')) {
        /* Set the strictly upper triangular or trapezoidal part of the
           array to ALPHA. */
        for (j = 2; j <= n; j++) {                     /* loop 20 */
            for (i = 1; i <= pyclap_imin(j - 1, m); i++) {  /* loop 10 */
                a[(i-1) + (j-1)*lda] = alpha;
            }
        }
    } else if (pyclap_lsame(uplo, 'L')) {
        /* Set the strictly lower triangular or trapezoidal part of the
           array to ALPHA. */
        for (j = 1; j <= pyclap_imin(m, n); j++) {     /* loop 40 */
            for (i = j + 1; i <= m; i++) {             /* loop 30 */
                a[(i-1) + (j-1)*lda] = alpha;
            }
        }
    } else {
        /* Set the leading m-by-n submatrix to ALPHA. */
        for (j = 1; j <= n; j++) {                     /* loop 60 */
            for (i = 1; i <= m; i++) {                 /* loop 50 */
                a[(i-1) + (j-1)*lda] = alpha;
            }
        }
    }
    /* Set the first min(M,N) diagonal elements to BETA. */
    for (i = 1; i <= pyclap_imin(m, n); i++) {         /* loop 70 */
        a[(i-1) + (i-1)*lda] = beta;
    }
}

/* ------------------------------------------------------------------ */
/* DLACPY (LAPACK master 2026): copies all or part of a two-dimensional*/
/* matrix A to another matrix B.                                       */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlacpy(char uplo, int m, int n,
                                     const double* a, int lda,
                                     double* b, int ldb)
{
    int i, j;
    if (pyclap_lsame(uplo, 'U')) {
        for (j = 1; j <= n; j++) {                     /* loop 20 */
            for (i = 1; i <= pyclap_imin(j, m); i++) { /* loop 10 */
                b[(i-1) + (j-1)*ldb] = a[(i-1) + (j-1)*lda];
            }
        }
    } else if (pyclap_lsame(uplo, 'L')) {
        for (j = 1; j <= pyclap_imin(m, n); j++) {     /* loop 40 */
            for (i = j; i <= m; i++) {                 /* loop 30 */
                b[(i-1) + (j-1)*ldb] = a[(i-1) + (j-1)*lda];
            }
        }
    } else {
        for (j = 1; j <= n; j++) {                     /* loop 60 */
            for (i = 1; i <= m; i++) {                 /* loop 50 */
                b[(i-1) + (j-1)*ldb] = a[(i-1) + (j-1)*lda];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* DLARTG (LAPACK dlartg.f90, master 2026): generates a plane rotation */
/* with real cosine and real sine so that [C S; -S C].[F;G] = [R;0].   */
/* Constants from la_constants.f90 (IEEE binary64, hex-exact):         */
/*   safmin = 2**max(minexp-1, 1-maxexp) = 2**-1022                    */
/*   safmax = 1/safmin                   = 2**1022                     */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlartg(double f, double g,
                                     double* c, double* s, double* r)
{
    const double zero = 0.0;
    const double one  = 1.0;
    const double safmin = 0x1p-1022;
    const double safmax = 0x1p+1022;
    double d, f1, fs, g1, gs, u, rtmin, rtmax;

    rtmin = sqrt(safmin);
    rtmax = sqrt(safmax / 2);

    f1 = fabs(f);
    g1 = fabs(g);
    if (g == zero) {
        *c = one;
        *s = zero;
        *r = f;
    } else if (f == zero) {
        *c = zero;
        *s = copysign(one, g);      /* sign(one, g) */
        *r = g1;
    } else if (f1 > rtmin && f1 < rtmax &&
               g1 > rtmin && g1 < rtmax) {
        d = sqrt(f * f + g * g);
        *c = f1 / d;
        *r = copysign(d, f);        /* sign(d, f) */
        *s = g / *r;
    } else {
        u = fmin(safmax, fmax(fmax(safmin, f1), g1));  /* min(safmax, max(safmin, f1, g1)) */
        fs = f / u;
        gs = g / u;
        d = sqrt(fs * fs + gs * gs);
        *c = fabs(fs) / d;
        *r = copysign(d, f);        /* sign(d, f) */
        *s = gs / *r;
        *r = *r * u;
    }
}

/* ------------------------------------------------------------------ */
/* DLAS2 (LAPACK master 2026): computes the singular values of the 2x2 */
/* matrix [[F, G], [0, H]]: SSMIN the smaller, SSMAX the larger.       */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlas2(double f, double g, double h,
                                    double* ssmin, double* ssmax)
{
    const double zero = 0.0;
    const double one  = 1.0;
    const double two  = 2.0;
    double as, at, au, c, fa, fhmn, fhmx, ga, ha;

    fa = fabs(f);
    ga = fabs(g);
    ha = fabs(h);
    fhmn = fmin(fa, ha);
    fhmx = fmax(fa, ha);
    if (fhmn == zero) {
        *ssmin = zero;
        if (fhmx == zero) {
            *ssmax = ga;
        } else {
            *ssmax = fmax(fhmx, ga) * sqrt(one +
                     (fmin(fhmx, ga) / fmax(fhmx, ga)) *
                     (fmin(fhmx, ga) / fmax(fhmx, ga)));
        }
    } else {
        if (ga < fhmx) {
            as = one + fhmn / fhmx;
            at = (fhmx - fhmn) / fhmx;
            au = (ga / fhmx) * (ga / fhmx);
            c = two / (sqrt(as * as + au) + sqrt(at * at + au));
            *ssmin = fhmn * c;
            *ssmax = fhmx / c;
        } else {
            au = fhmx / ga;
            if (au == zero) {
                /* Avoid possible harmful underflow if exponent range
                   asymmetric (true SSMIN may not underflow even if
                   AU underflows) */
                *ssmin = (fhmn * fhmx) / ga;
                *ssmax = ga;
            } else {
                as = one + fhmn / fhmx;
                at = (fhmx - fhmn) / fhmx;
                c = one / (sqrt(one + (as * au) * (as * au)) +
                           sqrt(one + (at * au) * (at * au)));
                *ssmin = (fhmn * c) * au;
                *ssmin = *ssmin + *ssmin;
                *ssmax = ga / (c + c);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* DLASV2 (LAPACK master 2026): computes the singular value            */
/* decomposition of the 2x2 triangular matrix [[F, G], [0, H]].        */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlasv2(double f, double g, double h,
                                     double* ssmin, double* ssmax,
                                     double* snr, double* csr,
                                     double* snl, double* csl)
{
    const double zero = 0.0;
    const double half = 0.5;
    const double one  = 1.0;
    const double two  = 2.0;
    const double four = 4.0;
    int gasmal, swap;
    int pmax;
    double a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m,
           mm, r, s, slt, srt, t, temp, tsign, tt;

    /* clt/crt/slt/srt are assigned on every reachable path in the
       Fortran; initialized here only to silence compiler warnings. */
    clt = zero; crt = zero; slt = zero; srt = zero;
    tsign = zero;

    ft = f;
    fa = fabs(ft);
    ht = h;
    ha = fabs(h);

    /* PMAX points to the maximum absolute element of matrix
         PMAX = 1 if F largest in absolute values
         PMAX = 2 if G largest in absolute values
         PMAX = 3 if H largest in absolute values                     */
    pmax = 1;
    swap = (ha > fa);
    if (swap) {
        pmax = 3;
        temp = ft;
        ft = ht;
        ht = temp;
        temp = fa;
        fa = ha;
        ha = temp;
        /* Now FA .ge. HA */
    }
    gt = g;
    ga = fabs(gt);
    if (ga == zero) {
        /* Diagonal matrix */
        *ssmin = ha;
        *ssmax = fa;
        clt = one;
        crt = one;
        slt = zero;
        srt = zero;
    } else {
        gasmal = 1;
        if (ga > fa) {
            pmax = 2;
            if ((fa / ga) < pyclap_dlamch('E')) {   /* DLAMCH('EPS') */
                /* Case of very large GA */
                gasmal = 0;
                *ssmax = ga;
                if (ha > one) {
                    *ssmin = fa / (ga / ha);
                } else {
                    *ssmin = (fa / ga) * ha;
                }
                clt = one;
                slt = ht / gt;
                srt = one;
                crt = ft / gt;
            }
        }
        if (gasmal) {
            /* Normal case */
            d = fa - ha;
            if (d == fa) {
                /* Copes with infinite F or H */
                l = one;
            } else {
                l = d / fa;
            }
            /* Note that 0 .le. L .le. 1 */
            m = gt / ft;
            /* Note that abs(M) .le. 1/macheps */
            t = two - l;
            /* Note that T .ge. 1 */
            mm = m * m;
            tt = t * t;
            s = sqrt(tt + mm);
            /* Note that 1 .le. S .le. 1 + 1/macheps */
            if (l == zero) {
                r = fabs(m);
            } else {
                r = sqrt(l * l + mm);
            }
            /* Note that 0 .le. R .le. 1 + 1/macheps */
            a = half * (s + r);
            /* Note that 1 .le. A .le. 1 + abs(M) */
            *ssmin = ha / a;
            *ssmax = fa * a;
            if (mm == zero) {
                /* Note that M is very tiny */
                if (l == zero) {
                    t = copysign(two, ft) * copysign(one, gt);
                } else {
                    t = gt / copysign(d, ft) + m / t;
                }
            } else {
                t = (m / (s + t) + m / (r + l)) * (one + a);
            }
            l = sqrt(t * t + four);
            crt = two / l;
            srt = t / l;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    if (swap) {
        *csl = srt;
        *snl = crt;
        *csr = slt;
        *snr = clt;
    } else {
        *csl = clt;
        *snl = slt;
        *csr = crt;
        *snr = srt;
    }
    /* Correct signs of SSMAX and SSMIN */
    if (pmax == 1)
        tsign = copysign(one, *csr) * copysign(one, *csl) * copysign(one, f);
    if (pmax == 2)
        tsign = copysign(one, *snr) * copysign(one, *csl) * copysign(one, g);
    if (pmax == 3)
        tsign = copysign(one, *snr) * copysign(one, *snl) * copysign(one, h);
    *ssmax = copysign(*ssmax, tsign);
    *ssmin = copysign(*ssmin, tsign * copysign(one, f) * copysign(one, h));
}

/* ------------------------------------------------------------------ */
/* DLASR (LAPACK master 2026): applies a sequence of plane rotations   */
/* to a real matrix A, from either the left or the right.              */
/* (Argument validation / XERBLA dropped.)                             */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlasr(char side, char pivot, char direct,
                                    int m, int n, const double* c,
                                    const double* s, double* a, int lda)
{
    const double one = 1.0, zero = 0.0;
    int i, j;
    double ctemp, stemp, temp;

    /* Quick return if possible */
    if ((m == 0) || (n == 0)) return;

    if (pyclap_lsame(side, 'L')) {
        /* Form  P * A */
        if (pyclap_lsame(pivot, 'V')) {
            if (pyclap_lsame(direct, 'F')) {
                for (j = 1; j <= m - 1; j++) {                 /* loop 20 */
                    ctemp = c[j-1];
                    stemp = s[j-1];
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= n; i++) {             /* loop 10 */
                            temp = a[j + (i-1)*lda];                   /* A(J+1,I) */
                            a[j + (i-1)*lda] = ctemp * temp - stemp * a[(j-1) + (i-1)*lda];
                            a[(j-1) + (i-1)*lda] = stemp * temp + ctemp * a[(j-1) + (i-1)*lda];
                        }
                    }
                }
            } else if (pyclap_lsame(direct, 'B')) {
                for (j = m - 1; j >= 1; j--) {                 /* loop 40 */
                    ctemp = c[j-1];
                    stemp = s[j-1];
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= n; i++) {             /* loop 30 */
                            temp = a[j + (i-1)*lda];                   /* A(J+1,I) */
                            a[j + (i-1)*lda] = ctemp * temp - stemp * a[(j-1) + (i-1)*lda];
                            a[(j-1) + (i-1)*lda] = stemp * temp + ctemp * a[(j-1) + (i-1)*lda];
                        }
                    }
                }
            }
        } else if (pyclap_lsame(pivot, 'T')) {
            if (pyclap_lsame(direct, 'F')) {
                for (j = 2; j <= m; j++) {                     /* loop 60 */
                    ctemp = c[j-2];                                    /* C(J-1) */
                    stemp = s[j-2];                                    /* S(J-1) */
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= n; i++) {             /* loop 50 */
                            temp = a[(j-1) + (i-1)*lda];               /* A(J,I) */
                            a[(j-1) + (i-1)*lda] = ctemp * temp - stemp * a[0 + (i-1)*lda];
                            a[0 + (i-1)*lda] = stemp * temp + ctemp * a[0 + (i-1)*lda];
                        }
                    }
                }
            } else if (pyclap_lsame(direct, 'B')) {
                for (j = m; j >= 2; j--) {                     /* loop 80 */
                    ctemp = c[j-2];                                    /* C(J-1) */
                    stemp = s[j-2];                                    /* S(J-1) */
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= n; i++) {             /* loop 70 */
                            temp = a[(j-1) + (i-1)*lda];               /* A(J,I) */
                            a[(j-1) + (i-1)*lda] = ctemp * temp - stemp * a[0 + (i-1)*lda];
                            a[0 + (i-1)*lda] = stemp * temp + ctemp * a[0 + (i-1)*lda];
                        }
                    }
                }
            }
        } else if (pyclap_lsame(pivot, 'B')) {
            if (pyclap_lsame(direct, 'F')) {
                for (j = 1; j <= m - 1; j++) {                 /* loop 100 */
                    ctemp = c[j-1];
                    stemp = s[j-1];
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= n; i++) {             /* loop 90 */
                            temp = a[(j-1) + (i-1)*lda];               /* A(J,I) */
                            a[(j-1) + (i-1)*lda] = stemp * a[(m-1) + (i-1)*lda] + ctemp * temp;
                            a[(m-1) + (i-1)*lda] = ctemp * a[(m-1) + (i-1)*lda] - stemp * temp;
                        }
                    }
                }
            } else if (pyclap_lsame(direct, 'B')) {
                for (j = m - 1; j >= 1; j--) {                 /* loop 120 */
                    ctemp = c[j-1];
                    stemp = s[j-1];
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= n; i++) {             /* loop 110 */
                            temp = a[(j-1) + (i-1)*lda];               /* A(J,I) */
                            a[(j-1) + (i-1)*lda] = stemp * a[(m-1) + (i-1)*lda] + ctemp * temp;
                            a[(m-1) + (i-1)*lda] = ctemp * a[(m-1) + (i-1)*lda] - stemp * temp;
                        }
                    }
                }
            }
        }
    } else if (pyclap_lsame(side, 'R')) {
        /* Form A * P**T */
        if (pyclap_lsame(pivot, 'V')) {
            if (pyclap_lsame(direct, 'F')) {
                for (j = 1; j <= n - 1; j++) {                 /* loop 140 */
                    ctemp = c[j-1];
                    stemp = s[j-1];
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= m; i++) {             /* loop 130 */
                            temp = a[(i-1) + j*lda];                   /* A(I,J+1) */
                            a[(i-1) + j*lda] = ctemp * temp - stemp * a[(i-1) + (j-1)*lda];
                            a[(i-1) + (j-1)*lda] = stemp * temp + ctemp * a[(i-1) + (j-1)*lda];
                        }
                    }
                }
            } else if (pyclap_lsame(direct, 'B')) {
                for (j = n - 1; j >= 1; j--) {                 /* loop 160 */
                    ctemp = c[j-1];
                    stemp = s[j-1];
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= m; i++) {             /* loop 150 */
                            temp = a[(i-1) + j*lda];                   /* A(I,J+1) */
                            a[(i-1) + j*lda] = ctemp * temp - stemp * a[(i-1) + (j-1)*lda];
                            a[(i-1) + (j-1)*lda] = stemp * temp + ctemp * a[(i-1) + (j-1)*lda];
                        }
                    }
                }
            }
        } else if (pyclap_lsame(pivot, 'T')) {
            if (pyclap_lsame(direct, 'F')) {
                for (j = 2; j <= n; j++) {                     /* loop 180 */
                    ctemp = c[j-2];                                    /* C(J-1) */
                    stemp = s[j-2];                                    /* S(J-1) */
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= m; i++) {             /* loop 170 */
                            temp = a[(i-1) + (j-1)*lda];               /* A(I,J) */
                            a[(i-1) + (j-1)*lda] = ctemp * temp - stemp * a[(i-1) + 0*lda];
                            a[(i-1) + 0*lda] = stemp * temp + ctemp * a[(i-1) + 0*lda];
                        }
                    }
                }
            } else if (pyclap_lsame(direct, 'B')) {
                for (j = n; j >= 2; j--) {                     /* loop 200 */
                    ctemp = c[j-2];                                    /* C(J-1) */
                    stemp = s[j-2];                                    /* S(J-1) */
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= m; i++) {             /* loop 190 */
                            temp = a[(i-1) + (j-1)*lda];               /* A(I,J) */
                            a[(i-1) + (j-1)*lda] = ctemp * temp - stemp * a[(i-1) + 0*lda];
                            a[(i-1) + 0*lda] = stemp * temp + ctemp * a[(i-1) + 0*lda];
                        }
                    }
                }
            }
        } else if (pyclap_lsame(pivot, 'B')) {
            if (pyclap_lsame(direct, 'F')) {
                for (j = 1; j <= n - 1; j++) {                 /* loop 220 */
                    ctemp = c[j-1];
                    stemp = s[j-1];
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= m; i++) {             /* loop 210 */
                            temp = a[(i-1) + (j-1)*lda];               /* A(I,J) */
                            a[(i-1) + (j-1)*lda] = stemp * a[(i-1) + (n-1)*lda] + ctemp * temp;
                            a[(i-1) + (n-1)*lda] = ctemp * a[(i-1) + (n-1)*lda] - stemp * temp;
                        }
                    }
                }
            } else if (pyclap_lsame(direct, 'B')) {
                for (j = n - 1; j >= 1; j--) {                 /* loop 240 */
                    ctemp = c[j-1];
                    stemp = s[j-1];
                    if ((ctemp != one) || (stemp != zero)) {
                        for (i = 1; i <= m; i++) {             /* loop 230 */
                            temp = a[(i-1) + (j-1)*lda];               /* A(I,J) */
                            a[(i-1) + (j-1)*lda] = stemp * a[(i-1) + (n-1)*lda] + ctemp * temp;
                            a[(i-1) + (n-1)*lda] = ctemp * a[(i-1) + (n-1)*lda] - stemp * temp;
                        }
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* ILADLR (LAPACK master 2026): scans A for its last non-zero row.     */
/* Returns the Fortran 1-based row index (0 if all zero).              */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_iladlr(int m, int n, const double* a, int lda)
{
    const double zero = 0.0;
    int i, j, ret;
    /* Quick test for the common case where one corner is non-zero. */
    if (m == 0) {
        ret = m;
    } else if (a[(m-1) + 0*lda] != zero || a[(m-1) + (n-1)*lda] != zero) {
        ret = m;
    } else {
        /* Scan up each column tracking the last zero row seen. */
        ret = 0;
        for (j = 1; j <= n; j++) {
            i = m;
            /* Fortran: DO WHILE((A(MAX(I,1),J).EQ.ZERO).AND.(I.GE.1)) */
            while ((a[(pyclap_imax(i, 1) - 1) + (j-1)*lda] == zero) &&
                   (i >= 1)) {
                i = i - 1;
            }
            ret = pyclap_imax(ret, i);
        }
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/* ILADLC (LAPACK master 2026): scans A for its last non-zero column.  */
/* Returns the Fortran 1-based column index (0 if all zero).           */
/* ------------------------------------------------------------------ */
__device__ static int pyclap_iladlc(int m, int n, const double* a, int lda)
{
    const double zero = 0.0;
    int i, ret;
    /* Quick test for the common case where one corner is non-zero. */
    if (n == 0) {
        ret = n;
    } else if (a[0 + (n-1)*lda] != zero || a[(m-1) + (n-1)*lda] != zero) {
        ret = n;
    } else {
        /* Now scan each column from the end, returning with the first
           non-zero. */
        for (ret = n; ret >= 1; ret--) {
            for (i = 1; i <= m; i++) {
                if (a[(i-1) + (ret-1)*lda] != zero) return ret;
            }
        }
        /* Fortran DO index after full loop completion: ret == 0 here. */
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/* DLARFG (LAPACK master 2026): generates an elementary reflector      */
/* (Householder matrix) H such that H * (alpha; x) = (beta; 0).        */
/* ALPHA and TAU are in/out scalars, passed by pointer.                */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlarfg(int n, double* alpha, double* x,
                                     int incx, double* tau)
{
    const double one = 1.0, zero = 0.0;
    int j, knt;
    double beta, rsafmn, safmin, xnorm;

    if (n <= 1) {
        *tau = zero;
        return;
    }

    xnorm = pyclap_dnrm2(n - 1, x, incx);

    if (xnorm == zero) {
        /* H  =  I */
        *tau = zero;
    } else {
        /* general case */
        beta = -copysign(pyclap_dlapy2(*alpha, xnorm), *alpha);
        safmin = pyclap_dlamch('S') / pyclap_dlamch('E');
        knt = 0;
        if (fabs(beta) < safmin) {
            /* XNORM, BETA may be inaccurate; scale X and recompute them */
            rsafmn = one / safmin;
            do {    /* Fortran label 10: rescaling loop */
                knt = knt + 1;
                pyclap_dscal(n - 1, rsafmn, x, incx);
                beta = beta * rsafmn;
                *alpha = *alpha * rsafmn;
            } while ((fabs(beta) < safmin) && (knt < 20));
            /* New BETA is at most 1, at least SAFMIN */
            xnorm = pyclap_dnrm2(n - 1, x, incx);
            beta = -copysign(pyclap_dlapy2(*alpha, xnorm), *alpha);
        }
        *tau = (beta - *alpha) / beta;
        pyclap_dscal(n - 1, one / (*alpha - beta), x, incx);
        /* If ALPHA is subnormal, it may lose relative accuracy */
        for (j = 1; j <= knt; j++) {   /* loop 20 */
            beta = beta * safmin;
        }
        *alpha = beta;
    }
}

/* ------------------------------------------------------------------ */
/* DLARF (LAPACK master 2026): applies an elementary reflector H to an */
/* M-by-N matrix C, from either the left or the right                  */
/* (H = I - tau * v * v**T). This fetched version does the             */
/* ILADLR/ILADLC trailing-zero scans inline (it does not delegate to   */
/* dlarf1f). WORK: dimension N (SIDE='L') or M (SIDE='R').             */
/* ------------------------------------------------------------------ */
__device__ static void pyclap_dlarf(char side, int m, int n,
                                    const double* v, int incv, double tau,
                                    double* c, int ldc, double* work)
{
    const double one = 1.0, zero = 0.0;
    int applyleft;
    int i, lastv, lastc;

    i = 1;   /* referenced only when tau != 0, where it is always set */
    applyleft = pyclap_lsame(side, 'L');
    lastv = 0;
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
        /* Look for the last non-zero row in V. */
        while (lastv > 0 && v[i-1] == zero) {
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
           For INCV > 0, set I = 1 */
        if (incv > 0) {
            i = 1;
        }
    }
    /* Note that lastc.eq.0 renders the BLAS operations null; no special
       case is needed at this level. */
    if (applyleft) {
        /* Form  H * C */
        if (lastv > 0) {
            /* w(1:lastc,1) := C(1:lastv,1:lastc)**T * v(1:lastv,1) */
            pyclap_dgemv('T', lastv, lastc, one, c, ldc, &v[i-1], incv,
                         zero, work, 1);
            /* C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**T */
            pyclap_dger(lastv, lastc, -tau, &v[i-1], incv, work, 1, c, ldc);
        }
    } else {
        /* Form  C * H */
        if (lastv > 0) {
            /* w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1) */
            pyclap_dgemv('N', lastc, lastv, one, c, ldc, &v[i-1], incv,
                         zero, work, 1);
            /* C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**T */
            pyclap_dger(lastc, lastv, -tau, work, 1, &v[i-1], incv, c, ldc);
        }
    }
}

#endif /* PYCLAP_LAPACK_LEAF_H */
