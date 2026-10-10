// Metal Shading Language prelude for the pycalphad Metal backend.
//
// Apple GPUs have no FP64 hardware, and the solver needs ~1e-14 relative
// precision (1e-16 least-squares cutoffs, 1e-10 convergence gates, J/mol
// energy differences on ~1e5 J/mol totals) over double's exponent range (the
// generated magnetic terms carry T^25 ~ 1e75 intermediates). Every `double`
// in the generated source therefore becomes `pyc_real`:
//
//     value = (hi + lo) * 2^e        hi, lo: float, e: int
//
// i.e. a double-float pair (Dekker/Knuth error-free arithmetic, ~48-bit
// significand) with an extra exponent. Values in [2^-62, 2^62) -- nearly all
// of them -- are kept with e == 0 and take a fast path that costs a couple
// of compares over plain double-float; only values outside that window pay
// for renormalization.
//
// GPU buffers keep the host's IEEE float64 layout (`pyc_f64`, raw bits) and
// convert on load/store, so buffer layouts and strides match the CUDA and
// C++ backends exactly.
//
// The Python lowering pass (metal_backend.lower_to_msl) rewrites the source
// before it reaches this prelude: floating literals become exact
// PYC_C(hi, lo, e) triples, pointers and references gain the `thread`
// address space, and `long long` becomes `long`. Compile with fast math OFF:
// the error terms below are exactly what fast math would "simplify" away.
#include <metal_stdlib>
using namespace metal;

#define PYCGPU_METAL 1

// Out-of-line helper: keeps rarely-taken or bulky paths from being inlined
// into every arithmetic site of the (very large) generated kernels.
#define PYC_NOINLINE __attribute__((noinline))

// ---------------------------------------------------------------------------
// double-float kernels on float2 (x = hi, y = lo)
// ---------------------------------------------------------------------------
static inline float2 pyc_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}
static inline float2 pyc_quick_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}
static inline float2 pyc_two_prod(float a, float b) {
    float p = a * b;
    return float2(p, fma(a, b, -p));
}
static inline float2 pyc_dd_add(float2 a, float2 b) {
    float2 s = pyc_two_sum(a.x, b.x);
    if (!isfinite(s.x)) return float2(s.x, 0.0f);
    float2 t = pyc_two_sum(a.y, b.y);
    s.y += t.x;
    s = pyc_quick_two_sum(s.x, s.y);
    s.y += t.y;
    return pyc_quick_two_sum(s.x, s.y);
}
static inline float2 pyc_dd_mul(float2 a, float2 b) {
    float2 p = pyc_two_prod(a.x, b.x);
    if (!isfinite(p.x)) return float2(p.x, 0.0f);
    p.y = fma(a.x, b.y, p.y);
    p.y = fma(a.y, b.x, p.y);
    return pyc_quick_two_sum(p.x, p.y);
}
static inline float2 pyc_dd_div(float2 a, float2 b) {
    float q1 = a.x / b.x;
    if (!isfinite(q1) || b.x == 0.0f) return float2(q1, 0.0f);
    float2 r = pyc_dd_add(a, -pyc_dd_mul(b, float2(q1, 0.0f)));
    float q2 = r.x / b.x;
    r = pyc_dd_add(r, -pyc_dd_mul(b, float2(q2, 0.0f)));
    float q3 = r.x / b.x;
    return pyc_dd_add(pyc_quick_two_sum(q1, q2), float2(q3, 0.0f));
}
static inline float2 pyc_dd_ldexp(float2 a, int k) { return float2(ldexp(a.x, k), ldexp(a.y, k)); }

// ---------------------------------------------------------------------------
// pyc_real
// ---------------------------------------------------------------------------
#define PYC_BIG   4.611686018427387904e18f   // 2^62
#define PYC_SMALL 2.168404344971008868e-19f  // 2^-62
#define PYC_MBIG  2147483648.0f              // 2^31: fast-path mul/div bound
#define PYC_MSMALL 4.656612873077393e-10f    // 2^-31

struct pyc_real;
static inline pyc_real pyc_make(float2 v, int e);

struct pyc_real {
    float hi;
    float lo;
    int e;
    pyc_real() = default;
    constexpr pyc_real(float h, float l, int ex) : hi(h), lo(l), e(ex) {}
    constexpr pyc_real(float h) : hi(h), lo(0.0f), e(0) {}
    constexpr pyc_real(int v) : hi((float)v), lo((float)(v - (int)(float)v)), e(0) {}
    constexpr pyc_real(uint v) : hi((float)v), lo((float)((int)(v - (uint)(float)v))), e(0) {}
    constexpr pyc_real(long v) : hi((float)v), lo((float)(v - (long)(float)v)), e(0) {}
    constexpr pyc_real(bool v) : hi(v ? 1.0f : 0.0f), lo(0.0f), e(0) {}
    float2 dd() const thread { return float2(hi, lo); }
    // Exponent-free double-float value (may overflow/underflow; for
    // conversions and comparisons only).
    float2 flat() const thread { return e == 0 ? float2(hi, lo) : pyc_dd_ldexp(float2(hi, lo), e); }
    explicit operator float() const thread { return e == 0 ? hi + lo : ldexp(hi + lo, e); }
    explicit operator int() const thread {
        if (e != 0) return e > 0 ? (hi > 0 ? INT_MAX : INT_MIN) : 0;
        // truncation toward zero of hi + lo; a non-integral hi cannot be
        // pushed across an integer by lo (|lo| <= ulp(hi)/2)
        if (trunc(hi) != hi) return (int)hi;
        return hi > 0.0f ? (int)hi + (int)floor(lo) : (int)hi + (int)ceil(lo);
    }
    explicit operator long() const thread { return (long)(int)(*this); }
    explicit operator uint() const thread { return (uint)(int)(*this); }
    explicit operator bool() const thread { return hi != 0.0f; }
};

#define PYC_C(h, l, ex) pyc_real((h), (l), (ex))

// Canonical form: e == 0 when the value lies in [2^-62, 2^62) (or is zero /
// non-finite), else hi in [1, 2) and e the binary exponent.
PYC_NOINLINE static pyc_real pyc_make_slow(float2 v, int e);
static inline pyc_real pyc_make(float2 v, int e) {
    float a = fabs(v.x);
    if (e == 0 && ((a < PYC_BIG && a >= PYC_SMALL) || a == 0.0f)) return pyc_real(v.x, v.y, 0);
    return pyc_make_slow(v, e);
}
PYC_NOINLINE static pyc_real pyc_make_slow(float2 v, int e) {
    float a = fabs(v.x);
    if (a == 0.0f) return pyc_real(0.0f, 0.0f, 0);
    if (!isfinite(v.x)) return pyc_real(v.x, 0.0f, 0);
    int k;
    frexp(v.x, k);
    k -= 1;                  // v.x = m * 2^k with |m| in [1, 2)
    int et = e + k;
    if (et < 62 && et >= -62) {
        float2 w = pyc_dd_ldexp(v, e);
        return pyc_real(w.x, w.y, 0);
    }
    float2 w = pyc_dd_ldexp(v, -k);
    return pyc_real(w.x, w.y, et);
}

// Mantissa/exponent split: m in [1, 2) (as double-float), value = m * 2^ex.
static inline float2 pyc_split(pyc_real a, thread int& ex) {
    int k;
    frexp(a.hi, k);
    k -= 1;
    ex = a.e + k;
    return pyc_dd_ldexp(float2(a.hi, a.lo), -k);
}

static inline pyc_real operator-(pyc_real a) { return pyc_real(-a.hi, -a.lo, a.e); }
static inline pyc_real operator+(pyc_real a) { return a; }

PYC_NOINLINE static pyc_real pyc_add_slow(pyc_real a, pyc_real b);
static inline pyc_real operator+(pyc_real a, pyc_real b) {
    if (a.e == 0 && b.e == 0) {
        float2 r = pyc_dd_add(a.dd(), b.dd());
        float m = fabs(r.x);
        if ((m < PYC_BIG && m >= PYC_SMALL) || m == 0.0f || !isfinite(r.x)) return pyc_real(r.x, r.y, 0);
        return pyc_make_slow(r, 0);
    }
    return pyc_add_slow(a, b);
}
PYC_NOINLINE static pyc_real pyc_add_slow(pyc_real a, pyc_real b) {
    if (a.hi == 0.0f || !isfinite(b.hi)) return b;
    if (b.hi == 0.0f || !isfinite(a.hi)) return a;
    int ea, eb;
    frexp(a.hi, ea);
    frexp(b.hi, eb);
    ea += a.e;
    eb += b.e;
    int em = max(ea, eb);
    float2 x = pyc_dd_ldexp(a.dd(), a.e - em);
    float2 y = pyc_dd_ldexp(b.dd(), b.e - em);
    return pyc_make(pyc_dd_add(x, y), em);
}
static inline pyc_real operator-(pyc_real a, pyc_real b) { return a + (-b); }

PYC_NOINLINE static pyc_real pyc_mul_slow(pyc_real a, pyc_real b);
static inline pyc_real operator*(pyc_real a, pyc_real b) {
    float ma = fabs(a.hi), mb = fabs(b.hi);
    if (a.e == 0 && b.e == 0 && ma <= PYC_MBIG && mb <= PYC_MBIG && ma >= PYC_MSMALL && mb >= PYC_MSMALL) {
        float2 r = pyc_dd_mul(a.dd(), b.dd());
        return pyc_real(r.x, r.y, 0);
    }
    return pyc_mul_slow(a, b);
}
PYC_NOINLINE static pyc_real pyc_mul_slow(pyc_real a, pyc_real b) {
    float ma = fabs(a.hi), mb = fabs(b.hi);
    if (ma == 0.0f || mb == 0.0f || !isfinite(a.hi) || !isfinite(b.hi)) {
        float2 r = pyc_dd_mul(float2(a.hi, 0.0f), float2(b.hi, 0.0f));
        return pyc_real(r.x, 0.0f, 0);
    }
    int ea, eb;
    float2 x = pyc_split(a, ea);
    float2 y = pyc_split(b, eb);
    return pyc_make(pyc_dd_mul(x, y), ea + eb);
}

PYC_NOINLINE static pyc_real operator/(pyc_real a, pyc_real b) {
    float ma = fabs(a.hi), mb = fabs(b.hi);
    if (a.e == 0 && b.e == 0 && ma <= PYC_MBIG && mb <= PYC_MBIG && ma >= PYC_MSMALL && mb >= PYC_MSMALL) {
        float2 r = pyc_dd_div(a.dd(), b.dd());
        return pyc_real(r.x, r.y, 0);
    }
    if (ma == 0.0f || mb == 0.0f || !isfinite(a.hi) || !isfinite(b.hi)) {
        return pyc_real(a.hi / b.hi, 0.0f, 0);
    }
    int ea, eb;
    float2 x = pyc_split(a, ea);
    float2 y = pyc_split(b, eb);
    return pyc_make(pyc_dd_div(x, y), ea - eb);
}

// Mixed-type overloads keep literal-free integer arithmetic unambiguous.
#define PYC_MIXED(OP) \
    static inline pyc_real operator OP(pyc_real a, float b) { return a OP pyc_real(b); } \
    static inline pyc_real operator OP(float a, pyc_real b) { return pyc_real(a) OP b; } \
    static inline pyc_real operator OP(pyc_real a, int b) { return a OP pyc_real(b); } \
    static inline pyc_real operator OP(int a, pyc_real b) { return pyc_real(a) OP b; } \
    static inline pyc_real operator OP(pyc_real a, long b) { return a OP pyc_real(b); } \
    static inline pyc_real operator OP(long a, pyc_real b) { return pyc_real(a) OP b; } \
    static inline pyc_real operator OP(pyc_real a, uint b) { return a OP pyc_real(b); } \
    static inline pyc_real operator OP(uint a, pyc_real b) { return pyc_real(a) OP b; }
PYC_MIXED(+)
PYC_MIXED(-)
PYC_MIXED(*)
PYC_MIXED(/)
#undef PYC_MIXED

#define PYC_COMPOUND(SPACE) \
    static inline SPACE pyc_real& operator+=(SPACE pyc_real& a, pyc_real b) { a = a + b; return a; } \
    static inline SPACE pyc_real& operator-=(SPACE pyc_real& a, pyc_real b) { a = a - b; return a; } \
    static inline SPACE pyc_real& operator*=(SPACE pyc_real& a, pyc_real b) { a = a * b; return a; } \
    static inline SPACE pyc_real& operator/=(SPACE pyc_real& a, pyc_real b) { a = a / b; return a; }
PYC_COMPOUND(thread)
#ifndef PYCGPU_MSL_ON_CPU  // address spaces collapse in the CPU debug build
PYC_COMPOUND(device)
#endif
#undef PYC_COMPOUND

// Comparisons are NaN-correct: anything compared with NaN is false.
PYC_NOINLINE static int pyc_cmp_slow(pyc_real a, pyc_real b) {
    // -1, 0, 1, or 2 for unordered
    pyc_real d = a - b;
    if (isnan(d.hi)) return 2;
    return d.hi < 0.0f ? -1 : (d.hi > 0.0f ? 1 : (d.lo < 0.0f ? -1 : (d.lo > 0.0f ? 1 : 0)));
}
static inline bool operator<(pyc_real a, pyc_real b) {
    if (a.e == 0 && b.e == 0) return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
    if (isinf(a.hi) || isinf(b.hi)) return a.hi < b.hi;
    return pyc_cmp_slow(a, b) == -1;
}
static inline bool operator>(pyc_real a, pyc_real b) { return b < a; }
static inline bool operator<=(pyc_real a, pyc_real b) {
    if (a.e == 0 && b.e == 0) return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
    if (isinf(a.hi) || isinf(b.hi)) return a.hi <= b.hi;
    int c = pyc_cmp_slow(a, b);
    return c == -1 || c == 0;
}
static inline bool operator>=(pyc_real a, pyc_real b) { return b <= a; }
static inline bool operator==(pyc_real a, pyc_real b) { return a.hi == b.hi && a.lo == b.lo && (a.e == b.e || a.hi == 0.0f); }
static inline bool operator!=(pyc_real a, pyc_real b) { return !(a == b); }
#define PYC_CMP(OP) \
    static inline bool operator OP(pyc_real a, float b) { return a OP pyc_real(b); } \
    static inline bool operator OP(float a, pyc_real b) { return pyc_real(a) OP b; } \
    static inline bool operator OP(pyc_real a, int b) { return a OP pyc_real(b); } \
    static inline bool operator OP(int a, pyc_real b) { return pyc_real(a) OP b; } \
    static inline bool operator OP(pyc_real a, long b) { return a OP pyc_real(b); } \
    static inline bool operator OP(long a, pyc_real b) { return pyc_real(a) OP b; }
PYC_CMP(<)
PYC_CMP(>)
PYC_CMP(<=)
PYC_CMP(>=)
PYC_CMP(==)
PYC_CMP(!=)
#undef PYC_CMP
static inline bool operator!(pyc_real a) { return a.hi == 0.0f; }

// ---- classification -------------------------------------------------------
static inline bool isnan(pyc_real a) { return isnan(a.hi) || isnan(a.lo); }
static inline bool isinf(pyc_real a) { return isinf(a.hi); }
static inline bool isfinite(pyc_real a) { return isfinite(a.hi) && isfinite(a.lo); }
static inline bool signbit(pyc_real a) { return signbit(a.hi); }

// ---- elementary functions -------------------------------------------------
static inline pyc_real fabs(pyc_real a) { return (a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f)) ? -a : a; }
static inline pyc_real abs(pyc_real a) { return fabs(a); }
static inline pyc_real fmax(pyc_real a, pyc_real b) { return isnan(a) ? b : (isnan(b) ? a : (a > b ? a : b)); }
static inline pyc_real fmin(pyc_real a, pyc_real b) { return isnan(a) ? b : (isnan(b) ? a : (a < b ? a : b)); }
static inline pyc_real fmax(pyc_real a, float b) { return fmax(a, pyc_real(b)); }
static inline pyc_real fmax(float a, pyc_real b) { return fmax(pyc_real(a), b); }
static inline pyc_real fmin(pyc_real a, float b) { return fmin(a, pyc_real(b)); }
static inline pyc_real fmin(float a, pyc_real b) { return fmin(pyc_real(a), b); }
static inline pyc_real fmax(pyc_real a, int b) { return fmax(a, pyc_real(b)); }
static inline pyc_real fmin(pyc_real a, int b) { return fmin(a, pyc_real(b)); }
static inline pyc_real copysign(pyc_real a, pyc_real b) { return signbit(a.hi) != signbit(b.hi) ? -a : a; }

PYC_NOINLINE static pyc_real floor(pyc_real a) {
    if (!isfinite(a.hi) || a.e > 0) return a;               // |a| >= 2^62 is integral
    if (a.e < 0) return a.hi < 0.0f ? pyc_real(-1.0f) : pyc_real(0.0f);  // |a| < 2^-62
    float h = floor(a.hi);
    if (h == a.hi) {
        float2 s = pyc_quick_two_sum(h, floor(a.lo));
        return pyc_real(s.x, s.y, 0);
    }
    return pyc_real(h, 0.0f, 0);
}
static inline pyc_real ceil(pyc_real a) { return -floor(-a); }
static inline pyc_real trunc(pyc_real a) { return a.hi >= 0.0f ? floor(a) : ceil(a); }
static inline pyc_real round(pyc_real a) { return floor(a + pyc_real(0.5f)); }
static inline pyc_real fmod(pyc_real a, pyc_real b) { return a - b * trunc(a / b); }

static inline pyc_real ldexp(pyc_real a, int k) { return pyc_make(a.dd(), a.e + k); }

PYC_NOINLINE static pyc_real sqrt(pyc_real a) {
    if (a.hi <= 0.0f || !isfinite(a.hi)) return pyc_real(sqrt(a.hi), 0.0f, 0);
    int ex;
    float2 m = pyc_split(a, ex);        // a = m * 2^ex, m in [1, 2)
    if (ex & 1) { m = pyc_dd_ldexp(m, 1); ex -= 1; }
    float x = rsqrt(m.x);
    float ax = m.x * x;
    float2 ax2 = pyc_two_prod(ax, ax);
    float corr = pyc_dd_add(m, -ax2).x * (x * 0.5f);
    return pyc_make(pyc_two_sum(ax, corr), ex / 2);
}

// exp: x = k*ln2 + r with a three-term ln2 (exact k*C products), r scaled by 2^-8, Taylor series,
// squared back eight times; k lands in the exponent field.
constant float2 PYC_LN2 = float2(0.693147182464599609375f, -1.904654323148236017e-09f);
PYC_NOINLINE static float2 pyc_expm1_reduced(float2 r) {
    r = pyc_dd_ldexp(r, -8);
    float2 term = r;
    float2 sum = r;
    for (int i = 2; i <= 8; ++i) {
        term = pyc_dd_div(pyc_dd_mul(term, r), float2((float)i, 0.0f));
        sum = pyc_dd_add(sum, term);
        if (fabs(term.x) <= 1e-16f * fabs(sum.x)) break;
    }
    // (1 + s)^2 - 1 = 2s + s^2 keeps precision while squaring back
    for (int i = 0; i < 8; ++i) sum = pyc_dd_add(pyc_dd_ldexp(sum, 1), pyc_dd_mul(sum, sum));
    return sum;
}
PYC_NOINLINE static pyc_real exp(pyc_real a) {
    if (isnan(a.hi)) return a;
    if (a.e > 0 || (a.e == 0 && fabs(a.hi) > 1.0e9f)) return a.hi > 0.0f ? pyc_real(INFINITY) : pyc_real(0.0f);
    if (a.e < 0) return pyc_real(1.0f);  // |a| < 2^-62
    float2 x = a.dd();
    if (x.x == 0.0f) return pyc_real(1.0f);
    float k = round(x.x / PYC_LN2.x);
    float2 r = pyc_dd_add(x, -pyc_two_prod(k, 0.693145751953125f));
    r = pyc_dd_add(r, -pyc_two_prod(k, 1.428606765330187e-06f));
    r = pyc_dd_add(r, float2(-k * 5.497923148104107e-14f, 0.0f));
    float2 res = pyc_dd_add(pyc_expm1_reduced(r), float2(1.0f, 0.0f));
    return pyc_make(res, (int)k);
}

// log: a = m * 2^ex with m in [sqrt(1/2), sqrt(2)); the float estimate of
// log(m) is accurate to ~1e-8, so one Newton step on exp reaches full
// precision.
PYC_NOINLINE static pyc_real log(pyc_real a) {
    if (isnan(a.hi) || a.hi < 0.0f) return pyc_real(NAN);
    if (a.hi == 0.0f) return pyc_real(-INFINITY);
    if (isinf(a.hi)) return a;
    int ex;
    float2 m = pyc_split(a, ex);
    if (m.x > 1.41421356f) { m = pyc_dd_ldexp(m, -1); ex += 1; }
    float2 x = float2(log(m.x), 0.0f);
    // x + m*exp(-x) - 1, with exp(-x) of a small argument
    float2 e = pyc_dd_add(pyc_expm1_reduced(-x), float2(1.0f, 0.0f));
    x = pyc_dd_add(pyc_dd_add(x, pyc_dd_mul(m, e)), float2(-1.0f, 0.0f));
    float2 eln2 = pyc_two_prod((float)ex, PYC_LN2.x);
    x = pyc_dd_add(x, eln2);
    x = pyc_dd_add(x, float2((float)ex * PYC_LN2.y, 0.0f));
    return pyc_real(x.x, x.y, 0);
}
static inline pyc_real log10(pyc_real a) { return log(a) / pyc_real(2.302585124969482421875f, -3.1975250e-08f, 0); }
static inline pyc_real log2(pyc_real a) { return log(a) / pyc_real(PYC_LN2.x, PYC_LN2.y, 0); }

PYC_NOINLINE static pyc_real pyc_powi(pyc_real b, int n) {
    uint m = n < 0 ? (uint)(-n) : (uint)n;
    pyc_real r = pyc_real(1.0f), p = b;
    while (m) {
        if (m & 1u) r = r * p;
        m >>= 1u;
        if (m) p = p * p;
    }
    return n < 0 ? pyc_real(1.0f) / r : r;
}
PYC_NOINLINE static pyc_real pow(pyc_real b, pyc_real e) {
    if (e.e == 0 && fabs(e.hi) < 1024.0f) {
        pyc_real t = trunc(e);
        if (t == e) return pyc_powi(b, (int)e);
    }
    if (b.hi == 0.0f) return e.hi > 0.0f ? pyc_real(0.0f) : pyc_real(INFINITY);
    return exp(e * log(b));
}
static inline pyc_real pow(pyc_real b, int e) { return pyc_powi(b, e); }
static inline pyc_real pow(pyc_real b, float e) { return pow(b, pyc_real(e)); }
static inline pyc_real pow(int b, pyc_real e) { return pow(pyc_real(b), e); }
static inline pyc_real pow(float b, pyc_real e) { return pow(pyc_real(b), e); }

PYC_NOINLINE static pyc_real sinh(pyc_real a) { pyc_real x = exp(a); return (x - pyc_real(1.0f) / x) * pyc_real(0.5f); }
PYC_NOINLINE static pyc_real cosh(pyc_real a) { pyc_real x = exp(a); return (x + pyc_real(1.0f) / x) * pyc_real(0.5f); }
PYC_NOINLINE static pyc_real tanh(pyc_real a) {
    if (a > pyc_real(20.0f)) return pyc_real(1.0f);
    if (a < pyc_real(-20.0f)) return pyc_real(-1.0f);
    pyc_real x = exp(a * pyc_real(2.0f));
    return (x - pyc_real(1.0f)) / (x + pyc_real(1.0f));
}

// ---------------------------------------------------------------------------
// IEEE float64 storage in GPU buffers
// ---------------------------------------------------------------------------
PYC_NOINLINE static pyc_real pyc_from_bits(ulong bits) {
    uint hi32 = (uint)(bits >> 32);
    bool neg = (hi32 >> 31) != 0;
    int bexp = (int)((hi32 >> 20) & 0x7FFu);
    ulong mant = bits & 0x000FFFFFFFFFFFFFul;
    if (bexp == 0x7FF) {
        float v = mant ? NAN : INFINITY;
        return pyc_real(neg ? -v : v, 0.0f, 0);
    }
    if (bexp == 0) {
        if (mant == 0) return pyc_real(neg ? -0.0f : 0.0f, 0.0f, 0);
        // subnormal: normalize
        int lz = clz(mant) - 11;            // leading zeros within the 53-bit field
        mant <<= lz;
        bexp = 1 - lz;
    }
    mant |= 0x0010000000000000ul;           // 53-bit significand, value = mant * 2^(bexp-1075)
    // hi = top 24 bits, lo = next 24 bits (bits below are dropped: 2^-48 rel)
    float h = (float)(uint)(mant >> 29);            // exact (24 bits)
    float l = (float)(uint)((mant >> 5) & 0xFFFFFFu);  // exact (24 bits)
    float2 v = pyc_quick_two_sum(h, ldexp(l, -24));     // significand scaled to [2^23, 2^24)
    if (neg) v = -v;
    return pyc_make(pyc_dd_ldexp(v, -23), bexp - 1023);
}

PYC_NOINLINE static ulong pyc_to_bits(pyc_real a) {
    float h = a.hi, l = a.lo;
    if (isnan(h)) return 0x7FF8000000000000ul;
    if (isinf(h)) return h > 0 ? 0x7FF0000000000000ul : 0xFFF0000000000000ul;
    if (h == 0.0f) return signbit(h) ? 0x8000000000000000ul : 0ul;
    bool neg = h < 0.0f;
    if (neg) { h = -h; l = -l; }
    int k;
    frexp(h, k);                 // h = m * 2^k, m in [0.5, 1)
    // significand of h + l as a signed 64-bit integer scaled by 2^(60-k)
    long sig = (long)ldexp(h, 60 - k);                 // exact: h has 24 bits
    long lsig = (long)rint(ldexp(l, 60 - k));          // |l| <= ulp(h)/2: fits
    sig += lsig;
    int ex = a.e + k - 60;                             // value = sig * 2^ex
    // normalize to 53 bits with round-to-nearest-even
    int msb = 63 - clz((ulong)sig);
    int shift = msb - 52;
    ulong m;
    if (shift > 0) {
        ulong u = (ulong)sig;
        ulong rem = u & ((1ul << shift) - 1ul);
        ulong halfway = 1ul << (shift - 1);
        m = u >> shift;
        if (rem > halfway || (rem == halfway && (m & 1ul))) m += 1ul;
        if (m >> 53) { m >>= 1; shift += 1; }
    } else {
        m = ((ulong)sig) << (-shift);
    }
    ex += shift;                                       // value = m * 2^ex, m in [2^52, 2^53)
    int bexp = ex + 52 + 1023;
    ulong sign = neg ? 0x8000000000000000ul : 0ul;
    if (bexp >= 0x7FF) return sign | 0x7FF0000000000000ul;
    if (bexp <= 0) {
        int s = 1 - bexp;
        if (s > 53) return sign;
        m >>= s;                                       // subnormal (truncating)
        return sign | m;
    }
    return sign | ((ulong)bexp << 52) | (m & 0x000FFFFFFFFFFFFFul);
}

// A float64 slot in a GPU buffer. Reads convert to pyc_real, assignments
// convert back, so `device double*` code keeps working unchanged.
struct pyc_f64 {
    ulong bits;
    operator pyc_real() const device { return pyc_from_bits(bits); }
    explicit operator int() const device { return (int)pyc_from_bits(bits); }
    explicit operator long() const device { return (long)pyc_from_bits(bits); }
    explicit operator float() const device { return (float)pyc_from_bits(bits); }
    device pyc_f64& operator=(pyc_real v) device { bits = pyc_to_bits(v); return *this; }
#ifndef PYCGPU_MSL_ON_CPU
    operator pyc_real() const thread { return pyc_from_bits(bits); }
    operator pyc_real() const constant { return pyc_from_bits(bits); }
    explicit operator int() const thread { return (int)pyc_from_bits(bits); }
    thread pyc_f64& operator=(pyc_real v) thread { bits = pyc_to_bits(v); return *this; }
#endif
};

// ---------------------------------------------------------------------------
// host-library shims
// ---------------------------------------------------------------------------
// The generated source is written for CUDA/C++; Metal has no printf, libc
// memory routines, or `long long`.
#define printf(...) ((void)0)
#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define __restrict__
#ifndef assert
#define assert(x) ((void)0)
#endif

// CUDA thread-index builtins appear in the solver only for debug output
// gated on thread 0. Metal helper functions cannot see the thread index;
// these constants keep every thread off those debug paths.
struct pyc_dim3 { uint x, y, z; };
constant pyc_dim3 threadIdx = {1u, 0u, 0u};
constant pyc_dim3 blockIdx = {0u, 0u, 0u};
constant pyc_dim3 blockDim = {1u, 1u, 1u};
constant pyc_dim3 gridDim = {1u, 1u, 1u};

static inline void pyc_memset(thread void* dst, int v, ulong n) {
    thread uchar* d = (thread uchar*)dst;
    for (ulong i = 0; i < n; ++i) d[i] = (uchar)v;
}
static inline void pyc_memcpy(thread void* dst, thread const void* src, ulong n) {
    thread uchar* d = (thread uchar*)dst;
    thread const uchar* s = (thread const uchar*)src;
    for (ulong i = 0; i < n; ++i) d[i] = s[i];
}
#define memset(d, v, n) pyc_memset((thread void*)(d), (v), (ulong)(n))
#define memcpy(d, s, n) pyc_memcpy((thread void*)(d), (thread const void*)(s), (ulong)(n))

// Limits of pyc_real: double's range, ~2^-48 precision.
#define DBL_MAX PYC_C(1.99999988079071044921875f, 0.0f, 1023)
#define DBL_MIN PYC_C(1.0f, 0.0f, -1022)
#define DBL_EPSILON PYC_C(3.5527136788005009e-15f, 0.0f, 0)
#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif
#define HUGE_VAL INFINITY

#define double pyc_real
