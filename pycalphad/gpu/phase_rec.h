#ifndef PHASE_REC_H
#define PHASE_REC_H


#ifdef PYCGPU_METAL
/* Metal has no function pointers. Each callable member is a functor that
 * stores the model index and dispatches through generated switch functions
 * (pycgpu_dispatch_*, emitted after the model functions), so call sites such
 * as `phase_record->obj(x)` and `phase_record->formulafused != nullptr` read
 * exactly as they do with the CUDA/C++ function-pointer table. */
enum {
    PYCGPU_FN_OBJ = 0, PYCGPU_FN_FORMULAOBJ, PYCGPU_FN_FORMULAGRAD, PYCGPU_FN_FORMULAHESS,
    PYCGPU_FN_INTERNAL_CONS_FUNC, PYCGPU_FN_INTERNAL_CONS_JAC, PYCGPU_FN_MASS_OBJ,
    PYCGPU_FN_FORMULAMOLE_OBJ, PYCGPU_FN_FORMULAMOLE_GRAD, PYCGPU_FN_FORMULAPARAMGRAD,
    PYCGPU_FN_FORMULAPARAMMIXED, PYCGPU_FN_FORMULAMOLE_HESS
};
double pycgpu_dispatch_scalar(int fn, int model, const double* x);
void pycgpu_dispatch_array(int fn, int model, double* out, const double* x);
void pycgpu_dispatch_fused(int model, double* out_eg, double* out_hess, const double* x);

struct pycgpu_func_t {
    int fn;
    int model;  /* -1: not provided */
    double operator()(const double* x) const { return pycgpu_dispatch_scalar(fn, model, x); }
    explicit operator bool() const { return model >= 0; }
};
struct pycgpu_array_func_t {
    int fn;
    int model;
    void operator()(double* out, const double* x) const { pycgpu_dispatch_array(fn, model, out, x); }
    explicit operator bool() const { return model >= 0; }
};
struct pycgpu_fused_func_t {
    int model;
    void operator()(double* out_eg, double* out_hess, const double* x) const {
        pycgpu_dispatch_fused(model, out_eg, out_hess, x);
    }
    explicit operator bool() const { return model >= 0; }
};
static inline bool operator==(pycgpu_func_t f, decltype(nullptr)) { return f.model < 0; }
static inline bool operator!=(pycgpu_func_t f, decltype(nullptr)) { return f.model >= 0; }
static inline bool operator==(pycgpu_array_func_t f, decltype(nullptr)) { return f.model < 0; }
static inline bool operator!=(pycgpu_array_func_t f, decltype(nullptr)) { return f.model >= 0; }
static inline bool operator==(pycgpu_fused_func_t f, decltype(nullptr)) { return f.model < 0; }
static inline bool operator!=(pycgpu_fused_func_t f, decltype(nullptr)) { return f.model >= 0; }

typedef struct PhaseRecord {
    pycgpu_func_t obj;
    pycgpu_func_t formulaobj;
    pycgpu_array_func_t formulagrad;
    pycgpu_array_func_t formulahess;
    pycgpu_array_func_t internal_cons_func;
    pycgpu_array_func_t internal_cons_jac;
    pycgpu_array_func_t mass_obj;
    pycgpu_array_func_t formulamole_obj;
    pycgpu_array_func_t formulamole_grad;
    pycgpu_array_func_t formulamole_hess;
    pycgpu_array_func_t formulaparamgrad;
    pycgpu_array_func_t formulaparammixed;
    pycgpu_fused_func_t formulafused;
    int num_statevars;
    int phase_dof;
    int num_vars;
    int num_elements;
    int num_internal_cons;
    int nonvacant_elements;
    void init_model(int m, bool has_hess, bool has_fused, bool has_param_derivs,
                    int ns, int pd, int ne, int nic, int nve) {
        obj = pycgpu_func_t{PYCGPU_FN_OBJ, m};
        formulaobj = pycgpu_func_t{PYCGPU_FN_FORMULAOBJ, m};
        formulagrad = pycgpu_array_func_t{PYCGPU_FN_FORMULAGRAD, m};
        formulahess = pycgpu_array_func_t{PYCGPU_FN_FORMULAHESS, has_hess ? m : -1};
        internal_cons_func = pycgpu_array_func_t{PYCGPU_FN_INTERNAL_CONS_FUNC, m};
        internal_cons_jac = pycgpu_array_func_t{PYCGPU_FN_INTERNAL_CONS_JAC, m};
        mass_obj = pycgpu_array_func_t{PYCGPU_FN_MASS_OBJ, m};
        formulamole_obj = pycgpu_array_func_t{PYCGPU_FN_FORMULAMOLE_OBJ, m};
        formulamole_grad = pycgpu_array_func_t{PYCGPU_FN_FORMULAMOLE_GRAD, m};
        formulamole_hess = pycgpu_array_func_t{PYCGPU_FN_FORMULAMOLE_HESS, -1};
        formulaparamgrad = pycgpu_array_func_t{PYCGPU_FN_FORMULAPARAMGRAD, has_param_derivs ? m : -1};
        formulaparammixed = pycgpu_array_func_t{PYCGPU_FN_FORMULAPARAMMIXED, has_param_derivs ? m : -1};
        formulafused = pycgpu_fused_func_t{has_fused ? m : -1};
        num_statevars = ns;
        phase_dof = pd;
        num_vars = ns + pd;
        num_elements = ne;
        num_internal_cons = nic;
        nonvacant_elements = nve > 0 ? nve : ne;
    }
} PhaseRecord;

#else  // CUDA / HIP / C++: function-pointer table

typedef double (*pycgpu_func_t)(const double*);
typedef void (*pycgpu_array_func_t)(double*, const double*);
/* Fused energy+gradient+Hessian evaluation: out_eg[0] = G (per formula
 * unit), out_eg[1..num_vars] = dG/dx_i, out_hess = row-major d2G/dx_i dx_j.
 * One shared CSE pool across all outputs — the values are bit-identical to
 * the separate formulaobj/formulagrad/formulahess functions (CSE only names
 * shared subtrees; it never reassociates arithmetic), but shared
 * subexpressions are computed once instead of three times. */
typedef void (*pycgpu_fused_func_t)(double*, double*, const double*);

typedef struct PhaseRecord {
    pycgpu_func_t obj;
    pycgpu_func_t formulaobj;
    pycgpu_array_func_t formulagrad;
    pycgpu_array_func_t formulahess;
    pycgpu_array_func_t internal_cons_func;
    pycgpu_array_func_t internal_cons_jac;
    pycgpu_array_func_t mass_obj;
    pycgpu_array_func_t formulamole_obj;
    pycgpu_array_func_t formulamole_grad;
    /* Site-fraction Hessians of the moles per formula unit,
     * out[(comp * phase_dof + i) * phase_dof + j]; set by an emitted
     * assignment after init() for phases whose moles are non-linear in the
     * site fractions (two-sublattice ionic liquids), nullptr otherwise. */
    pycgpu_array_func_t formulamole_hess;
    /* Jansson parameter derivatives (set by emitted assignments after init()
     * when fit parameters are present and requested; nullptr otherwise). */
    pycgpu_array_func_t formulaparamgrad;   /* out[p] = dG/dp            */
    pycgpu_array_func_t formulaparammixed;  /* out[j*MAX_PARAMS+p] = d2G/dy_j dp */
    /* Fused G+grad+hess (set by emitted assignment after init() when the
     * Hessian is generated; nullptr otherwise -> callers use the separate
     * functions). */
    pycgpu_fused_func_t formulafused;
    int num_statevars; //number of STATE variables
    int phase_dof; //number of SITE variables
    int num_vars;
    int num_elements;
    int num_internal_cons;
    int nonvacant_elements; //number of non-vacancy components
    __device__ void init(pycgpu_func_t on, pycgpu_func_t fon, pycgpu_array_func_t fgn, pycgpu_array_func_t fhn, pycgpu_array_func_t icfn, pycgpu_array_func_t icjn, pycgpu_array_func_t mon, pycgpu_array_func_t fmon, pycgpu_array_func_t fmgn, int ns, int pd, int ne, int nic, int nve = 0) {
        obj = on;
        formulaobj = fon;
        formulagrad = fgn;
        formulahess = fhn;
        internal_cons_func = icfn;
        internal_cons_jac = icjn;
        mass_obj = mon;
        formulamole_obj = fmon;
        formulamole_grad = fmgn;
        formulamole_hess = nullptr;
        formulaparamgrad = nullptr;
        formulaparammixed = nullptr;
        formulafused = nullptr;
        num_statevars = ns;
        phase_dof = pd;
        num_vars = ns+pd;
        num_elements = ne;
        num_internal_cons = nic;
        nonvacant_elements = nve > 0 ? nve : ne; // Default to num_elements if not specified
    }
    
    // Default initialization (not a constructor in CUDA)
    __device__ void reset() {
        obj = nullptr;
        formulaobj = nullptr;
        formulagrad = nullptr;
        formulahess = nullptr;
        internal_cons_func = nullptr;
        internal_cons_jac = nullptr;
        mass_obj = nullptr;
        formulamole_obj = nullptr;
        formulamole_grad = nullptr;
        formulamole_hess = nullptr;
        formulaparamgrad = nullptr;
        formulaparammixed = nullptr;
        formulafused = nullptr;
        num_statevars = 0;
        phase_dof = 0;
        num_vars = 0;
        num_elements = 0;
        num_internal_cons = 0;
        nonvacant_elements = 0;
    }
} PhaseRecord;

#endif // PYCGPU_METAL

#endif // PHASE_REC_H