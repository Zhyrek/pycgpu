#ifndef PHASE_REC_H
#define PHASE_REC_H

#ifdef PYCGPU_FP32EMU
// FP32-emulation prototype: round values to float precision at the outputs of
// the generated model functions and the linear-algebra solves, so the FP64
// kernel behaves as if those stages ran in single precision. Used to answer
// empirically whether an FP32 solver pass could produce usable warm starts.
__device__ inline double pycgpu_f32(double v) { return (double)(float)v; }
__device__ inline void pycgpu_f32_arr(double* a, int n) {
    for (int i = 0; i < n; ++i) a[i] = (double)(float)a[i];
}
#endif

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

#endif // PHASE_REC_H