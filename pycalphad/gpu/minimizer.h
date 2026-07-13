#pragma once
#if defined(__CUDACC_RTC__) || defined(__HIPCC_RTC__)
// NVRTC (runtime compilation, pip-only CUDA) has no host C headers; the
// handful of constants/declarations the kernels use are provided directly.
// Math functions, memcpy/memset and device printf are NVRTC builtins.
#ifndef PYCGPU_RTC_COMPAT
#define PYCGPU_RTC_COMPAT
#define DBL_MAX 1.7976931348623157e+308
#define DBL_MIN 2.2250738585072014e-308
#define DBL_EPSILON 2.2204460492503131e-16
#define FLT_MAX 3.402823466e+38f
#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif
extern "C" __device__ int printf(const char*, ...);
#endif // PYCGPU_RTC_COMPAT
#else
#include <math.h>       // For fabs, fmax, fmin, etc.
#include <float.h>      // For DBL_EPSILON if needed
#include <string.h>     // For memset
#endif // __CUDACC_RTC__
#include "phase_rec.h" // PhaseRecord definition
#include "comp_set.h" // CompositionSet definition
#include "debug_gpu.h" // GPU debug system

// Forward declarations for types defined in eqsolver.h
struct DeviceGrid;
struct DevicePhaseData;

// (legacy SVD forward declarations removed with svd.c)

// Forward declarations for functions defined later in this file
__device__ void compute_phase_matrix(double* phase_matrix_out, const double* hess_in,
                                    const double* cons_jac_tmp_in,
                                    const CompositionSet& compset_ref, int num_statevars_val,
                                    const double* phase_dof_site_fracs);
__device__ void lstsq(double* A, int nrows, int ncols, double* b, double tolerance,
                      double* U, double* V, double* singular_values, double* superdiag);

#ifndef MAX_PARAMS
#define MAX_PARAMS 0
#endif

// From constants.py
#define MIN_SITE_FRACTION 1e-14
#define MIN_PHASE_FRACTION 1e-6
#define COMP_DIFFERENCE_TOL 1e-4

#ifdef PYCGPU_PROF
// Per-thread cycle accumulators for kernel-time attribution (PYCGPU_PROF=1).
// Zeroed at run_loop entry; printed in run_loop's [PROF] line.
#define PYCGPU_PROF_MAXT 8192
__device__ long long g_prof_recompute[PYCGPU_PROF_MAXT];
__device__ long long g_prof_fill[PYCGPU_PROF_MAXT];
__device__ long long g_prof_lstsq[PYCGPU_PROF_MAXT];
__device__ long long g_prof_hess[PYCGPU_PROF_MAXT];   // formulahess evaluations
__device__ long long g_prof_inv[PYCGPU_PROF_MAXT];    // phase-matrix LU inversions
__device__ long long g_prof_funcs[PYCGPU_PROF_MAXT];  // other generated funcs (obj/grad/mole)
#endif

#ifndef MAX_EQ_SOLN_LEN
#define MAX_EQ_SOLN_LEN 50
#endif
#define INTERNAL_CONSTRAINT_SCALING 1.0
#define MAX_ENDMEMBER_PAIRS 5000
#define MAX_EXTRA_POINTS 90000

// Maximum expected sizes for static allocation
#ifndef MAX_COMPONENTS
#define MAX_COMPONENTS 32
#endif
#ifndef MAX_PHASES
#define MAX_PHASES 64
#endif
#ifndef MAX_STATEVARS
#define MAX_STATEVARS 8
#endif
#ifndef MAX_DOF_PER_PHASE
#define MAX_DOF_PER_PHASE 64
#endif
#ifndef MAX_INTERNAL_CONSTRAINTS
#define MAX_INTERNAL_CONSTRAINTS 32
#endif
// MAX_PHASE_LOCAL_CONDITIONS is effectively 0
#ifndef MAX_FIXED_MOLE_FRACTION_CONDITIONS
#define MAX_FIXED_MOLE_FRACTION_CONDITIONS MAX_COMPONENTS
#endif

// Forward declaration
struct SystemState;
// struct CompositionSet; // Already defined in comp_set.h
// struct PhaseRecord; // Already defined in phase_rec.h

// Define the matrix dimensions first
#define MAX_SVD_DIM (MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2)
#define MAX_SVD_M MAX_SVD_DIM
#define MAX_SVD_N MAX_SVD_DIM
#define MAX_PHASE_MATRIX_DIM (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)

// Maximum number of threads that can run simultaneously
// This should be at least as large as the maximum number of conditions
#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif

// Note: All work arrays for lstsq are now allocated by Python and passed as kernel parameters
// This allows for dynamic allocation based on the actual number of conditions
// Each thread gets its own slice of these arrays to avoid race conditions

// Corresponds to SystemSpecification in minimizer.pyx
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SystemSpecification {
    int num_statevars;
    int num_components;
    double prescribed_system_amount;
    double initial_chemical_potentials[MAX_COMPONENTS];
    double prescribed_mole_fraction_coefficients[MAX_FIXED_MOLE_FRACTION_CONDITIONS][MAX_COMPONENTS];
    double prescribed_mole_fraction_rhs[MAX_FIXED_MOLE_FRACTION_CONDITIONS];
    int num_prescribed_mole_fraction_conditions;
    int num_prescribed_mole_fraction_coefficients_cols;

    int free_chemical_potential_indices[MAX_COMPONENTS];
    int num_free_chemical_potentials;
    int free_statevar_indices[MAX_STATEVARS];
    int num_free_statevars;
    int fixed_chemical_potential_indices[MAX_COMPONENTS];
    int num_fixed_chemical_potentials;
    int fixed_statevar_indices[MAX_STATEVARS];
    int num_fixed_statevars;
    int fixed_stable_compset_indices[MAX_PHASES];
    int num_fixed_stable_compsets;
    int max_num_free_stable_phases;
    double ALLOWED_MASS_RESIDUAL;
    // Runtime fit-parameter values (copied into every compset's trailing dof
    // slots at initialization; per-condition storage enables per-walker
    // parameter vectors for batched MCMC ensembles).
    double fit_params[MAX_PARAMS + 1];
    int num_params;

    // Work arrays removed - now passed as parameters from Python to functions that need them
    // This allows for dynamic allocation based on actual number of conditions

    // Note: work_inv has been moved out of SystemSpecification to reduce stack usage.
    // It is now passed as a separate pointer (from global memory) through the call chain.
    // This fixes stack overflow on AMD/HIP GPUs when running multiple conditions.

    __device__ void init(int ns, int nc, double psa,
                         const double* icp, int num_icp,
                         const double* pmfc, int num_pmfc_rows, int num_pmfc_cols,
                         const double* pmfr, int num_pmfr,
                         const int* fcpi, int n_fcpi,
                         const int* fsvi, int n_fsvi,
                         const int* fix_cpi, int n_fix_cpi,
                         const int* fix_svi, int n_fix_svi,
                         const int* fsci, int n_fsci) {
        num_statevars = ns;
        num_components = nc;
        prescribed_system_amount = psa;
        num_prescribed_mole_fraction_conditions = num_pmfc_rows;
        num_prescribed_mole_fraction_coefficients_cols = num_pmfc_cols;

        for (int i = 0; i < num_icp; ++i) initial_chemical_potentials[i] = icp[i];
        for (int i = num_icp; i < MAX_COMPONENTS; ++i) initial_chemical_potentials[i] = 0.0;

        for (int i = 0; i < num_pmfc_rows; ++i) {
            for (int j = 0; j < num_pmfc_cols; ++j) {
                prescribed_mole_fraction_coefficients[i][j] = pmfc[i * num_pmfc_cols + j];
            }
            for (int j = num_pmfc_cols; j < MAX_COMPONENTS; ++j) {
                prescribed_mole_fraction_coefficients[i][j] = 0.0;
            }
        }
        for (int i = num_pmfc_rows; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) {
            for (int j = 0; j < MAX_COMPONENTS; ++j) {
                 prescribed_mole_fraction_coefficients[i][j] = 0.0;
            }
        }

        for (int i = 0; i < num_pmfr; ++i) prescribed_mole_fraction_rhs[i] = pmfr[i];
        for (int i = num_pmfr; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) prescribed_mole_fraction_rhs[i] = 0.0;

        num_free_chemical_potentials = n_fcpi;
        for (int i = 0; i < n_fcpi; ++i) free_chemical_potential_indices[i] = fcpi[i];
        num_free_statevars = n_fsvi;
        for (int i = 0; i < n_fsvi; ++i) free_statevar_indices[i] = fsvi[i];
        num_fixed_chemical_potentials = n_fix_cpi;
        for (int i = 0; i < n_fix_cpi; ++i) fixed_chemical_potential_indices[i] = fix_cpi[i];
        num_fixed_statevars = n_fix_svi;
        for (int i = 0; i < n_fix_svi; ++i) fixed_statevar_indices[i] = fix_svi[i];
        num_fixed_stable_compsets = n_fsci;
        for (int i = 0; i < n_fsci; ++i) fixed_stable_compset_indices[i] = fsci[i];

        max_num_free_stable_phases = num_components + num_free_statevars - num_fixed_stable_compsets;

        // Match CPU behavior exactly: always use 1e-8
        // The CPU code has complex logic but ultimately sets:
        // self.ALLOWED_MASS_RESIDUAL = 1e-8
        ALLOWED_MASS_RESIDUAL = 1e-8;
    }
    SystemSpecification(){}
} SystemSpecification;

#ifdef __cplusplus
}
#endif

typedef struct CompsetState {
    double x[MAX_STATEVARS + MAX_DOF_PER_PHASE];
    int x_length;
    double energy;
    double grad[MAX_STATEVARS + MAX_DOF_PER_PHASE];
    int grad_length;
    double hess[(MAX_STATEVARS + MAX_DOF_PER_PHASE) * (MAX_STATEVARS + MAX_DOF_PER_PHASE)];
    int hess_rows;
    int hess_cols;
    double masses[MAX_COMPONENTS];
    int masses_length;
    double mass_jac[MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)];
    int mass_jac_rows;
    int mass_jac_cols;
    // phase_matrix_dim = phase_dof + num_internal_cons (since num_phase_local_conditions is 0)
    double phase_matrix[(MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)];
    int phase_matrix_dim;
    double full_e_matrix[(MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)];
    int full_e_matrix_dim;

    double c_G[MAX_DOF_PER_PHASE];
    int c_G_length;
    double c_statevars[MAX_DOF_PER_PHASE * MAX_STATEVARS];
    int c_statevars_rows;
    int c_statevars_cols;
    double c_component[MAX_COMPONENTS * MAX_DOF_PER_PHASE];
    int c_component_rows;
    int c_component_cols;
    double delta_y[MAX_DOF_PER_PHASE];
    int delta_y_length;
    double moles_normalization;
    double internal_cons[MAX_INTERNAL_CONSTRAINTS];
    int internal_cons_length;
    double moles_normalization_grad[MAX_STATEVARS + MAX_DOF_PER_PHASE];
    int moles_normalization_grad_length;
    double cons_jac_tmp[MAX_INTERNAL_CONSTRAINTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)];
    int cons_jac_tmp_rows;
    int cons_jac_tmp_cols;
    // phase_local_jac_tmp is removed as num_phase_local_conditions is 0

    __device__ void init(const SystemSpecification* spec, const CompositionSet* compset) {
        // Assuming compset and compset->phase_record are valid pointers
        const PhaseRecord* pr = compset->phase_record; // Convenience pointer

        x_length = spec->num_statevars + pr->phase_dof;
        for(int i=0; i<x_length; ++i) x[i] = 0.0; // Will be set from compset->dof
        for(int i=x_length; i<MAX_STATEVARS + MAX_DOF_PER_PHASE; ++i) x[i] = 0.0;

        energy = 0.0;

        grad_length = spec->num_statevars + pr->phase_dof;
        for(int i=0; i<grad_length; ++i) grad[i] = 0.0;
        for(int i=grad_length; i<MAX_STATEVARS + MAX_DOF_PER_PHASE; ++i) grad[i] = 0.0;

        hess_rows = spec->num_statevars + pr->phase_dof;
        hess_cols = spec->num_statevars + pr->phase_dof;
        for(int i=0; i < hess_rows * hess_cols; ++i) hess[i] = 0.0;
        for(int i=hess_rows * hess_cols; i < (MAX_STATEVARS + MAX_DOF_PER_PHASE) * (MAX_STATEVARS + MAX_DOF_PER_PHASE); ++i) hess[i] = 0.0;

        masses_length = spec->num_components; // Should be pr->num_elements if PhaseRecord stores that
        for(int i=0; i<masses_length; ++i) masses[i] = 0.0;
        for(int i=masses_length; i<MAX_COMPONENTS; ++i) masses[i] = 0.0;

        mass_jac_rows = spec->num_components; // pr->num_elements
        mass_jac_cols = spec->num_statevars + pr->phase_dof;
        for(int i=0; i<mass_jac_rows * mass_jac_cols; ++i) mass_jac[i] = 0.0;
        for(int i=mass_jac_rows*mass_jac_cols; i < MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE); ++i) mass_jac[i] = 0.0;

        phase_matrix_dim = pr->phase_dof + pr->num_internal_cons; // num_phase_local_conditions is 0
        for(int i=0; i<phase_matrix_dim * phase_matrix_dim; ++i) phase_matrix[i] = 0.0;
        for(int i=phase_matrix_dim*phase_matrix_dim; i < (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS); ++i) phase_matrix[i] = 0.0;

        full_e_matrix_dim = phase_matrix_dim;
        for(int i=0; i<full_e_matrix_dim * full_e_matrix_dim; ++i) full_e_matrix[i] = 0.0;
        for(int i=full_e_matrix_dim*full_e_matrix_dim; i < (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS); ++i) full_e_matrix[i] = 0.0;

        c_G_length = pr->phase_dof;
        for(int i=0; i<c_G_length; ++i) c_G[i] = 0.0;
        for(int i=c_G_length; i<MAX_DOF_PER_PHASE; ++i) c_G[i] = 0.0;

        c_statevars_rows = pr->phase_dof;
        c_statevars_cols = spec->num_statevars;
        for(int i=0; i<c_statevars_rows*c_statevars_cols; ++i) c_statevars[i] = 0.0;
        for(int i=c_statevars_rows*c_statevars_cols; i < MAX_DOF_PER_PHASE * MAX_STATEVARS; ++i) c_statevars[i] = 0.0;

        c_component_rows = spec->num_components; // pr->num_elements
        c_component_cols = pr->phase_dof;
        for(int i=0; i<c_component_rows*c_component_cols; ++i) c_component[i] = 0.0;
        for(int i=c_component_rows*c_component_cols; i < MAX_COMPONENTS * MAX_DOF_PER_PHASE; ++i) c_component[i] = 0.0;

        delta_y_length = pr->phase_dof;
        for(int i=0; i<delta_y_length; ++i) delta_y[i] = 0.0;
        for(int i=delta_y_length; i<MAX_DOF_PER_PHASE; ++i) delta_y[i] = 0.0;

        moles_normalization = 0.0;

        internal_cons_length = pr->num_internal_cons;
        for(int i=0; i<internal_cons_length; ++i) internal_cons[i] = 0.0;
        for(int i=internal_cons_length; i<MAX_INTERNAL_CONSTRAINTS; ++i) internal_cons[i] = 0.0;

        moles_normalization_grad_length = mass_jac_cols; // Same as mass_jac columns since both are in workspace format
        for(int i=0; i<moles_normalization_grad_length; ++i) moles_normalization_grad[i] = 0.0;
        for(int i=moles_normalization_grad_length; i<MAX_STATEVARS + MAX_DOF_PER_PHASE; ++i) moles_normalization_grad[i] = 0.0;

        cons_jac_tmp_rows = pr->num_internal_cons;
        cons_jac_tmp_cols = spec->num_statevars + pr->phase_dof;
        for(int i=0; i<cons_jac_tmp_rows * cons_jac_tmp_cols; ++i) cons_jac_tmp[i] = 0.0;
        for(int i=cons_jac_tmp_rows * cons_jac_tmp_cols; i<MAX_INTERNAL_CONSTRAINTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE); ++i) cons_jac_tmp[i] = 0.0;
    }
    __device__ CompsetState(){}
} CompsetState;

typedef struct SystemState {
    CompositionSet compsets[MAX_PHASES];
    int num_compsets;
    CompsetState cs_states[MAX_PHASES];

    int iteration;
    int iterations_since_last_phase_change;
    int metastable_phase_iterations[MAX_PHASES];
    int times_compset_removed[MAX_PHASES];
    double mass_residual;
    double phase_amt[MAX_PHASES];
    double chemical_potentials[MAX_COMPONENTS];
    int condition_idx;  // Thread/condition index for debug output
    double previous_chemical_potentials[MAX_COMPONENTS];
    double largest_chemical_potential_difference;
    double* delta_ms;  // Now points to global memory instead of stack allocation
    int delta_ms_rows;
    int delta_ms_cols;
    double delta_statevars[MAX_STATEVARS];
    double* phase_compositions;  // Now points to global memory instead of stack allocation
    int phase_compositions_rows;
    int phase_compositions_cols;

    int free_stable_compset_indices[MAX_PHASES];
    int num_free_stable_compsets;

    double largest_statevar_change;
    double largest_phase_amt_change;
    double largest_y_change;
    double system_amount;
    double mole_fractions[MAX_COMPONENTS];

    double _driving_forces_arr[MAX_PHASES];
    double _phase_energies_per_mole_atoms_arr[MAX_PHASES];
    double* _phase_amounts_per_mole_atoms_arr;  // Now points to global memory instead of stack allocation

    __device__ void init(SystemSpecification* spec, CompositionSet* initial_compsets, int initial_num_compsets, double* work_inv = nullptr) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: SystemState::init called with num_compsets=%d\n", initial_num_compsets);
        #endif
        num_compsets = initial_num_compsets;
        if (num_compsets > MAX_PHASES) num_compsets = MAX_PHASES; // Cap at MAX_PHASES

        for (int i = 0; i < num_compsets; ++i) {
            compsets[i] = initial_compsets[i]; // Simple assignment
            compsets[i].fixed = false;
            if (compsets[i].phase_record != nullptr) { // Ensure phase_record is valid
                 cs_states[i].init(spec, &compsets[i]);
            } else {
                // Handle error: phase_record is null. Use memset to avoid temporary on stack
                memset(&cs_states[i], 0, sizeof(CompsetState));
            }
        }
        for (int i = num_compsets; i < MAX_PHASES; ++i) {
             // Use memset to avoid creating temporaries on the stack
             memset(&cs_states[i], 0, sizeof(CompsetState));
             memset(&compsets[i], 0, sizeof(CompositionSet));
        }

        for (int i = 0; i < spec->num_fixed_stable_compsets; ++i) {
            int fixed_idx = spec->fixed_stable_compset_indices[i];
            if (fixed_idx < num_compsets) {
                compsets[fixed_idx].fixed = true;
            }
        }

        iteration = 0;
        iterations_since_last_phase_change = 0;
        for (int i = 0; i < MAX_PHASES; ++i) {
            metastable_phase_iterations[i] = 0;
            times_compset_removed[i] = 0;
            // Initialize phase_amt from NP, but we'll normalize below
            phase_amt[i] = (i < num_compsets) ? compsets[i].NP : 0.0;
            _driving_forces_arr[i] = 0.0;
            _phase_energies_per_mole_atoms_arr[i] = 0.0;
        }
        mass_residual = 1e10;

        for (int i = 0; i < MAX_COMPONENTS; ++i) {
            chemical_potentials[i] = 0.0;
            previous_chemical_potentials[i] = 0.0;
            mole_fractions[i] = 0.0;
        }
        for (int i = 0; i < spec->num_fixed_chemical_potentials; ++i) {
            int c_idx = spec->fixed_chemical_potential_indices[i];
            if (c_idx < spec->num_components) {
                chemical_potentials[c_idx] = spec->initial_chemical_potentials[c_idx];
            }
        }

        largest_chemical_potential_difference = -INFINITY;

        delta_ms_rows = num_compsets; // Initial value, might change if num_compsets changes
        delta_ms_cols = spec->num_components;
        for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) delta_ms[i] = 0.0;
        for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) _phase_amounts_per_mole_atoms_arr[i] = 0.0;

        for (int i = 0; i < MAX_STATEVARS; ++i) delta_statevars[i] = 0.0;

        phase_compositions_rows = num_compsets; // Initial value
        phase_compositions_cols = spec->num_components;
        for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) phase_compositions[i] = 0.0;
        
        // Calculate phase_compositions using formulamole_obj like CPU does
        // This is essential for phase amount normalization to work correctly
        
        
        double phase_comp_sum;
        for (int idx = 0; idx < num_compsets; ++idx) {
            CompositionSet* compset = &compsets[idx];
            if (compset->phase_record == nullptr) continue;
            
            // Calculate moles of each element per formula unit
            double formulamoles[MAX_COMPONENTS];
            // Initialize to zero since formulamole_obj only fills nonvacant elements
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                formulamoles[i] = 0.0;
            }
            
            // Check if formulamole_obj function pointer is valid
            if (compset->phase_record->formulamole_obj != nullptr) {
                // DEBUG: Check DOF array before calling formulamole_obj
                bool dof_valid = true;
                for (int i = 0; i < compset->phase_record->num_statevars + compset->phase_record->phase_dof; ++i) {
                    if (isnan(compset->dof[i]) || isinf(compset->dof[i])) {
                        dof_valid = false;
                        break;
                    }
                }
                if (!dof_valid) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU ERROR: Invalid DOF values in recompute for phase %d\\n", idx);
                    for (int i = 0; i < 5; ++i) {
                        printf("  dof[%d] = %f\\n", i, i < (compset->phase_record->num_statevars + compset->phase_record->phase_dof) ? compset->dof[i] : 0.0);
                    }
                    #endif
                } else {
                    // Create Model DOF array from Workspace DOF
                    // With updated energy functions, use full workspace DOF
                    if (compset->phase_record->formulamole_obj != nullptr) {
                        
        compset->phase_record->formulamole_obj(formulamoles, compset->dof);
                    }
                }
            }
            
            phase_comp_sum = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                phase_compositions[idx * MAX_COMPONENTS + comp_idx] = formulamoles[comp_idx];
        // DEBUG removed
            
                phase_comp_sum += formulamoles[comp_idx];
            }
            
            // Convert phase amounts to formula units like CPU does
            // CPU minimizer.pyx line 776: self.phase_amt[idx] /= phase_comp_sum
            // This normalization must happen in __init__ to match CPU behavior!
            
            if (phase_comp_sum > 1e-12) {
                // Normalize phase amount to formula units
                phase_amt[idx] /= phase_comp_sum;
            }
        }
        

        num_free_stable_compsets = 0;
        // Collecting free stable compsets
        for (int i = 0; i < num_compsets; ++i) {
            // Check compset[i]
            if (!compsets[i].fixed && compsets[i].NP > 0.0) { // CPU minimizer.pyx:792: NP > 0
                if (num_free_stable_compsets < MAX_PHASES) {
                    free_stable_compset_indices[num_free_stable_compsets++] = i;
                    // Added to free_stable set
                }
            }
        }
        // Final num_free_stable_compsets set

        largest_statevar_change = 0.0;
        largest_phase_amt_change = 0.0;
        largest_y_change = 0.0;
        system_amount = 0.0;

        recompute(spec, work_inv);
    }
    __device__ SystemState(){}

    __device__ void recompute(SystemSpecification* spec, double* work_inv) {
        // Get thread ID for debug messages
        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        
        // DEBUG: Check spec pointer validity at entry
        if (spec == nullptr) {
            #ifdef VERBOSE_DEBUG
            printf("GPU ERROR: spec pointer is NULL in recompute!\n");
            #endif
            return;
        }
        
        // DEBUG: Check if spec values look reasonable
        if (spec->num_statevars < 0 || spec->num_statevars > 10 || 
            spec->num_components < 0 || spec->num_components > 10) {
            #ifdef VERBOSE_DEBUG
            printf("GPU ERROR: spec appears corrupted at recompute entry!\n");
            printf("  spec=%p\n", spec);
            printf("  spec->num_statevars=%d (0x%X)\n", spec->num_statevars, spec->num_statevars);
            printf("  spec->num_components=%d (0x%X)\n", spec->num_components, spec->num_components);
            #endif
            // Don't return - try to continue
        }
        
        // Entered recompute function - removed pointer printf to avoid alignment issues
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: num_compsets=%d, iteration=%d\n", num_compsets, iteration);
        #endif
        // SEGMENT 22: STATE RECOMPUTE - GLOBAL QUANTITIES
        gpu_debug_log(22, "State recompute - global quantities", condition_idx);
        // Direct printf to test if this code is being reached
        #ifdef VERBOSE_DEBUG
        if (condition_idx < 3) {
            printf("[GPU] SEGMENT 22 DEBUG: iteration=%d, num_components=%d\n", iteration, spec->num_components);
            printf("[GPU]   chemical_potentials: [%.15e, %.15e]\n", chemical_potentials[0], chemical_potentials[1]);
            printf("[GPU]   system_amount: %.15e\n", system_amount);
            printf("[GPU]   mole_fractions: [");
            for (int comp_i = 0; comp_i < spec->num_components; comp_i++) {
                printf("%.6f", mole_fractions[comp_i]);
                if (comp_i < spec->num_components - 1) printf(", ");
            }
            printf("]\n");
            gpu_debug_log_array("chemical_potentials", chemical_potentials, spec->num_components);
            gpu_debug_log_value("system_amount", system_amount);
            gpu_debug_log_array("mole_fractions", mole_fractions, spec->num_components);
        }
        #endif
        
        // REMOVED: current_dof_for_phase array - now using model_dof_for_calcs created locally where needed

        for(int i=0; i < spec->num_components; ++i) mole_fractions[i] = 0.0;
        // Correct sizing for delta_ms for current num_compsets
        for(int i=0; i < num_compsets * spec->num_components; ++i) delta_ms[i] = 0.0;
        for(int i=num_compsets * spec->num_components; i < MAX_PHASES * MAX_COMPONENTS; ++i) delta_ms[i] = 0.0; // Zero out rest

        system_amount = 0.0;
        
        // DEBUG: Track total phase amounts through iterations
        double phase_amt_sum = 0.0;
        for (int idx = 0; idx < num_compsets; ++idx) {
            phase_amt_sum += phase_amt[idx];
        }
        #ifdef VERBOSE_DEBUG
        printf("[GPU MASS BALANCE] recompute() - iteration %d: sum(phase_amt) = %.15e\n", iteration, phase_amt_sum);
        #endif
        
        // REMOVED: Extraneous normalization not present in CPU code
        

        for (int idx = 0; idx < num_compsets; ++idx) {
            CompositionSet* compset = &compsets[idx];
            CompsetState* csst = &cs_states[idx];
            if (compset->phase_record == nullptr) continue; // Skip if phase_record is invalid

            // REMOVED: Old code that created current_dof_for_phase incorrectly
            // Now we create model_dof_for_calcs properly from workspace DOF when needed
            
            // Calculate phase_compositions using formulamole_obj (matching CPU minimizer.pyx)
            double formulamoles[MAX_COMPONENTS];
            // Initialize to zero since formulamole_obj only fills nonvacant elements
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                formulamoles[i] = 0.0;
            }
            
            // Check if formulamole_obj function pointer is valid
            if (compset->phase_record->formulamole_obj != nullptr) {
                // DEBUG: Check DOF array before calling formulamole_obj
                bool dof_valid = true;
                for (int i = 0; i < compset->phase_record->num_statevars + compset->phase_record->phase_dof; ++i) {
                    if (isnan(compset->dof[i]) || isinf(compset->dof[i])) {
                        dof_valid = false;
                        break;
                    }
                }
                if (!dof_valid) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU ERROR: Invalid DOF values in recompute for phase %d\\n", idx);
                    for (int i = 0; i < 5; ++i) {
                        printf("  dof[%d] = %f\\n", i, i < (compset->phase_record->num_statevars + compset->phase_record->phase_dof) ? compset->dof[i] : 0.0);
                    }
                    #endif
                } else {
                    // Create Model DOF array from Workspace DOF
                    // With updated energy functions, use full workspace DOF
                    if (compset->phase_record->formulamole_obj != nullptr) {
                        compset->phase_record->formulamole_obj(formulamoles, compset->dof);
                    }
                    
                    // Calculate mass jacobians (missing from original GPU implementation)
                    // This matches CPU line 820: compset.phase_record.formulamole_grad(csst.mass_jac[comp_idx, :], x, comp_idx)
                    // Zero out mass_jac first
                    for (int i = 0; i < csst->mass_jac_rows * csst->mass_jac_cols; ++i) {
                        csst->mass_jac[i] = 0.0;
                    }
                    
                    if (compset->phase_record->formulamole_grad != nullptr) {
                        // CSE formulamole_grad functions output reduced format
                        // Output format: [comp0_dT, comp0_dY1, comp0_dY2, ..., comp1_dT, comp1_dY1, comp1_dY2, ...]
                        double temp_mass_jac[MAX_COMPONENTS * (1 + MAX_DOF_PER_PHASE)];
                        for (int i = 0; i < MAX_COMPONENTS * (1 + MAX_DOF_PER_PHASE); ++i) {
                            temp_mass_jac[i] = 0.0;
                        }
                        
                        // Call formulamole_grad with workspace DOF
                        compset->phase_record->formulamole_grad(temp_mass_jac, compset->dof);
                        #ifdef PYCGPU_FP32EMU
                        pycgpu_f32_arr(temp_mass_jac, MAX_COMPONENTS * (1 + MAX_DOF_PER_PHASE));
                        #endif
                        
                        // Zero out the full mass_jac array
                        for (int i = 0; i < csst->mass_jac_rows * csst->mass_jac_cols; ++i) {
                            csst->mass_jac[i] = 0.0;
                        }
                        
                        // Map the reduced gradient to full workspace format
                        // CSE outputs: [comp0_dT, comp0_dY1, comp0_dY2, ..., comp1_dT, comp1_dY1, comp1_dY2, ...]
                        // Full format expects: [comp0_dN, comp0_dP, comp0_dT, comp0_dY1, comp0_dY2, ..., comp1_dN, comp1_dP, comp1_dT, ...]
                        int reduced_cols = 1 + compset->phase_record->phase_dof; // T + site fractions
                        int full_cols = spec->num_statevars + compset->phase_record->phase_dof;
                        
                        for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                            // Temperature derivative (index 0 in reduced -> index 2 in full)
                            csst->mass_jac[comp_idx * full_cols + 2] = temp_mass_jac[comp_idx * reduced_cols + 0];
                            
                            // Site fraction derivatives (indices 1..n in reduced -> indices num_statevars..num_statevars+n-1 in full)
                            for (int j = 0; j < compset->phase_record->phase_dof; ++j) {
                                csst->mass_jac[comp_idx * full_cols + spec->num_statevars + j] = 
                                    temp_mass_jac[comp_idx * reduced_cols + 1 + j];
                            }
                        }
                        
                        // DEBUG: Print raw gradient values from formulamole_grad
                        #ifdef VERBOSE_DEBUG
                        if (idx == 0 && iteration < 2) {
                            printf("GPU DEBUG: Raw formulamole_grad output (phase %d):\n", idx);
                            int reduced_cols = 1 + compset->phase_record->phase_dof;
                            for (int comp_idx = 0; comp_idx < spec->num_components && comp_idx < 2; ++comp_idx) {
                                printf("  Component %d gradients (reduced format): ", comp_idx);
                                for (int j = 0; j < reduced_cols; ++j) {
                                    printf("d/d[T,Y1,Y2][%d]=%e ", j, temp_mass_jac[comp_idx * reduced_cols + j]);
                                }
                                printf("\n");
                            }
                        }
                        #endif
                        
                        // DEBUG: Print mass_jac for first phase and iteration
                        #ifdef VERBOSE_DEBUG
                        if (idx == 0 && iteration < 2) {
                            printf("GPU DEBUG: Phase %d mass_jac matrix (workspace DOF format):\\n", idx);
                            printf("  Columns: [N, P, T, Y_NB, Y_TI, ...]\\n");
                            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                                printf("  Component %d: ", comp_idx);
                                for (int j = 0; j < csst->mass_jac_cols && j < 5; ++j) {
                                    printf("%.6e ", csst->mass_jac[comp_idx * csst->mass_jac_cols + j]);
                                }
                                printf("\\n");
                            }
                        }
                        #endif
                    }
                }
            }

            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                // masses should contain mole fractions from formulamole_obj
                // This matches CPU line 749: compset.phase_record.formulamole_obj(csst.masses[comp_idx, :], x, comp_idx)
                csst->masses[comp_idx] = formulamoles[comp_idx];
                
                // DEBUG: Print phase compositions
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0 && iteration < 3 && comp_idx < 2) {
                    printf("GPU: Phase %d composition[%d] = %.6f (will update after compset update)\n", idx, comp_idx, 
                           phase_compositions[idx * MAX_COMPONENTS + comp_idx]);
                }
                #endif

                // phase_amt is in formula units; CPU (minimizer.pyx:867) guards with > 0
                if (phase_amt[idx] > 0.0) {
                    mole_fractions[comp_idx] += phase_amt[idx] * csst->masses[comp_idx];
                    system_amount += phase_amt[idx] * csst->masses[comp_idx];
                }
            }

            // CPU (minimizer.pyx:870) updates phase_compositions UNCONDITIONALLY,
            // including zero-amount (metastable) compsets.
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                phase_compositions[idx * MAX_COMPONENTS + comp_idx] = csst->masses[comp_idx];
            }
        }

        if (fabs(system_amount) > 1e-12) {
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                mole_fractions[comp_idx] /= system_amount;
            }
        } else { // If system_amount is zero, mole_fractions should reflect this (e.g. uniform or zero)
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                 mole_fractions[comp_idx] = (spec->num_components > 0) ? 1.0/spec->num_components : 0.0; // Or just 0.0
            }
        }
        
        if (condition_idx < 3) {
            gpu_debug_log_value("system_amount", system_amount);
        }
        
        // Count active phases
        int num_phases_active = 0;
        for (int idx = 0; idx < num_compsets; ++idx) {
            if (phase_amt[idx] > 1e-10) num_phases_active++;
        }
        if (condition_idx < 3) {
            gpu_debug_log_value("num_phases_active", (double)num_phases_active);
        }
        
        // Log phase details
        if (condition_idx < 3) {
            #ifdef VERBOSE_DEBUG
            for (int idx = 0; idx < num_compsets; ++idx) {
                if (phase_amt[idx] > 1e-10) {
                    // Use direct printf instead of sprintf for device code
                    printf("[GPU]   phase_%d_%d: %.15e\n", idx, idx, phase_amt[idx]);
                    printf("[GPU]   phase_%d_X: [", idx);
                    for (int i = 0; i < spec->num_components && i < 5; ++i) {
                        printf("%.6f", phase_compositions[idx * MAX_COMPONENTS + i]);
                        if (i < spec->num_components - 1) printf(", ");
                    }
                    if (spec->num_components > 5) printf("...");
                    printf("]\n");
                }
            }
            #endif
        }
        
        if (condition_idx < 3) {
            gpu_debug_log_array("mole_fractions", mole_fractions, spec->num_components);
        }

        mass_residual = 0.0;
        for (int cond_idx = 0; cond_idx < spec->num_prescribed_mole_fraction_conditions; ++cond_idx) {
            double current_sum = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_prescribed_mole_fraction_coefficients_cols; ++comp_idx) {
                 current_sum += spec->prescribed_mole_fraction_coefficients[cond_idx][comp_idx] * mole_fractions[comp_idx];
            }
            double residual_contrib = fabs(current_sum - spec->prescribed_mole_fraction_rhs[cond_idx]);
            
            // DEBUG: Print details for first few iterations
            #ifdef VERBOSE_DEBUG
            if (condition_idx < 3 && iteration < 3) {
                printf("[GPU] Mass residual calc (iteration %d, cond %d):\n", iteration, cond_idx);
                printf("  Coefficients: [");
                for (int i = 0; i < spec->num_prescribed_mole_fraction_coefficients_cols; ++i) {
                    printf("%.3f", spec->prescribed_mole_fraction_coefficients[cond_idx][i]);
                    if (i < spec->num_prescribed_mole_fraction_coefficients_cols - 1) printf(", ");
                }
                printf("]\n");
                printf("  Current sum: %.15e\n", current_sum);
                printf("  RHS: %.15e (accessing index %d)\n", spec->prescribed_mole_fraction_rhs[cond_idx], cond_idx);
                // DEBUG: Print all RHS values
                if (cond_idx == 0) {
                    printf("  All RHS values: [");
                    for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS && i < 4; ++i) {
                        printf("%.3f", spec->prescribed_mole_fraction_rhs[i]);
                        if (i < 3) printf(", ");
                    }
                    printf("]\n");
                }
                printf("  Residual contribution: %.15e\n", residual_contrib);
            }
            #endif
            
            mass_residual += residual_contrib;
        }
        
        if (condition_idx < 3) {
            gpu_debug_log_value("mass_residual", mass_residual);
        }

        // SEGMENT 23: STATE RECOMPUTE - PHASE QUANTITIES
        gpu_debug_log(23, "State recompute - phase quantities", condition_idx);
        if (condition_idx < 3) {
            gpu_debug_log_value("num_phases_active", (double)num_phases_active);
        }
        
        if (condition_idx < 3) {
            #ifdef VERBOSE_DEBUG
            for (int idx = 0; idx < num_compsets; ++idx) {
                if (phase_amt[idx] > 1e-10) {
                    // Use direct printf instead of sprintf for device code
                    printf("[GPU]   phase_%d_NP: %.15e\n", idx, phase_amt[idx]);
                    printf("[GPU]   phase_%d_X: [", idx);
                    for (int i = 0; i < spec->num_components && i < 5; ++i) {
                        printf("%.6f", phase_compositions[idx * MAX_COMPONENTS + i]);
                        if (i < spec->num_components - 1) printf(", ");
                    }
                    if (spec->num_components > 5) printf("...");
                    printf("]\n");
                }
            }
            #endif
        }

        for (int idx = 0; idx < num_compsets; ++idx) {
            CompositionSet* compset = &compsets[idx];
            CompsetState* csst = &cs_states[idx];
            if (compset->phase_record == nullptr) continue;

            // CPU (minimizer.pyx:912) recomputes phase quantities for ALL compsets,
            // with NO amount filter: removed (amt=0) compsets keep getting fresh
            // grad/hess/c_G so advance_state moves their dof consistently with the
            // current chemical potentials, and change_phases can re-add them with a
            // meaningful driving force. Skipping them lets stale delta_y drive their
            // dof to a sublattice vertex, permanently blocking re-addition.

            const PhaseRecord* pr = compset->phase_record;

            // REMOVED: Old code that created current_dof_for_phase incorrectly
            // Now we create model_dof_for_calcs properly from workspace DOF when needed

            // Calculate phase_comp_sum from stored phase_compositions
            // This matches CPU behavior (minimizer.pyx line 880-881)
            // For multi-sublattice phases, this equals the sum of site ratios (e.g., 20 for ALCU_ZETA)
            double phase_sum_moles_atoms_per_formula = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_components; comp_idx++) {
                phase_sum_moles_atoms_per_formula += phase_compositions[idx * MAX_COMPONENTS + comp_idx];
            }
            
            // Safety check
            // CPU doesn't have a fallback here - let it be what it is

            // Call compset update. NP is moles of formula units.
            // Match CPU algorithm - multiply phase_amt by phase_sum_moles_atoms_per_formula
            // This converts from formula units back to mole fractions, matching CPU minimizer.pyx line 885
            double update_amount = phase_amt[idx] * phase_sum_moles_atoms_per_formula;
            
            // DEBUG: Print update calculation
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && iteration < 3) {
                printf("[GPU UPDATE] Phase %d: phase_amt=%.15e (formula units), phase_comp_sum=%.15e\n",
                       idx, phase_amt[idx], phase_sum_moles_atoms_per_formula);
                printf("[GPU UPDATE] Phase %d: update_amount=%.15e (mole fractions for NP)\n",
                       idx, update_amount);
            }
            #endif
            
            // Additional safety check for update amount
            // CPU doesn't check update_amount bounds
            // Pass the actual workspace DOF to update, not the model DOF
            // The update function expects workspace state variables, not model state variables
            #ifdef PYCGPU_PROF
            long long prof_f0 = clock64();
            #endif
            compset->update(&compset->dof[spec->num_statevars], update_amount, compset->dof, spec->num_statevars);
            #ifdef PYCGPU_PROF
            if (thread_id < PYCGPU_PROF_MAXT) g_prof_funcs[thread_id] += clock64() - prof_f0;
            #endif
            
            // csst->energy will be G per formula unit (from pr->formulaobj)
            // Pass full workspace DOF to energy calculation, matching CPU behavior
            // The generated functions now expect workspace DOF format [N, P, T, Y1, Y2...]
            // Use pr->formulaobj() for equilibrium matrix (per formula unit, not per mole atoms)
            // This matches the CPU behavior where equilibrium matrix uses unnormalized energy values
            csst->energy = pr->formulaobj(compset->dof);
            
            // Add numerical debug output for phase energy
            #ifdef VERBOSE_DEBUG
            printf("[GPU]   phase_%d_comp_sum: %.15e\n", idx, phase_sum_moles_atoms_per_formula);
            printf("[GPU]   phase_%d_energy: %.15e\n", idx, csst->energy);
            #endif


            for(int i=0; i<csst->mass_jac_rows * csst->mass_jac_cols; ++i) csst->mass_jac[i] = 0.0;
            for(int i=0; i<csst->phase_matrix_dim * csst->phase_matrix_dim; ++i) csst->phase_matrix[i] = 0.0;
            for(int i=0; i<csst->internal_cons_length; ++i) csst->internal_cons[i] = 0.0;
            for(int i=0; i<csst->hess_rows * csst->hess_cols; ++i) csst->hess[i] = 0.0;
            for(int i=0; i<csst->grad_length; ++i) csst->grad[i] = 0.0;

            // DEBUG: Print workspace DOF before calling formulamole_grad
            #ifdef VERBOSE_DEBUG
            if (idx == 0 && iteration < 2) {
                printf("GPU DEBUG: Before DOF print - spec=%p, pr=%p, compset=%p\n", spec, pr, compset);
                printf("  spec->num_statevars=%d, pr->phase_dof=%d, total=%d\n", 
                       spec->num_statevars, pr->phase_dof, spec->num_statevars + pr->phase_dof);
                
                if (spec->num_statevars + pr->phase_dof > 0 && spec->num_statevars + pr->phase_dof < 20) {
                    printf("GPU DEBUG: workspace DOF before formulamole_grad: [");
                    for (int i = 0; i < spec->num_statevars + pr->phase_dof; i++) {
                        printf("%.6f", compset->dof[i]);
                        if (i < spec->num_statevars + pr->phase_dof - 1) printf(", ");
                    }
                    printf("]\n");
                } else {
                    printf("GPU ERROR: Invalid DOF size: %d\n", spec->num_statevars + pr->phase_dof);
                }
            }
            #endif
            
            // Properly handle mass_jac from formulamole_grad output
            // formulamole_grad outputs a matrix of size (num_nonvacant_elements × num_model_dof)
            // where num_model_dof = pr->num_statevars + pr->phase_dof
            // But csst->mass_jac expects size (num_components × (spec->num_statevars + pr->phase_dof))
            
            // First, we need a temporary array to hold the formulamole_grad output
            double temp_mass_jac[MAX_COMPONENTS * MAX_DOF_PER_PHASE];
            for (int i = 0; i < MAX_COMPONENTS * MAX_DOF_PER_PHASE; ++i) {
                temp_mass_jac[i] = 0.0;
            }
            
            // Call formulamole_grad to get gradients for nonvacant elements only
            if (pr->formulamole_grad != nullptr) {
                // DEBUG: Print the DOF values being passed to formulamole_grad
                #ifdef VERBOSE_DEBUG
                if (iteration == 0) {
                    printf("[GPU MASS_JAC] Phase %d calling formulamole_grad with DOF: ", idx);
                    for (int k = 0; k < spec->num_statevars + pr->phase_dof; ++k) {
                        printf("%e ", compset->dof[k]);
                    }
                    printf("\n");
                    printf("[GPU MASS_JAC DEBUG] pr->num_statevars=%d, spec->num_statevars=%d, pr->phase_dof=%d\n",
                           pr->num_statevars, spec->num_statevars, pr->phase_dof);
                }
                #endif
                pr->formulamole_grad(temp_mass_jac, compset->dof);
                
                // DEBUG: Print raw formulamole_grad output
                #ifdef VERBOSE_DEBUG
                if (iteration < 2) {
                    printf("GPU DEBUG: Raw formulamole_grad output (phase %d):\n", idx);
                    // CSE functions output in reduced format [T, Y1, Y2, ...]
                    // So the gradient matrix is nonvacant_elements x (1 + phase_dof)
                    int reduced_cols = 1 + pr->phase_dof; // T + site fractions
                    for (int i = 0; i < pr->nonvacant_elements && i < 3; i++) {
                        printf("  Component %d gradients (CSE format [T,Y1,Y2]): ", i);
                        for (int j = 0; j < reduced_cols && j < 5; j++) {
                            printf("[%d]=%e ", j, temp_mass_jac[i * reduced_cols + j]);
                        }
                        printf("\n");
                    }
                }
                #endif
            } else {
                #ifdef VERBOSE_DEBUG
                printf("GPU ERROR: formulamole_grad is null for phase %d\\n", idx);
                #endif
            }
            
            // Now copy the gradients to the correct positions in csst->mass_jac
            // CSE functions output in reduced format [T, Y1, Y2, ...]
            // We need to map this to workspace format [N, P, T, Y1, Y2, ...]
            int nonvacant_idx = 0;
            int reduced_cols = 1 + pr->phase_dof; // CSE output columns: T + site fractions
            
            for (int comp_idx = 0; comp_idx < spec->num_components; comp_idx++) {
                // Check if this component is a nonvacant element
                // For now, assume components are ordered as [NB, TI, VA] and nonvacant are [NB, TI]
                if (comp_idx < pr->nonvacant_elements) {
                    // This is a nonvacant element - copy its gradients
                    // First, zero out the entire row
                    for (int col = 0; col < csst->mass_jac_cols; col++) {
                        csst->mass_jac[comp_idx * csst->mass_jac_cols + col] = 0.0;
                    }
                    
                    // Map from CSE reduced format to Workspace DOF
                    for (int cse_col = 0; cse_col < reduced_cols; cse_col++) {
                        int workspace_col;
                        if (cse_col == 0) {
                            // Temperature column in CSE -> position 2 in workspace [N, P, T]
                            workspace_col = 2;
                        } else {
                            // Site fraction columns: cse_col 1,2,... -> workspace cols 3,4,...
                            workspace_col = spec->num_statevars + (cse_col - 1);
                        }
                        
                        if (workspace_col < csst->mass_jac_cols) {
                            csst->mass_jac[comp_idx * csst->mass_jac_cols + workspace_col] = 
                                temp_mass_jac[nonvacant_idx * reduced_cols + cse_col];
                        }
                    }
                    nonvacant_idx++;
                } else {
                    // This is VA or a component not in this phase - all gradients are zero
                    for (int col = 0; col < csst->mass_jac_cols; col++) {
                        csst->mass_jac[comp_idx * csst->mass_jac_cols + col] = 0.0;
                    }
                }
            }
            
            // NOTE: The CPU code treats site fractions as independent variables
            // So d(moles_i)/d(Y_j) = 0 for i != j
            // The generated formulamole_grad function already handles this correctly
            
            // DEBUG: Print mass_jac matrix
            #ifdef VERBOSE_DEBUG
            if (iteration < 2) {
                printf("GPU DEBUG: Phase %d mass_jac matrix (workspace DOF format):\\n", idx);
                printf("  Columns: [N, P, T, Y_NB, Y_TI, ...]\\n");
                for (int comp_idx = 0; comp_idx < spec->num_components && comp_idx < 3; comp_idx++) {
                    printf("  Component %d: ", comp_idx);
                    for (int col = 0; col < csst->mass_jac_cols && col < 5; col++) {
                        printf("%e ", csst->mass_jac[comp_idx * csst->mass_jac_cols + col]);
                    }
                    printf("\\n");
                }
            }
            #endif
            
            if (pr->formulahess != nullptr) {
                // Calling formulahess - MUST pass full workspace DOF like CPU does
                // DEBUG: Print DOF values passed to formulahess
                #ifdef VERBOSE_DEBUG
                if (iteration < 3) {
                    printf("[GPU FORMULAHESS INPUT] Phase %d iteration %d, DOF: ", idx, iteration);
                    for (int i = 0; i < 5; i++) {
                        printf("%.15e ", compset->dof[i]);
                    }
                    printf("\n");
                }
                #endif
                // Temporary array to hold the reduced Hessian output from CSE functions
                double temp_hess[(MAX_DOF_PER_PHASE + 1) * (MAX_DOF_PER_PHASE + 1)];
                #ifdef PYCGPU_PROF
                long long prof_h0 = clock64();
                #endif
                pr->formulahess(temp_hess, compset->dof);
                #ifdef PYCGPU_FP32EMU
                pycgpu_f32_arr(temp_hess, (1 + pr->phase_dof) * (1 + pr->phase_dof));
                #endif
                #ifdef PYCGPU_PROF
                if (thread_id < PYCGPU_PROF_MAXT) g_prof_hess[thread_id] += clock64() - prof_h0;
                #endif
                
                // DEBUG: Print raw Hessian output
                #ifdef VERBOSE_DEBUG
                if (idx == 0 && iteration < 2) {
                    printf("GPU DEBUG: Raw Hessian output for phase %d:\n", idx);
                    int reduced_dims = 1 + pr->phase_dof; // T + site fractions
                    for (int i = 0; i < reduced_dims; ++i) {
                        printf("  Row %d: ", i);
                        for (int j = 0; j < reduced_dims; ++j) {
                            printf("%.6e ", temp_hess[i * reduced_dims + j]);
                        }
                        printf("\n");
                    }
                }
                #endif
                
                // Map the reduced Hessian (T + site fractions) to the full matrix (N, P, T + site fractions)
                // The CSE Hessian functions output a (1 + phase_dof) x (1 + phase_dof) matrix:
                // - temp_hess[0] corresponds to d²G/dT² (T,T element) 
                // - temp_hess[i] corresponds to d²G/dTdY_i (T,site_fraction elements)
                // - temp_hess[j*(1+phase_dof)+i] corresponds to d²G/dY_i dY_j (site_fraction block)
                
                int reduced_dim = 1 + pr->phase_dof; // T + site fractions
                
                // Zero out the full Hessian matrix first
                for (int i = 0; i < csst->hess_rows * csst->hess_cols; ++i) {
                    csst->hess[i] = 0.0;
                }
                
                // Map T,T element: temp_hess[0] -> csst->hess[2,2] (T is at index 2)
                csst->hess[2 * csst->hess_cols + 2] = temp_hess[0];
                
                // Map T,site_fraction elements: temp_hess[i] -> csst->hess[2, 3+i-1] and csst->hess[3+i-1, 2]
                for (int i = 1; i < reduced_dim; i++) {
                    int site_idx = spec->num_statevars + (i - 1); // Convert to full matrix site fraction index
                    // T,site_fraction element
                    csst->hess[2 * csst->hess_cols + site_idx] = temp_hess[i];
                    // site_fraction,T element (symmetric)
                    csst->hess[site_idx * csst->hess_cols + 2] = temp_hess[i];
                }
                
                // Map site_fraction,site_fraction block
                for (int i = 1; i < reduced_dim; i++) {
                    for (int j = 1; j < reduced_dim; j++) {
                        int reduced_idx = i * reduced_dim + j;
                        int full_row = spec->num_statevars + (i - 1);
                        int full_col = spec->num_statevars + (j - 1);
                        csst->hess[full_row * csst->hess_cols + full_col] = temp_hess[reduced_idx];
                    }
                }
                
        // DEBUG: Print Hessian values
        // DEBUG: Print Hessian values
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0 && idx < 2) {
            printf("GPU DEBUG: Hessian calculated for phase record %d\n", idx);
            printf("  Hessian size: %dx%d, hess_cols=%d\n", pr->phase_dof + spec->num_statevars, pr->phase_dof + spec->num_statevars, csst->hess_cols);
            printf("  spec->num_statevars=%d, pr->phase_dof=%d\n", spec->num_statevars, pr->phase_dof);
        }
        if (thread_id == 0 && idx < 2 && pr->phase_dof > 0) {
            printf("  Site fraction Hessian block:\n");
            for (int i = 0; i < pr->phase_dof; i++) {
                printf("    [%d]", i);
                for (int j = 0; j < pr->phase_dof; j++) {
                    int hidx = (spec->num_statevars + i) * csst->hess_cols + (spec->num_statevars + j);
                    printf(" %e (idx=%d)", csst->hess[hidx], hidx);
                }
                printf("\n");
            }
        }
        #endif

                // Completed formulahess
            }
            // CPU doesn't have a fallback for missing Hessian
            
            // Removed debug prints for formulagrad to debug alignment issue
            
            // Check if formulagrad pointer looks valid
            if (pr->formulagrad == nullptr) {
                #ifdef VERBOSE_DEBUG
                printf("GPU ERROR: formulagrad pointer is null, skipping\n");
                #endif
                // Set gradient to zero (already initialized)
            } else {
                // CSE gradient functions output reduced gradient in the correct order
                // Expected order: [dG/dT, dG/dY1, dG/dY2, ...]
                double temp_grad[1 + MAX_DOF_PER_PHASE];  // Temporary array for reduced gradient
                #ifdef PYCGPU_PROF
                long long prof_f1 = clock64();
                #endif
                pr->formulagrad(temp_grad, compset->dof);
                #ifdef PYCGPU_FP32EMU
                pycgpu_f32_arr(temp_grad, 1 + pr->phase_dof);
                #endif
                #ifdef PYCGPU_PROF
                if (thread_id < PYCGPU_PROF_MAXT) g_prof_funcs[thread_id] += clock64() - prof_f1;
                #endif
                
                // Zero out the full gradient array first
                for (int i = 0; i < csst->grad_length; ++i) {
                    csst->grad[i] = 0.0;
                }
                
                // Map the reduced gradient to full gradient:
                // temp_grad[0] -> temperature derivative (index 2 in full gradient)
                // temp_grad[1..n] -> site fraction derivatives (indices num_statevars+0..num_statevars+n-1)
                
                // Temperature derivative
                csst->grad[2] = temp_grad[0];
                
                // Site fraction derivatives
                for (int i = 0; i < pr->phase_dof; i++) {
                    csst->grad[spec->num_statevars + i] = temp_grad[1 + i];
                }
                
                // DEBUG: Print gradient values for first few phases at iteration 0
                #ifdef VERBOSE_DEBUG
                if (spec->num_statevars == 3 && (pr->phase_dof == 2 || pr->phase_dof == 3)) {
                    printf("[GPU GRAD DEBUG] Phase formulagrad results (phase_dof=%d):\n", pr->phase_dof);
                    if (pr->phase_dof == 2) {
                        printf("  temp_grad: [%e, %e, %e] (T, Y1, Y2)\n", temp_grad[0], temp_grad[1], temp_grad[2]);
                        printf("  mapped to grad[2]=%e, grad[3]=%e, grad[4]=%e\n", 
                               csst->grad[2], csst->grad[3], csst->grad[4]);
                        printf("  DOF values: T=%e, Y1=%e, Y2=%e\n", 
                               compset->dof[2], compset->dof[3], compset->dof[4]);
                    } else if (pr->phase_dof == 3) {
                        printf("  temp_grad: [%e, %e, %e, %e] (T, Y1, Y2, Y3)\n", 
                               temp_grad[0], temp_grad[1], temp_grad[2], temp_grad[3]);
                        printf("  mapped to grad[2]=%e, grad[3]=%e, grad[4]=%e, grad[5]=%e\n", 
                               csst->grad[2], csst->grad[3], csst->grad[4], csst->grad[5]);
                        printf("  DOF values: T=%e, Y1=%e, Y2=%e, Y3=%e\n", 
                               compset->dof[2], compset->dof[3], compset->dof[4], compset->dof[5]);
                    }
                }
                #endif
                
                // N and P derivatives (indices 0, 1) remain zero as CSE doesn't compute them
            }
            // Completed formulagrad
            
            // DEBUG: Check if gradients are truly zero
            bool all_gradients_zero = true;
            for (int i = 0; i < pr->num_statevars + pr->phase_dof; ++i) {
                if (fabs(csst->grad[i]) > 1e-15) {
                    all_gradients_zero = false;
                    break;
                }
            }
            #ifdef VERBOSE_DEBUG
            if (all_gradients_zero && iteration < 3) {
                printf("GPU WARNING: All gradients are zero for phase %d at iteration %d\n", idx, iteration);
                printf("  workspace_dof: ");
                for (int i = 0; i < spec->num_statevars + pr->phase_dof; ++i) {
                    printf("%.6f ", compset->dof[i]);
                }
                printf("\n");
                printf("  energy: %.15e\n", csst->energy);
            }
            #endif
            
            pr->internal_cons_func(csst->internal_cons, compset->dof);
            
            // CSE constraint Jacobian functions output reduced format
            if (pr->internal_cons_jac != nullptr) {
                // Temporary array for reduced constraint Jacobian
                // CSE outputs flat array: [dC1/dT, dC1/dY1, dC1/dY2, ..., dC2/dT, dC2/dY1, ...]
                double temp_cons_jac[(1 + MAX_DOF_PER_PHASE) * MAX_INTERNAL_CONSTRAINTS];
                #ifdef PYCGPU_PROF
                long long prof_f2 = clock64();
                #endif
                pr->internal_cons_jac(temp_cons_jac, compset->dof);
                #ifdef PYCGPU_FP32EMU
                pycgpu_f32_arr(temp_cons_jac, pr->num_internal_cons * (pr->num_statevars + pr->phase_dof));
                #endif
                #ifdef PYCGPU_PROF
                if (thread_id < PYCGPU_PROF_MAXT) g_prof_funcs[thread_id] += clock64() - prof_f2;
                #endif
                
                // Zero out the full constraint Jacobian first
                for (int i = 0; i < pr->num_internal_cons * (spec->num_statevars + pr->phase_dof); ++i) {
                    csst->cons_jac_tmp[i] = 0.0;
                }
                
                // Map the reduced constraint Jacobian to full format
                // CSE outputs: [dC/dT, dC/dY1, dC/dY2, ...] in flat array
                // Full format expects: [dC/dN, dC/dP, dC/dT, dC/dY1, dC/dY2, ...] per constraint
                int reduced_size = 1 + pr->phase_dof; // T + site fractions
                int full_cols = spec->num_statevars + pr->phase_dof;
                
                for (int i = 0; i < pr->num_internal_cons; ++i) {
                    // Temperature derivative (index 0 in reduced -> index 2 in full)
                    csst->cons_jac_tmp[i * full_cols + 2] = temp_cons_jac[i * reduced_size + 0];
                    
                    // Site fraction derivatives (indices 1..n in reduced -> indices num_statevars..num_statevars+n-1 in full)
                    for (int j = 0; j < pr->phase_dof; ++j) {
                        csst->cons_jac_tmp[i * full_cols + spec->num_statevars + j] = temp_cons_jac[i * reduced_size + 1 + j];
                    }
                }
            }
            
            // DEBUG: Print constraint Jacobian values
            #ifdef VERBOSE_DEBUG
            if (idx == 0 && iteration < 2 && pr->num_internal_cons > 0) {
                printf("GPU DEBUG: Constraint Jacobian for phase %d (num_cons=%d):\n", idx, pr->num_internal_cons);
                int cons_jac_dim = spec->num_statevars + pr->phase_dof;
                for (int i = 0; i < pr->num_internal_cons; ++i) {
                    printf("  Constraint %d: ", i);
                    for (int j = 0; j < cons_jac_dim; ++j) {
                        printf("%.6e ", csst->cons_jac_tmp[i * cons_jac_dim + j]);
                    }
                    printf("\n");
                }
            }
            #endif

            // Pass site fractions from workspace DOF to compute_phase_matrix
            // Site fractions start at spec->num_statevars in workspace DOF
            compute_phase_matrix(csst->phase_matrix, csst->hess, csst->cons_jac_tmp,
                                 *compset, spec->num_statevars,
                                 &compset->dof[spec->num_statevars]);

            // DEBUG: Check phase_matrix before inversion
            #ifdef VERBOSE_DEBUG
            if (idx == 0 && iteration < 2) {
                printf("GPU DEBUG: Phase matrix before inversion (dim=%d):\n", csst->full_e_matrix_dim);
                for (int i = 0; i < csst->full_e_matrix_dim; ++i) {
                    printf("  Row %d: ", i);
                    for (int j = 0; j < csst->full_e_matrix_dim; ++j) {
                        printf("%e ", csst->phase_matrix[i * csst->full_e_matrix_dim + j]);
                    }
                    printf("\n");
                }
            }
            #endif

            for (int i = 0; i < csst->full_e_matrix_dim * csst->full_e_matrix_dim; ++i) {
                csst->full_e_matrix[i] = csst->phase_matrix[i];
            }
            
            // Use LU decomposition instead of SVD to match CPU behavior exactly
            // CPU uses LAPACK's dgesv (LU decomposition with partial pivoting)
            // GPU was using SVD which produces different results for constrained matrices
            #ifdef PYCGPU_PROF
            long long prof_i0 = clock64();
            #endif
            {
                // LAPACK-transliterated inverse with the reference wrapper's
                // exact semantics (minimizer.pyx invert_matrix: NaN scrub ->
                // zeros, dgetrf+dgetri with lwork=n, failure -> -1e19).  The
                // reference feeds its C-ordered buffer straight to Fortran, so
                // passing our row-major buffer unchanged sees identical bytes.
                int _ipiv[MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS];
                pyclap_invert_pycalphad(csst->full_e_matrix,
                                        csst->full_e_matrix_dim, _ipiv, work_inv);
            }
            #ifdef PYCGPU_PROF
            if (thread_id < PYCGPU_PROF_MAXT) g_prof_inv[thread_id] += clock64() - prof_i0;
            #endif
            
            // DEBUG: Check full_e_matrix after inversion for BOTH phases
            #ifdef VERBOSE_DEBUG
            if (iteration == 0) {
                printf("GPU DEBUG: Phase %d Full E matrix after inversion (dim=%d):\n", idx, csst->full_e_matrix_dim);
                for (int i = 0; i < csst->full_e_matrix_dim; ++i) {
                    printf("  Row %d: ", i);
                    for (int j = 0; j < csst->full_e_matrix_dim; ++j) {
                        printf("%e ", csst->full_e_matrix[i * csst->full_e_matrix_dim + j]);
                    }
                    printf("\n");
                }
                
                // Print diagonal values specifically
                printf("GPU DEBUG: Phase %d diagonal values after inversion:\n", idx);
                for (int i = 0; i < csst->full_e_matrix_dim; ++i) {
                    printf("  [%d,%d] = %.15e\n", i, i, csst->full_e_matrix[i * csst->full_e_matrix_dim + i]);
                }
            }
            #endif

            int num_phase_dof_for_csst = pr->phase_dof;
            for(int i=0; i < csst->c_G_length; ++i) csst->c_G[i] = 0.0;
            for(int i=0; i < csst->c_statevars_rows * csst->c_statevars_cols; ++i) csst->c_statevars[i] = 0.0;
            for(int i=0; i < csst->c_component_rows * csst->c_component_cols; ++i) csst->c_component[i] = 0.0;
            csst->moles_normalization = 0.0;
            for(int i=0; i < csst->moles_normalization_grad_length; ++i) csst->moles_normalization_grad[i] = 0.0;

            // DEBUG: Print gradient values before computing c_G
            #ifdef VERBOSE_DEBUG
            if (idx < 2 && iteration < 5) {
                printf("GPU DEBUG: Phase %d gradients before c_G calculation (iter %d):\n", idx, iteration);
                printf("  energy = %.15e\n", csst->energy);
                printf("  phase_amt = %.15e\n", phase_amt[idx]);
                for (int j = 0; j < csst->grad_length; ++j) {
                    printf("  grad[%d] = %.15e\n", j, csst->grad[j]);
                }
                // Also print masses for debugging
                printf("  masses: [");
                for (int i = 0; i < spec->num_components; ++i) {
                    printf("%.6f", csst->masses[i]);
                    if (i < spec->num_components - 1) printf(", ");
                }
                printf("]\n");
            }
            #endif
            
            // DEBUG: Print matrix and gradient values for problematic phases
            #ifdef VERBOSE_DEBUG
            if (iteration == 0 && condition_idx == 0 && idx <= 2) {
                printf("[GPU c_G CALC] Phase %d matrix inversion and c_G calculation:\n", idx);
                printf("  Inverted matrix (full_e_matrix):\n");
                for (int i = 0; i < num_phase_dof_for_csst; i++) {
                    printf("    Row %d: ", i);
                    for (int j = 0; j < num_phase_dof_for_csst; j++) {
                        printf("%e ", csst->full_e_matrix[i * csst->full_e_matrix_dim + j]);
                    }
                    printf("\n");
                }
                printf("  Gradient values (grad[num_statevars+j]):\n");
                for (int j = 0; j < num_phase_dof_for_csst; j++) {
                    printf("    grad[%d] = %e\n", spec->num_statevars + j, csst->grad[spec->num_statevars + j]);
                }
            }
            #endif
            
            for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                for (int j = 0; j < num_phase_dof_for_csst; ++j) {
                    // With updated energy functions, grad array is now in Workspace format [N, P, T, Y1, Y2, ...]
                    double matrix_elem = csst->full_e_matrix[i * csst->full_e_matrix_dim + j];
                    double grad_elem = csst->grad[spec->num_statevars + j];
                    csst->c_G[i] -= matrix_elem * grad_elem;
                    
                    // DEBUG: Print calculation details for first phase and iteration
                    #ifdef VERBOSE_DEBUG
                    if (thread_id == 0 && idx == 0 && iteration < 2) {
                        printf("  c_G[%d] calc: full_e_matrix[%d,%d]=%e * grad[%d]=%e = %e\n", 
                               i, i, j, matrix_elem, spec->num_statevars + j, grad_elem, matrix_elem * grad_elem);
                    }
                    #endif
                }
            }
            
            // DEBUG: Print final c_G values  
            #ifdef VERBOSE_DEBUG
            if (iteration == 0 && condition_idx == 0 && idx <= 2) {
                printf("  Final c_G values: ");
                for (int i = 0; i < num_phase_dof_for_csst; i++) {
                    printf("c_G[%d]=%e ", i, csst->c_G[i]);
                }
                printf("\n");
            }
            #endif
            
            // DEBUG: Print c_G values after calculation
            #ifdef VERBOSE_DEBUG
            if (idx < 2 && iteration < 5) {
                printf("GPU DEBUG: Phase %d c_G values (iter %d):\n", idx, iteration);
                printf("  phase_amt = %.15e\n", phase_amt[idx]);
                printf("  gradient values: [");
                for (int i = 0; i < pr->phase_dof; ++i) {
                    printf("%.6e", csst->grad[spec->num_statevars + i]);
                    if (i < pr->phase_dof - 1) printf(", ");
                }
                printf("]\n");
                printf("  full_e_matrix diagonal: [");
                for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                    printf("%.6e", csst->full_e_matrix[i * csst->full_e_matrix_dim + i]);
                    if (i < num_phase_dof_for_csst - 1) printf(", ");
                }
                printf("]\n");
                for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                    printf("  c_G[%i] = %.15e\n", i, csst->c_G[i]);
                }
                // Also print current site fractions to track convergence
                printf("  Current site fractions: [");
                for (int i = 0; i < pr->phase_dof; ++i) {
                    printf("%.6f", compset->dof[spec->num_statevars + i]);
                    if (i < pr->phase_dof - 1) printf(", ");
                }
                printf("]\n");
            }
            #endif
            for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                for (int j = 0; j < num_phase_dof_for_csst; ++j) {
                    for (int sv_idx = 0; sv_idx < spec->num_statevars; ++sv_idx) {
                        // With CSE Hessian functions, only T derivatives (sv_idx=2) are non-zero
                        // N and P derivatives (sv_idx=0,1) are always zero
                        if (sv_idx == 2) { // Temperature index
                            csst->c_statevars[i * csst->c_statevars_cols + sv_idx] -=
                                csst->full_e_matrix[i * csst->full_e_matrix_dim + j] *
                                csst->hess[(spec->num_statevars + j) * csst->hess_cols + sv_idx];
                        }
                        // For N and P (sv_idx=0,1), the derivative is zero, so no contribution
                    }
                }
            }
            
            // Calculate c_component IMMEDIATELY after phase matrix inversion
            // This must be done before fill_equilibrium_system uses c_component
            // DEBUG: Print mass_jac values before c_component calculation
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && iteration < 2) {
                printf("[GPU DEBUG] Phase %d mass_jac BEFORE c_component calc (phase_dof=%d, num_elements=%d):\n", 
                       idx, pr->phase_dof, pr->num_elements);
                for (int cidx = 0; cidx < spec->num_components && cidx < 3; cidx++) {
                    printf("  Component %d: ", cidx);
                    for (int col = spec->num_statevars; col < spec->num_statevars + pr->phase_dof && col < spec->num_statevars + 3; col++) {
                        printf("mass_jac[%d,%d]=%e ", cidx, col, csst->mass_jac[cidx * csst->mass_jac_cols + col]);
                    }
                    printf("\n");
                }
            }
            #endif
            
            for (int cidx = 0; cidx < spec->num_components; ++cidx) {
                for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                    for (int j = 0; j < num_phase_dof_for_csst; ++j) {
                         if (cidx < pr->num_elements) { // Ensure we are using valid mass_jac entries
                            // mass_jac is in Workspace format, not Model format!
                            // We need to access the site fraction columns which start at spec->num_statevars
                            double mass_jac_val = csst->mass_jac[cidx * csst->mass_jac_cols + (spec->num_statevars + j)];
                            double e_matrix_val = csst->full_e_matrix[i * csst->full_e_matrix_dim + j];
                            csst->c_component[cidx * csst->c_component_cols + i] += mass_jac_val * e_matrix_val;
                            
                            // DEBUG: Print calculation for BOTH phases
                            #ifdef VERBOSE_DEBUG
                            if (thread_id == 0 && iteration < 2 && cidx < 2 && i < 2 && j < 2) {
                                printf("  Phase %d: c_component[%d,%d] += mass_jac[%d,%d]=%e * e_matrix[%d,%d]=%e = %e\n",
                                       idx, cidx, i, cidx, spec->num_statevars + j, mass_jac_val, i, j, e_matrix_val,
                                       mass_jac_val * e_matrix_val);
                            }
                            #endif
                         }
                    }
                }
            }
            
            // DEBUG: Print c_component matrix for BOTH phases
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && iteration < 5) {
                printf("[GPU C_COMPONENT] Phase %d matrix (iter %d):\n", idx, iteration);
                printf("  phase_amt = %.15e\n", phase_amt[idx]);
                printf("  phase_dof = %d, num_elements = %d\n", pr->phase_dof, pr->num_elements);
                for (int cidx = 0; cidx < 2; cidx++) {
                    printf("  Component %d: ", cidx);
                    for (int i = 0; i < 2; i++) {
                        printf("%e ", csst->c_component[cidx * csst->c_component_cols + i]);
                    }
                    printf("\n");
                }
                
                // Check if c_component is all zeros
                bool all_zeros = true;
                for (int cidx = 0; cidx < spec->num_components && cidx < pr->num_elements; cidx++) {
                    for (int i = 0; i < pr->phase_dof; i++) {
                        if (fabs(csst->c_component[cidx * csst->c_component_cols + i]) > 1e-15) {
                            all_zeros = false;
                            break;
                        }
                    }
                }
                if (all_zeros) {
                    printf("  WARNING: c_component is all zeros!\n");
                }
            }
            #endif
            for (int cidx = 0; cidx < spec->num_components; ++cidx) {
                for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                    double mu_c_sum = 0.0;
                    for (int j_chem = 0; j_chem < spec->num_components; ++j_chem) {
                        mu_c_sum += csst->c_component[j_chem * csst->c_component_cols + i] * chemical_potentials[j_chem];
                    }
                    if (cidx < pr->num_elements) {
                        delta_ms[idx * delta_ms_cols + cidx] += // Use delta_ms_cols from SystemState
                            csst->mass_jac[cidx * csst->mass_jac_cols + (spec->num_statevars + i)] *
                            (mu_c_sum + csst->c_G[i]);
                    }
                }
            }
            for (int cidx = 0; cidx < spec->num_components; ++cidx) {
                 if (cidx < pr->num_elements) {
                    csst->moles_normalization += csst->masses[cidx];
                    // moles_normalization_grad should be in Workspace format
                    // since mass_jac is in Workspace format
                    for (int i_dof = 0; i_dof < csst->mass_jac_cols; ++i_dof) {
                        csst->moles_normalization_grad[i_dof] += csst->mass_jac[cidx * csst->mass_jac_cols + i_dof];
                    }
                 }
            }
            
            // DEBUG: Print moles_normalization for each phase
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && iteration < 2) {
                printf("[GPU MOLES_NORM] Phase %d: moles_normalization = %e (should be ~%e for %d sublattices)\n", 
                       idx, csst->moles_normalization, 
                       pr->phase_dof > 2 ? 20.0 : 1.0,  // Rough estimate
                       pr->phase_dof > 2 ? 2 : 1);
            }
            #endif
        }
        delta_ms_rows = num_compsets; // Update after loop in case num_compsets changed (though not in recompute)
        phase_compositions_rows = num_compsets;
    }

    __device__ void driving_forces(SystemSpecification* spec, double* out_driving_forces, int num_out_df_max_cap) {
    for (int i = 0; i < num_out_df_max_cap; ++i) out_driving_forces[i] = 0.0;

    // REMOVED: current_dof_for_phase array - now using model_dof_for_calcs created locally where needed

    for (int idx = 0; idx < num_compsets; ++idx) {
            if (idx >= num_out_df_max_cap) continue;
            CompositionSet* compset = &compsets[idx];
            if (compset->phase_record == nullptr) {
                continue;
            }
            const PhaseRecord* pr = compset->phase_record;

            // REMOVED: Old code that created current_dof_for_phase incorrectly
            // Now we create model_dof_for_calcs properly from workspace DOF when needed

            // CPU driving_forces (minimizer.pyx:1036-1051) evaluates mass_obj and obj
            // FRESH from the current dof for every compset, including metastable ones.
            // phase_compositions is only refreshed by recompute() for phases with
            // phase_amt > 1e-10, so it can be stale here; recompute formula moles
            // directly from dof instead.
            double formulamoles_df[MAX_COMPONENTS];
            for (int comp_idx = 0; comp_idx < MAX_COMPONENTS; ++comp_idx) formulamoles_df[comp_idx] = 0.0;
            if (pr->formulamole_obj != nullptr) {
                pr->formulamole_obj(formulamoles_df, compset->dof);
            }
            double atoms_per_formula_unit = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                 atoms_per_formula_unit += formulamoles_df[comp_idx];
            }
            if (fabs(atoms_per_formula_unit) < 1e-12) atoms_per_formula_unit = 1.0;

            // The generated pr->obj is Model.GM = energy PER MOLE OF ATOMS
            // (gpu_codegen.py:2215; formulaobj is the per-formula-unit Model.G).
            // Dividing by atoms_per_formula_unit here double-divided — invisible
            // for 1-atom/f.u. phases but ruinous for e.g. GAMMA_D83 (13 atoms/f.u.).
            double gm_per_atom = pr->obj(compset->dof);

            out_driving_forces[idx] = 0.0; // Initialize for current phase
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                out_driving_forces[idx] += chemical_potentials[comp_idx] *
                                           (formulamoles_df[comp_idx] / atoms_per_formula_unit);
            }
            out_driving_forces[idx] -= gm_per_atom;
        }
    }

    __device__ void increment_phase_metastability_counters() {
        for (int idx = 0; idx < num_compsets; ++idx) {
            bool is_free_stable = false;
            for (int i = 0; i < num_free_stable_compsets; ++i) {
                if (free_stable_compset_indices[i] == idx) {
                    is_free_stable = true;
                    break;
                }
            }
            if (is_free_stable || (idx < num_compsets && compsets[idx].fixed) ) {
                metastable_phase_iterations[idx] = 0;
            } else if (idx < num_compsets) { // Only increment for valid compsets
                metastable_phase_iterations[idx]++;
                #ifdef DEBUG_METASTABLE
                if (condition_idx == 0 && metastable_phase_iterations[idx] > 0) {
                    printf("[GPU] Condition 0: Phase %d now metastable for %d iterations\n", 
                           idx, metastable_phase_iterations[idx]);
                }
                #endif
            }
        }
    }
} SystemState;

// Forward declaration for identify_candidate_phase_to_add function from eqsolver.h (needs to be after SystemState)
__device__ bool identify_candidate_phase_to_add(
    int* candidate_phase_grid_idx,
    double* candidate_driving_force,
    const SystemState* current_sys_state,
    const SystemSpecification* spec,
    const DeviceGrid* grid_data,
    const DevicePhaseData* phase_data,
    const double* state_variables_values,
    double minimum_df,
    const CompositionSet* removed_compsets,
    int num_removed_compsets);

// Function definitions (previously declared)

__device__ void compute_phase_matrix(double* phase_matrix_out, const double* hess_in,
                                    const double* cons_jac_tmp_in,
                                    const CompositionSet& compset_ref, int num_statevars_val,
                                    // chemical_potentials not used in C version from prompt
                                    const double* phase_dof_site_fracs) {
    // Based on the C version in the original minimizer.h prompt
    // Assumes phase_matrix_out is pre-sized to (phase_dof + num_internal_cons) x (phase_dof + num_internal_cons)
    // num_phase_local_conditions is zero, so it's omitted from dimensions.

    if (compset_ref.phase_record == nullptr) return; // Safety check
    const PhaseRecord* pr = compset_ref.phase_record;

    int phase_dof_val = pr->phase_dof;
    int num_internal_cons_val = pr->num_internal_cons;
    int current_phase_matrix_dim = phase_dof_val + num_internal_cons_val;
    int hess_total_dim = num_statevars_val + phase_dof_val; // num_cols of hess
    int cons_jac_total_dim = num_statevars_val + phase_dof_val; // num_cols of cons_jac_tmp

    // Fill phase matrix from Hessian (diagonal blocks)
    for (int i = 0; i < phase_dof_val; i++) {
        for (int j = 0; j < phase_dof_val; j++) {
            // phase_matrix[i][j] = hess[num_statevars+i][num_statevars+j]
            phase_matrix_out[i * current_phase_matrix_dim + j] =
                hess_in[(num_statevars_val + i) * hess_total_dim + (num_statevars_val + j)];
        }
    }

    // Fill phase matrix from constraint Jacobian (off-diagonal blocks)
    for (int i = 0; i < num_internal_cons_val; i++) {
        for (int j = 0; j < phase_dof_val; j++) {
            // Upper right block: phase_matrix[phase_dof+i][j] (row-major index)
            phase_matrix_out[(phase_dof_val + i) * current_phase_matrix_dim + j] =
                cons_jac_tmp_in[i * cons_jac_total_dim + (num_statevars_val + j)];

            // Lower left block: phase_matrix[j][phase_dof+i] (row-major index)
            phase_matrix_out[j * current_phase_matrix_dim + (phase_dof_val + i)] =
                cons_jac_tmp_in[i * cons_jac_total_dim + (num_statevars_val + j)];
        }
    }
     // Zero out the bottom-right block corresponding to (constraint, constraint) interactions if it's not filled by above
    for (int i = 0; i < num_internal_cons_val; ++i) {
        for (int j = 0; j < num_internal_cons_val; ++j) {
            // This block should be zero in the standard formulation if not explicitly calculated.
            // phase_matrix[phase_dof+i][phase_dof+j]
            if (i!=j) { // Off-diagonal typically zero unless hessian of constraints considered.
                 // phase_matrix_out[(phase_dof_val + i) * current_phase_matrix_dim + (phase_dof_val + j)] = 0.0;
            }
            // Diagonal (lambda_i, lambda_i) terms are also typically zero.
            // If they are filled from Hessian, this part is not needed. Sundman 2015 Eq 41 has this block as 0.
             phase_matrix_out[(phase_dof_val + i) * current_phase_matrix_dim + (phase_dof_val + j)] = 0.0; // Explicitly zero for Sundman formulation
        }
    }
}


// Remaining function definitions (solve_state, advance_state, etc.)

// with the "no phase local conditions" constraint, which mainly affects dimensions.


__device__ void write_row_stable_phase(double* out_row, double* out_rhs,
                                     const int* free_chemical_potential_indices, int num_free_chemical_potentials,
                                     const int* free_stable_compset_indices, int num_free_stable_compsets, // These are indices into the main compset array
                                     const int* free_statevar_indices, int num_free_statevars,
                                     const int* fixed_chemical_potential_indices, int num_fixed_chemical_potentials,
                                     const double* current_chemical_potentials, // Renamed from state->chemical_potentials
                                     const double* masses_for_compset, // csst->masses
                                     const double* grad_for_compset,   // csst->grad
                                     double energy_for_compset) {     // csst->energy
    // DEBUG: Print the row being written
    #ifdef VERBOSE_DEBUG
    printf("[GPU EQUILIBRIUM MATRIX DEBUG] Writing row for stable phase:\n");
    printf("  Energy: %e\n", energy_for_compset);
    printf("  Masses: ");
    for (int debug_idx = 0; debug_idx < MAX_COMPONENTS; debug_idx++) {
        if (masses_for_compset[debug_idx] != 0.0) {
            printf("%e ", masses_for_compset[debug_idx]);
        }
    }
    printf("\n");
    #endif
    
    int free_variable_column_offset = 0;
    int chempot_idx, statevar_idx, i;

    // Write masses for ALL components in free_chemical_potential_indices
    // CPU includes ALL non-VA components as free chemical potentials, even when
    // mole fractions are prescribed. The constraints are handled separately.
    // This matches the CPU behavior of having columns for all non-VA components.
    for (i = 0; i < num_free_chemical_potentials; i++) {
        chempot_idx = free_chemical_potential_indices[i];
        // Write mass for each free chemical potential component
        out_row[free_variable_column_offset + i] = masses_for_compset[chempot_idx];
    }
    free_variable_column_offset += num_free_chemical_potentials;

    // Free stable composition sets part (columns for d(NP_j))
    // The value is 1 if this row corresponds to NP_j, 0 otherwise.
    // This function is called PER ROW. The row index (stable_idx or fixed_idx from fill_equilibrium_system)
    // implies WHICH compset this row is for. The columns for d(NP) are ordered by `free_stable_compset_indices`.
    // This function doesn't know which *row* it's writing, only the *data for a phase*.
    // The logic in `fill_equilibrium_system` needs to handle placing 1s correctly.
    // The pyx version's write_row_stable_phase does NOT fill the d(NP) part of its own row with 1.
    // That's handled by the structure of Ax=b where x includes delta_NP.
    // So these columns are 0 from the perspective of *this specific phase's contribution via chemical potentials/statevars*.
    // The columns corresponding to `delta phase_amts` are handled by the overall system matrix structure.
    // So, for this row, the contribution to d(NP_k) columns (where k != this phase) is 0.
    // The contribution to its OWN d(NP) column comes from the N=1 constraint, not here.
    // This means this part of the row is all zeros.
    for (i = 0; i < num_free_stable_compsets; ++i) {
        out_row[free_variable_column_offset + i] = 0.0; // Default to 0
    }
    free_variable_column_offset += num_free_stable_compsets;


    for (i = 0; i < num_free_statevars; i++) {
        statevar_idx = free_statevar_indices[i]; // This is the workspace index of the free state variable
        // Use workspace indices directly, just like CPU code does!
        // The gradient array is already populated with workspace indices from formulagrad
        out_row[free_variable_column_offset + i] = -grad_for_compset[statevar_idx];
    }

    out_rhs[0] = energy_for_compset;

    // Subtract fixed chemical potentials from each phase RHS
    // This matches the CPU code in minimizer.pyx line 122-124
    for (i = 0; i < num_fixed_chemical_potentials; i++) {
        chempot_idx = fixed_chemical_potential_indices[i];
        out_rhs[0] -= masses_for_compset[chempot_idx] * current_chemical_potentials[chempot_idx];
    }
    
    // DEBUG: Print the final row values and RHS
    #ifdef VERBOSE_DEBUG
    printf("  Row values (first %d): ", num_free_chemical_potentials + num_free_stable_compsets + num_free_statevars);
    for (int debug_idx = 0; debug_idx < num_free_chemical_potentials + num_free_stable_compsets + num_free_statevars; debug_idx++) {
        printf("%e ", out_row[debug_idx]);
    }
    printf("\n");
    printf("  RHS: %e\n", out_rhs[0]);
    #endif
}

__device__ void write_row_fixed_mole_fraction(double* out_row, double* out_rhs,
                                            int component_idx_of_constraint, // The component X_i in sum Ci*Xi = R
                                            const int* free_chemical_potential_indices, int num_free_chemical_potentials,
                                            const int* free_stable_compset_indices, int num_free_stable_compsets, // Global indices
                                            const int* free_statevar_indices, int num_free_statevars,
                                            const int* fixed_chemical_potential_indices, int num_fixed_chemical_potentials,
                                            const double* current_chemical_potentials_sys, // System's chemical potentials
                                            const double* system_mole_fractions_sys, // System's overall mole_fractions
                                            double current_system_amount_sys,
                                            const double* mass_jac_cs, int mass_jac_cols_cs, // Compset's mass_jac
                                            const double* c_component_cs, int c_component_cols_cs, // Compset's c_component
                                            const double* c_statevars_cs, int c_statevars_cols_cs, // Compset's c_statevars
                                            const double* c_G_cs, int c_G_length_cs,             // Compset's c_G
                                            const double* masses_cs,                // Compset's masses (per formula)
                                            double moles_normalization_cs,
                                            const double* moles_normalization_grad_cs,
                                            const double* phase_amt_sys, // System's phase_amt array
                                            int compset_original_idx_sys, // Index of the current compset in system arrays
                                            double prefactor_for_this_component) { // Coefficient from prescribed_mole_fraction_coefficients

    if (fabs(prefactor_for_this_component) < 1e-12) return; // No contribution if prefactor is zero

    int free_variable_column_offset = 0;
    // num_statevars_proxy should be the number of state variables in the system (e.g. spec->num_statevars)
    // This is used for indexing mass_jac and moles_normalization_grad.
    // c_statevars_cols_cs is also this value.
    int num_system_statevars = c_statevars_cols_cs;


    // 2a. This component row: free chemical potentials
    for (int i = 0; i < num_free_chemical_potentials; i++) {
        int free_chempot_global_idx = free_chemical_potential_indices[i];
        double term1 = 0.0;
        double term2 = 0.0;
        for (int j = 0; j < c_component_cols_cs; j++) { // j is index over phase_dof
            // Both mass_jac and moles_normalization_grad are in Workspace format!
            // In Workspace format: [N, P, T, Y1, Y2, ...] so site fractions start at num_system_statevars
            double mass_jac_val = mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)];
            double c_comp_val = c_component_cs[free_chempot_global_idx * c_component_cols_cs + j];
            term1 += mass_jac_val * c_comp_val;
            
            // (-system_mole_fractions_sys[comp_idx_of_constraint] * moles_normalization_grad_cs[num_sv+j]) * c_component_cs[free_chempot_glob_idx, j]
            double mole_norm_grad_val = moles_normalization_grad_cs[num_system_statevars + j];
            term2 += (-system_mole_fractions_sys[component_idx_of_constraint] * mole_norm_grad_val) * c_comp_val;
            
            // DEBUG: Print intermediate values for both phases, component 1, chempot 0
            #ifdef VERBOSE_DEBUG
            if (component_idx_of_constraint == 1 && i == 0 && j < 2) {
                printf("  Phase %d, j=%d: mass_jac[%d,%d]=%e, c_component[%d,%d]=%e\n", 
                       compset_original_idx_sys, j, component_idx_of_constraint, num_system_statevars + j, 
                       mass_jac_val, free_chempot_global_idx, j, c_comp_val);
                printf("       mole_norm_grad[%d]=%e, X[%d]=%e\n",
                       num_system_statevars + j, mole_norm_grad_val,
                       component_idx_of_constraint, system_mole_fractions_sys[component_idx_of_constraint]);
                printf("       term1 contribution: %e, term2 contribution: %e\n",
                       mass_jac_val * c_comp_val,
                       (-system_mole_fractions_sys[component_idx_of_constraint] * mole_norm_grad_val) * c_comp_val);
            }
            #endif
        }
        if (fabs(current_system_amount_sys)>1e-12) {
             double contribution = prefactor_for_this_component *
                (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (term1 + term2);
             out_row[free_variable_column_offset + i] += contribution;
             
             // DEBUG: Show exact contribution for mole fraction constraint
             #ifdef VERBOSE_DEBUG
             if (component_idx_of_constraint == 1 && i < 2) {
                 printf("[GPU MOLE FRAC] Phase %d adds %e to col %d (chem pot %d)\n", 
                        compset_original_idx_sys, contribution, free_variable_column_offset + i, i);
             }
             #endif
        }
        
        // DEBUG: Print values for both phases and first constraint
        #ifdef VERBOSE_DEBUG
        if (component_idx_of_constraint == 1 && i == 0) {
            printf("[GPU MOLE FRAC DEBUG] Phase %d, Component 1, ChemPot 0:\n", compset_original_idx_sys);
            printf("  prefactor=%e, phase_amt=%e, sys_amt=%e\n", 
                   prefactor_for_this_component, phase_amt_sys[compset_original_idx_sys], current_system_amount_sys);
            printf("  free_chempot_global_idx=%d, num_free_chemical_potentials=%d\n", 
                   free_chempot_global_idx, num_free_chemical_potentials);
            printf("  c_component_cols_cs=%d, mass_jac_cols_cs=%d, num_system_statevars=%d\n",
                   c_component_cols_cs, mass_jac_cols_cs, num_system_statevars);
            printf("  term1=%e, term2=%e, total=%e\n", term1, term2, term1+term2);
            printf("  Result added to out_row[%d]: %e\n", free_variable_column_offset + i,
                   prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (term1 + term2));
        }
        #endif
    }
    free_variable_column_offset += num_free_chemical_potentials;

    // 2a. This component row: free stable composition sets
    for (int i = 0; i < num_free_stable_compsets; i++) {
        int current_free_compset_original_sys_idx = free_stable_compset_indices[i];
        if (current_free_compset_original_sys_idx == compset_original_idx_sys) { // If this column is for the current phase
             if (fabs(current_system_amount_sys)>1e-12) {
                double coeff_value = (masses_cs[component_idx_of_constraint] - system_mole_fractions_sys[component_idx_of_constraint] * moles_normalization_cs);
                double contribution = prefactor_for_this_component * (1.0 / current_system_amount_sys) * coeff_value;
                
                #ifdef VERBOSE_DEBUG
                // Print debug info for mole fraction constraint calculations
                if (component_idx_of_constraint == 1) {
                    printf("[GPU MOLE FRAC COEFF] Phase %d, Component 1:\n", compset_original_idx_sys);
                    printf("  masses[1] = %.15e\n", masses_cs[component_idx_of_constraint]);
                    printf("  system_mole_fractions[1] = %.15e\n", system_mole_fractions_sys[component_idx_of_constraint]);
                    printf("  moles_normalization = %.15e\n", moles_normalization_cs);
                    printf("  coeff = masses[1] - sys_mole_frac[1] * moles_norm = %.15e - %.15e * %.15e = %.15e\n",
                           masses_cs[component_idx_of_constraint], 
                           system_mole_fractions_sys[component_idx_of_constraint],
                           moles_normalization_cs,
                           coeff_value);
                    printf("  contribution to matrix = %.15e * (1.0 / %.15e) * %.15e = %.15e\n",
                           prefactor_for_this_component, current_system_amount_sys, coeff_value, contribution);
                    printf("  out_row[%d] += %.15e\n", free_variable_column_offset + i, contribution);
                }
                #endif
                
                out_row[free_variable_column_offset + i] += contribution;
             }
        }
    }
    free_variable_column_offset += num_free_stable_compsets;

    // 2a. This component row: free state variables
    for (int i = 0; i < num_free_statevars; i++) {
        int free_statevar_global_idx = free_statevar_indices[i];
        double term1 = 0.0;
        double term2 = 0.0;
        for (int j = 0; j < c_statevars_cols_cs; j++) { // j is index over phase_dof
            // Use workspace indexing for mass_jac and moles_normalization_grad
            term1 += mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)] *
                     c_statevars_cs[j * c_statevars_cols_cs + free_statevar_global_idx];
            // (-system_mole_fractions_sys[comp_idx_of_constraint] * moles_normalization_grad_cs[num_sv+j]) * c_statevars_cs[j, free_statevar_glob_idx]
            term2 += (-system_mole_fractions_sys[component_idx_of_constraint] * moles_normalization_grad_cs[num_system_statevars + j]) *
                     c_statevars_cs[j * c_statevars_cols_cs + free_statevar_global_idx];
        }
         if (fabs(current_system_amount_sys)>1e-12) {
            out_row[free_variable_column_offset + i] += prefactor_for_this_component *
                (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (term1 + term2);
         }
    }

    // 3. Contributions to RHS (from c_G_cs)
    double rhs_term1 = 0.0;
    double rhs_term2 = 0.0;
    
    // DEBUG: Print c_G values for iteration 0
    #ifdef VERBOSE_DEBUG
    if (component_idx_of_constraint == 1 && compset_original_idx_sys < 3) {
        printf("[GPU c_G DEBUG] Phase %d, constraint component %d:\n", 
               compset_original_idx_sys, component_idx_of_constraint);
        printf("  c_G values: ");
        for (int j = 0; j < c_G_length_cs && j < 5; j++) {
            printf("[%d]=%e ", j, c_G_cs[j]);
        }
        printf("\n");
        printf("  mass_jac[%d,3+j]: ", component_idx_of_constraint);
        for (int j = 0; j < c_G_length_cs && j < 5; j++) {
            printf("%e ", mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)]);
        }
        printf("\n");
    }
    #endif
    
    for (int j = 0; j < c_G_length_cs; j++) { // j is index over phase_dof
        // Use workspace indexing for mass_jac and moles_normalization_grad
        rhs_term1 += mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)] * c_G_cs[j];
        rhs_term2 += (-system_mole_fractions_sys[component_idx_of_constraint] * moles_normalization_grad_cs[num_system_statevars + j]) * c_G_cs[j];
    }
    if (fabs(current_system_amount_sys)>1e-12) {
        out_rhs[0] += -prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (rhs_term1 + rhs_term2);
        
        // DEBUG: Print RHS contribution
        #ifdef VERBOSE_DEBUG
        if (component_idx_of_constraint == 1) {
            printf("[GPU RHS] Phase %d adds %e to constraint %d RHS (term1=%e, term2=%e)\n",
                   compset_original_idx_sys, 
                   -prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (rhs_term1 + rhs_term2),
                   component_idx_of_constraint, rhs_term1, rhs_term2);
        }
        #endif
    }
    
    // DEBUG: Print RHS contribution for both phases and first constraint
    #ifdef VERBOSE_DEBUG
    if (component_idx_of_constraint == 1) {
        printf("[GPU MOLE FRAC RHS DEBUG] Phase %d, Component 1:\n", compset_original_idx_sys);
        printf("  c_G_length=%d, c_G[0]=%e, c_G[1]=%e\n", c_G_length_cs, 
               c_G_length_cs > 0 ? c_G_cs[0] : 0.0, c_G_length_cs > 1 ? c_G_cs[1] : 0.0);
        printf("  rhs_term1=%e, rhs_term2=%e\n", rhs_term1, rhs_term2);
        printf("  phase_amt=%e, system_amt=%e, prefactor=%e\n",
               phase_amt_sys[compset_original_idx_sys], current_system_amount_sys, prefactor_for_this_component);
        printf("  RHS contribution: %e\n", 
               -prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (rhs_term1 + rhs_term2));
        printf("  Formula: -%.3f * (%.3f / %.3f) * (%.3f + %.3f) = %.6f\n",
               prefactor_for_this_component, phase_amt_sys[compset_original_idx_sys], current_system_amount_sys,
               rhs_term1, rhs_term2,
               -prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (rhs_term1 + rhs_term2));
    }
    #endif

    // 4. Subtract fixed chemical potentials from fixed component RHS
    for (int i = 0; i < num_fixed_chemical_potentials; i++) {
        int fixed_chempot_global_idx = fixed_chemical_potential_indices[i];
        double sub_term1 = 0.0;
        double sub_term2 = 0.0;
        for (int j = 0; j < c_component_cols_cs; j++) { // j is index over phase_dof
            // Use workspace indexing for mass_jac and moles_normalization_grad
            sub_term1 += mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)] *
                         c_component_cs[fixed_chempot_global_idx * c_component_cols_cs + j];
            sub_term2 += (-system_mole_fractions_sys[component_idx_of_constraint] * moles_normalization_grad_cs[num_system_statevars + j]) *
                         c_component_cs[fixed_chempot_global_idx * c_component_cols_cs + j];
        }
        if (fabs(current_system_amount_sys)>1e-12) {
            out_rhs[0] -= prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) *
                          current_chemical_potentials_sys[fixed_chempot_global_idx] * (sub_term1 + sub_term2);
        }
    }
}


// --- Definitions for the rest of the __device__ functions ---

// fill_equilibrium_system, check_convergence, pre_solve_hook, post_solve_hook,
// solve_state, advance_state, remove_and_consolidate_phases, change_phases, run_loop

// Phase-local conditions are always absent in this port, which simplifies
// CompsetState's phase_matrix_dim (and MAX_PHASE_MATRIX_DIM); the functions
// below rely on the dimensions carried by CompsetState and PhaseRecord.

// Implementation of write_row_fixed_mole_amount function
__device__ void write_row_fixed_mole_amount(double* out_row, double* out_rhs,
                                            int component_idx, 
                                            const int* free_chemical_potential_indices, int num_free_chemical_potentials,
                                            const int* free_stable_compset_indices, int num_free_stable_compsets,
                                            const int* free_statevar_indices, int num_free_statevars,
                                            const int* fixed_chemical_potential_indices, int num_fixed_chemical_potentials,
                                            const double* current_chemical_potentials_sys,
                                            const double* mass_jac_cs, int mass_jac_cols_cs,
                                            const double* c_component_cs, int c_component_cols_cs,
                                            const double* c_statevars_cs, int c_statevars_cols_cs,
                                            const double* c_G_cs, int c_G_length_cs,
                                            const double* masses_cs,
                                            double moles_normalization_cs,
                                            const double* moles_normalization_grad_cs,
                                            const double* phase_amt_sys,
                                            int compset_original_idx_sys) {
    
    // This function writes one row of the equilibrium matrix for mass balance constraint
    // Matches CPU's write_row_fixed_mole_amount logic
    
    int free_variable_column_offset = 0;
    int num_system_statevars = c_statevars_cols_cs;
    
    // FIX: CPU code does NOT normalize by moles_normalization (sum of site ratios)
    // Setting normalization_factor to 1.0 to match CPU behavior
    double normalization_factor = 1.0;
    
    // DEBUG: Print normalization factor for each phase
    #ifdef VERBOSE_DEBUG
    if (component_idx == 0 && phase_amt_sys[compset_original_idx_sys] > 1e-10) {
        printf("[GPU SYSTEM AMOUNT] Phase %d: moles_norm=%e, using factor=%e\n", 
               compset_original_idx_sys, moles_normalization_cs, normalization_factor);
    }
    #endif
    
    // 2a. This component row: free chemical potentials
    for (int i = 0; i < num_free_chemical_potentials; ++i) {
        int chempot_idx = free_chemical_potential_indices[i];
        for (int j = 0; j < c_component_cols_cs; ++j) {  // j is phase_dof index
            // out_row[offset + i] += phase_amt * mass_jac[comp_idx, num_sv+j] * c_component[chempot_idx, j] / moles_norm
            out_row[free_variable_column_offset + i] += 
                (phase_amt_sys[compset_original_idx_sys] / normalization_factor) * 
                mass_jac_cs[component_idx * mass_jac_cols_cs + num_system_statevars + j] * 
                c_component_cs[chempot_idx * c_component_cols_cs + j];
        }
    }
    free_variable_column_offset += num_free_chemical_potentials;
    
    // 2a. This component row: free stable composition sets
    for (int i = 0; i < num_free_stable_compsets; ++i) {
        int compset_idx = free_stable_compset_indices[i];
        // Only fill this out if the current idx is equal to a free composition set
        if (compset_idx == compset_original_idx_sys) {
            // For fixed_mole_amount, the coefficient is the mass of this component normalized by moles_normalization
            out_row[free_variable_column_offset + i] += masses_cs[component_idx] / normalization_factor;
        }
    }
    free_variable_column_offset += num_free_stable_compsets;
    
    // 2a. This component row: free state variables
    for (int i = 0; i < num_free_statevars; ++i) {
        int statevar_idx = free_statevar_indices[i];
        for (int j = 0; j < c_statevars_cols_cs; ++j) {  // j is phase_dof index
            // out_row[offset + i] += phase_amt * mass_jac[comp_idx, num_sv+j] * c_statevars[j, statevar_idx] / moles_norm
            out_row[free_variable_column_offset + i] += 
                (phase_amt_sys[compset_original_idx_sys] / normalization_factor) * 
                mass_jac_cs[component_idx * mass_jac_cols_cs + num_system_statevars + j] * 
                c_statevars_cs[j * c_statevars_cols_cs + statevar_idx];
        }
    }
    
    // 3. RHS contribution from c_G (also needs normalization)
    for (int j = 0; j < c_G_length_cs; ++j) {
        *out_rhs += -(phase_amt_sys[compset_original_idx_sys] / normalization_factor) * 
                    mass_jac_cs[component_idx * mass_jac_cols_cs + num_system_statevars + j] * 
                    c_G_cs[j];
    }
    
    // 4. Subtract fixed chemical potentials from RHS
    for (int i = 0; i < num_fixed_chemical_potentials; ++i) {
        int chempot_idx = fixed_chemical_potential_indices[i];
        // 6. Subtract fixed chemical potentials from the N=1 row
        for (int j = 0; j < c_component_cols_cs; ++j) {
            *out_rhs -= (phase_amt_sys[compset_original_idx_sys] / normalization_factor) * 
                        current_chemical_potentials_sys[chempot_idx] * 
                        mass_jac_cs[component_idx * mass_jac_cols_cs + num_system_statevars + j] * 
                        c_component_cs[chempot_idx * c_component_cols_cs + j];
        }
    }
}


// It is crucial that the C PhaseRecord struct and its associated function pointers
// (e.g., for formulamole_obj, formulamole_grad) are implemented in a way that
// is consistent with how they are called (e.g., if they operate per-component or fill arrays for all components).
// Note: this compute_phase_matrix does not use chemical potentials, unlike
// the pyx counterpart's delta_y calculation.

// Re-inserting and checking fill_equilibrium_system for sanity with the new context:
__device__ void fill_equilibrium_system(double* equilibrium_matrix, int equilibrium_matrix_cols,
                                      double* equilibrium_rhs,
                                      SystemSpecification* spec, SystemState* state) {
    // SEGMENT 27: CONSTRUCT EQUILIBRIUM SYSTEM
    gpu_debug_log(27, "Construct equilibrium system", state->condition_idx);
    
    int stable_idx, compset_original_idx, current_component_idx, fixed_cs_idx;
    CompositionSet* current_compset;
    CompsetState* current_cs_state;
    int num_total_components = spec->num_components;
    int num_free_stable_phases = state->num_free_stable_compsets;
    int num_fixed_stable_cs = spec->num_fixed_stable_compsets;
    int num_fixed_mole_frac_conds = spec->num_prescribed_mole_fraction_conditions;
    double prefactor;

    // Add +1 back to match CPU matrix dimensions exactly
    // CPU DOES include a system amount constraint row (with [1,1,1] for phase amounts)
    int total_rows = num_free_stable_phases + num_fixed_stable_cs + num_fixed_mole_frac_conds + 1;
    if (state->condition_idx < 3) {
        gpu_debug_log_value("matrix_dimensions", (double)(total_rows * equilibrium_matrix_cols));
        gpu_debug_log_value("num_rows", (double)total_rows);
        gpu_debug_log_value("num_cols", (double)equilibrium_matrix_cols);
    }
    
    // DEBUG: Print equilibrium matrix construction details
    #ifdef VERBOSE_DEBUG
    if (state->condition_idx == 0 && (state->iteration == 0 || num_free_stable_phases == 1)) {
        printf("[GPU EQUILIBRIUM MATRIX] Filling equilibrium system at iteration %d\n", state->iteration);
        printf("  num_free_stable_phases: %d\n", num_free_stable_phases);
        printf("  num_fixed_stable_cs: %d\n", num_fixed_stable_cs);
        printf("  num_fixed_mole_frac_conds: %d\n", num_fixed_mole_frac_conds);
        printf("  total_rows: %d\n", total_rows);
        printf("  equilibrium_matrix_cols: %d\n", equilibrium_matrix_cols);
        
        // Special debug for single phase case
        if (num_free_stable_phases == 1) {
            printf("[GPU DEBUG] Single phase detected after consolidation\n");
            int phase_idx = state->free_stable_compset_indices[0];
            CompositionSet* cs = &state->compsets[phase_idx];
            printf("  Phase %d site fractions: [", phase_idx);
            for (int i = 0; i < cs->phase_record->phase_dof; i++) {
                printf("%.10f", cs->dof[spec->num_statevars + i]);
                if (i < cs->phase_record->phase_dof - 1) printf(", ");
            }
            printf("]\n");
            printf("  Target X[1] = %.10f\n", spec->prescribed_mole_fraction_rhs[0]);
            printf("  Current X[1] = %.10f\n", state->phase_compositions[phase_idx * MAX_COMPONENTS + 1]);
            printf("  Mass residual = %.10e\n", state->mass_residual);
        }
    }
    #endif
    
    for(int i=0; i < total_rows * equilibrium_matrix_cols; ++i) equilibrium_matrix[i] = 0.0;
    for(int i=0; i < total_rows; ++i) equilibrium_rhs[i] = 0.0;
    
    // Do NOT initialize RHS to target values; phase contributions build the constraint equation
    // The RHS should start from 0 and accumulate phase contributions, then have residual subtracted later
    // The target value (0.5) is handled in the residual calculation, not pre-loaded into RHS

    for (stable_idx = 0; stable_idx < num_free_stable_phases; stable_idx++) {
        compset_original_idx = state->free_stable_compset_indices[stable_idx];
        if (compset_original_idx < 0 || compset_original_idx >= state->num_compsets) continue; // Boundary check
        current_compset = &state->compsets[compset_original_idx];
        current_cs_state = &state->cs_states[compset_original_idx];
        if (current_compset->phase_record == nullptr) continue;

        write_row_stable_phase(
            &equilibrium_matrix[stable_idx * equilibrium_matrix_cols],
            &equilibrium_rhs[stable_idx],
            spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
            state->free_stable_compset_indices, state->num_free_stable_compsets,
            spec->free_statevar_indices, spec->num_free_statevars,
            spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
            state->chemical_potentials, current_cs_state->masses,
            current_cs_state->grad, current_cs_state->energy);
    }

    int current_row_offset = num_free_stable_phases;
    for (fixed_cs_idx = 0; fixed_cs_idx < num_fixed_stable_cs; fixed_cs_idx++) {
        compset_original_idx = spec->fixed_stable_compset_indices[fixed_cs_idx];
         if (compset_original_idx < 0 || compset_original_idx >= state->num_compsets) continue;
        current_compset = &state->compsets[compset_original_idx];
        current_cs_state = &state->cs_states[compset_original_idx];
        if (current_compset->phase_record == nullptr) continue;

        write_row_stable_phase(
            &equilibrium_matrix[(current_row_offset + fixed_cs_idx) * equilibrium_matrix_cols],
            &equilibrium_rhs[current_row_offset + fixed_cs_idx],
            spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
            state->free_stable_compset_indices, state->num_free_stable_compsets,
            spec->free_statevar_indices, spec->num_free_statevars,
            spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
            state->chemical_potentials, current_cs_state->masses,
            current_cs_state->grad, current_cs_state->energy);
    }
    current_row_offset += num_fixed_stable_cs;

    // Loop over free stable phases only (matching CPU behavior)
    for (int free_idx = 0; free_idx < state->num_free_stable_compsets; free_idx++) {
        stable_idx = state->free_stable_compset_indices[free_idx];
        compset_original_idx = stable_idx; // The index in the main compsets array
        current_compset = &state->compsets[compset_original_idx];
        current_cs_state = &state->cs_states[compset_original_idx];
        if (current_compset->phase_record == nullptr) continue;


        for (int mole_frac_cond_row_idx = 0; mole_frac_cond_row_idx < num_fixed_mole_frac_conds; mole_frac_cond_row_idx++) {
            for (current_component_idx = 0; current_component_idx < spec->num_prescribed_mole_fraction_coefficients_cols; current_component_idx++) {
                prefactor = spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx];
                // write_row_fixed_mole_fraction accumulates, so it's okay to call multiple times for the same matrix row
                write_row_fixed_mole_fraction(
                    &equilibrium_matrix[(current_row_offset + mole_frac_cond_row_idx) * equilibrium_matrix_cols],
                    &equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx],
                    current_component_idx,
                    spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                    state->free_stable_compset_indices, state->num_free_stable_compsets,
                    spec->free_statevar_indices, spec->num_free_statevars,
                    spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                    state->chemical_potentials, state->mole_fractions, state->system_amount,
                    current_cs_state->mass_jac, current_cs_state->mass_jac_cols,
                    current_cs_state->c_component, current_cs_state->c_component_cols,
                    current_cs_state->c_statevars, current_cs_state->c_statevars_cols,
                    current_cs_state->c_G, current_cs_state->c_G_length, current_cs_state->masses,
                    current_cs_state->moles_normalization, current_cs_state->moles_normalization_grad,
                    state->phase_amt, compset_original_idx, prefactor);
            }
        }
    }
    
    // DEBUG: Check c_G values before calling write_row_fixed_mole_fraction
    #ifdef VERBOSE_DEBUG
    if (state->iteration < 2) {
        printf("\n[GPU] Before write_row_fixed_mole_fraction calls:\n");
        for (int i = 0; i < state->num_compsets && i < 2; ++i) {
            printf("  Phase %d c_G: [%.6e, %.6e]\n", i,
                   state->cs_states[i].c_G_length > 0 ? state->cs_states[i].c_G[0] : 0.0,
                   state->cs_states[i].c_G_length > 1 ? state->cs_states[i].c_G[1] : 0.0);
        }
    }
    #endif
    
    // Loop over fixed stable phases (matching CPU behavior)
    for (int fixed_idx = 0; fixed_idx < spec->num_fixed_stable_compsets; fixed_idx++) {
        stable_idx = spec->fixed_stable_compset_indices[fixed_idx];
        compset_original_idx = stable_idx;
        current_compset = &state->compsets[compset_original_idx];
        current_cs_state = &state->cs_states[compset_original_idx];
        if (current_compset->phase_record == nullptr) continue;

        for (int mole_frac_cond_row_idx = 0; mole_frac_cond_row_idx < num_fixed_mole_frac_conds; mole_frac_cond_row_idx++) {
            for (current_component_idx = 0; current_component_idx < spec->num_prescribed_mole_fraction_coefficients_cols; current_component_idx++) {
                prefactor = spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx];
                write_row_fixed_mole_fraction(
                    &equilibrium_matrix[(current_row_offset + mole_frac_cond_row_idx) * equilibrium_matrix_cols],
                    &equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx],
                    current_component_idx,
                    spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                    state->free_stable_compset_indices, state->num_free_stable_compsets,
                    spec->free_statevar_indices, spec->num_free_statevars,
                    spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                    state->chemical_potentials, state->mole_fractions, state->system_amount,
                    current_cs_state->mass_jac, current_cs_state->mass_jac_cols,
                    current_cs_state->c_component, current_cs_state->c_component_cols,
                    current_cs_state->c_statevars, current_cs_state->c_statevars_cols,
                    current_cs_state->c_G, current_cs_state->c_G_length, current_cs_state->masses,
                    current_cs_state->moles_normalization, current_cs_state->moles_normalization_grad,
                    state->phase_amt, compset_original_idx, prefactor);
            }
        }
    }

    // REMOVED: The zeroing was incorrectly discarding contributions from the first loop
    // The first loop already handles all active phases, so we don't need a second loop
    
    // COMMENTED OUT: This second loop is redundant - the first loop already handles all phases
    /*
    for (int phase_idx = 0; phase_idx < state->num_compsets; phase_idx++) {
        CompositionSet* phase_compset = &state->compsets[phase_idx];
        CompsetState* phase_cs_state = &state->cs_states[phase_idx];
        
        if (phase_compset->phase_record == nullptr) continue;
        
        // DEBUG: Print which phases are contributing
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration == 0) {
            printf("[GPU DEBUG] Phase %d contributing to mole frac constraints\n", phase_idx);
            printf("  Phase amount: %e\n", state->phase_amt[phase_idx]);
            printf("  Is this phase in free_stable_compset_indices? ");
            bool is_free = false;
            for (int i = 0; i < state->num_free_stable_compsets; i++) {
                if (state->free_stable_compset_indices[i] == phase_idx) {
                    is_free = true;
                    break;
                }
            }
            printf("%s\n", is_free ? "YES" : "NO");
        }
        #endif
        
        // Contribute this phase to all mole fraction constraint rows
        for (int mole_frac_cond_row_idx = 0; mole_frac_cond_row_idx < num_fixed_mole_frac_conds; mole_frac_cond_row_idx++) {
            for (current_component_idx = 0; current_component_idx < spec->num_prescribed_mole_fraction_coefficients_cols; current_component_idx++) {
                prefactor = spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx];
                
                // DEBUG: Print constraint details for all phases
                #ifdef VERBOSE_DEBUG
                if (state->iteration == 0 && mole_frac_cond_row_idx == 0) {
                    printf("[GPU MOLE FRAC CONSTRAINT] Phase %d, Constraint %d, Component %d: prefactor=%e\n", 
                           phase_idx, mole_frac_cond_row_idx, current_component_idx, prefactor);
                    printf("  Phase amount: %e, num_compsets=%d\n", state->phase_amt[phase_idx], state->num_compsets);
                }
                #endif
                
                // write_row_fixed_mole_fraction accumulates, so it's okay to call multiple times for the same matrix row
                write_row_fixed_mole_fraction(
                    &equilibrium_matrix[(current_row_offset + mole_frac_cond_row_idx) * equilibrium_matrix_cols],
                    &equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx],
                    current_component_idx,
                    spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                    state->free_stable_compset_indices, state->num_free_stable_compsets,
                    spec->free_statevar_indices, spec->num_free_statevars,
                    spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                    state->chemical_potentials, state->mole_fractions, state->system_amount,
                    phase_cs_state->mass_jac, phase_cs_state->mass_jac_cols,
                    phase_cs_state->c_component, phase_cs_state->c_component_cols,
                    phase_cs_state->c_statevars, phase_cs_state->c_statevars_cols,
                    phase_cs_state->c_G, phase_cs_state->c_G_length, phase_cs_state->masses,
                    phase_cs_state->moles_normalization, phase_cs_state->moles_normalization_grad,
                    state->phase_amt, phase_idx, prefactor);
            }
        }
    }
    */
    
    // DEBUG: Print RHS values before residual subtraction
    #ifdef VERBOSE_DEBUG
    if (state->condition_idx == 0 && state->iteration == 0) {
        printf("[GPU DEBUG] Before residual subtraction:\n");
        for (int i = 0; i < num_fixed_mole_frac_conds; i++) {
            printf("  Mole frac constraint %d RHS: %e\n", i, equilibrium_rhs[current_row_offset + i]);
        }
    }
    #endif
    
    // After accumulating all phase contributions, subtract the residual from RHS
    for (int mole_frac_cond_row_idx = 0; mole_frac_cond_row_idx < num_fixed_mole_frac_conds; mole_frac_cond_row_idx++) {
        double component_residual = 0.0;
        
        // DEBUG: Print what we're calculating
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration == 0) {
            printf("[GPU CONSTRAINT DEBUG] Calculating residual for constraint row %d:\n", mole_frac_cond_row_idx);
            printf("  System mole fractions: [");
            for (int comp_i = 0; comp_i < spec->num_prescribed_mole_fraction_coefficients_cols; comp_i++) {
                printf("%e", state->mole_fractions[comp_i]);
                if (comp_i < spec->num_prescribed_mole_fraction_coefficients_cols - 1) printf(", ");
            }
            printf("]\n");
            printf("  Coefficients for this constraint:");
            for (int i = 0; i < spec->num_prescribed_mole_fraction_coefficients_cols; i++) {
                printf(" [%d]=%e", i, spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][i]);
            }
            printf("\n");
            printf("  Target value (prescribed_mole_fraction_rhs[%d]) = %e\n", 
                   mole_frac_cond_row_idx, spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx]);
        }
        #endif
        
        for (current_component_idx = 0; current_component_idx < spec->num_prescribed_mole_fraction_coefficients_cols; current_component_idx++) {
            component_residual += spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx] *
                                  state->mole_fractions[current_component_idx];
            #ifdef VERBOSE_DEBUG
            if (state->condition_idx == 0 && state->iteration == 0 && mole_frac_cond_row_idx == 0) {
                printf("[GPU RESIDUAL CALC] comp_idx=%d: coeff=%e * mole_frac=%e = %e (cumulative=%e)\n",
                       current_component_idx,
                       spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx],
                       state->mole_fractions[current_component_idx],
                       spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx] * state->mole_fractions[current_component_idx],
                       component_residual);
            }
            #endif
        }
        component_residual -= spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx];
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration == 0 && mole_frac_cond_row_idx == 0) {
            printf("[GPU RESIDUAL CALC] After subtracting target %e: residual = %e\n",
                   spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx], component_residual);
        }
        #endif
        
        // DEBUG: Print mole fraction constraint calculation
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration < 3) {
            printf("[GPU MOLE FRAC CONSTRAINT] Row %d: residual = %e (current X*coeff = %e, target = %e)\n",
                   mole_frac_cond_row_idx, component_residual, 
                   component_residual + spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx],
                   spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx]);
            printf("  state->mole_fractions: [");
            for (int comp_i = 0; comp_i < spec->num_prescribed_mole_fraction_coefficients_cols; comp_i++) {
                printf("%e", state->mole_fractions[comp_i]);
                if (comp_i < spec->num_prescribed_mole_fraction_coefficients_cols - 1) printf(", ");
            }
            printf("]\n");
        }
        #endif
        
        // DEBUG: Print RHS before and after residual subtraction
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration < 3) {
            printf("[GPU MOLE FRAC] Row %d RHS before residual: %e\n", 
                   mole_frac_cond_row_idx, equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx]);
        }
        #endif
        
        equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx] -= component_residual;
        
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration < 3) {
            printf("[GPU MOLE FRAC] Row %d RHS after residual: %e (residual was %e)\n", 
                   mole_frac_cond_row_idx, equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx], component_residual);
        }
        #endif
    }
    // Add system amount constraint row to match CPU EXACTLY
    // The CPU calls write_row_fixed_mole_amount for each component to build this row
    // This adds small contributions to the chemical potential columns (not exactly zero!)
    
    // SYSTEM AMOUNT CONSTRAINT ROW - Match CPU exactly by calling write_row_fixed_mole_amount
    int system_amount_row_idx = current_row_offset + num_fixed_mole_frac_conds;
    
    // Zero out the row first (it should already be zeroed but let's be explicit)
    for (int col = 0; col < equilibrium_matrix_cols; col++) {
        equilibrium_matrix[system_amount_row_idx * equilibrium_matrix_cols + col] = 0.0;
    }
    equilibrium_rhs[system_amount_row_idx] = 0.0;
    
    // Loop over all active phases and call write_row_fixed_mole_amount for each component
    // This matches CPU's fill_equilibrium_system logic exactly
    for (int stable_idx = 0; stable_idx < state->num_free_stable_compsets; stable_idx++) {
        int compset_original_idx = state->free_stable_compset_indices[stable_idx];
        CompositionSet* current_compset = &state->compsets[compset_original_idx];
        CompsetState* current_cs_state = &state->cs_states[compset_original_idx];
        
        if (current_compset->phase_record == nullptr) continue;
        
        // Call write_row_fixed_mole_amount for each component (matching CPU)
        for (int component_idx = 0; component_idx < spec->num_components; component_idx++) {
            write_row_fixed_mole_amount(
                &equilibrium_matrix[system_amount_row_idx * equilibrium_matrix_cols],
                &equilibrium_rhs[system_amount_row_idx],
                component_idx,
                spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                state->free_stable_compset_indices, state->num_free_stable_compsets,
                spec->free_statevar_indices, spec->num_free_statevars,
                spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                state->chemical_potentials,
                current_cs_state->mass_jac, current_cs_state->mass_jac_cols,
                current_cs_state->c_component, current_cs_state->c_component_cols,
                current_cs_state->c_statevars, current_cs_state->c_statevars_cols,
                current_cs_state->c_G, current_cs_state->c_G_length,
                current_cs_state->masses,
                current_cs_state->moles_normalization,
                current_cs_state->moles_normalization_grad,
                state->phase_amt,
                compset_original_idx);
        }
    }
    
    // Also handle fixed stable phases (if any) - matching CPU
    for (int fixed_idx = 0; fixed_idx < spec->num_fixed_stable_compsets; fixed_idx++) {
        int compset_original_idx = spec->fixed_stable_compset_indices[fixed_idx];
        CompositionSet* current_compset = &state->compsets[compset_original_idx];
        CompsetState* current_cs_state = &state->cs_states[compset_original_idx];
        
        if (current_compset->phase_record == nullptr) continue;
        
        // Call write_row_fixed_mole_amount for each component
        for (int component_idx = 0; component_idx < spec->num_components; component_idx++) {
            write_row_fixed_mole_amount(
                &equilibrium_matrix[system_amount_row_idx * equilibrium_matrix_cols],
                &equilibrium_rhs[system_amount_row_idx],
                component_idx,
                spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                state->free_stable_compset_indices, state->num_free_stable_compsets,
                spec->free_statevar_indices, spec->num_free_statevars,
                spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                state->chemical_potentials,
                current_cs_state->mass_jac, current_cs_state->mass_jac_cols,
                current_cs_state->c_component, current_cs_state->c_component_cols,
                current_cs_state->c_statevars, current_cs_state->c_statevars_cols,
                current_cs_state->c_G, current_cs_state->c_G_length,
                current_cs_state->masses,
                current_cs_state->moles_normalization,
                current_cs_state->moles_normalization_grad,
                state->phase_amt,
                compset_original_idx);
        }
    }
    
    // Finally, subtract the system amount residual from RHS (matching CPU)
    double system_residual = state->system_amount - spec->prescribed_system_amount;
    equilibrium_rhs[system_amount_row_idx] -= system_residual;
    
#ifdef VERBOSE_DEBUG
    // DEBUG: Print the complete equilibrium matrix for iteration 0
    if (state->iteration == 0) {
        printf("[GPU EQUILIBRIUM MATRIX] Iteration 0 (rows=%d, cols=%d):\n", total_rows, equilibrium_matrix_cols);
        printf("  Phase amounts before solve: ");
        for (int i = 0; i < state->num_free_stable_compsets; i++) {
            int idx = state->free_stable_compset_indices[i];
            printf("phase_%d=%.6e ", idx, state->phase_amt[idx]);
        }
        printf("\n");
        
        for (int row = 0; row < total_rows; row++) {
            printf("  Row %d: ", row);
            for (int col = 0; col < equilibrium_matrix_cols; col++) {
                printf("%+.6e ", equilibrium_matrix[row * equilibrium_matrix_cols + col]);
            }
            printf("| RHS: %+.6e", equilibrium_rhs[row]);
            
            // Identify what this row represents
            if (row < state->num_free_stable_compsets) {
                printf(" (phase %d energy)", state->free_stable_compset_indices[row]);
            } else if (row < state->num_free_stable_compsets + spec->num_prescribed_mole_fraction_conditions) {
                printf(" (mole fraction constraint %d)", row - state->num_free_stable_compsets);
            } else if (row == total_rows - 1) {
                printf(" (system amount)");
            }
            printf("\n");
        }
    }
#endif
}

// run_loop, solve_state, advance_state, remove_and_consolidate_phases, change_phases
// are largely compatible with these changes,
// as they primarily operate on the counts and indices managed by SystemState and SystemSpecification,
// which are now implicitly simpler due to MAX_PHASE_LOCAL_CONDITIONS being 0.
// Ensure MAX_EQ_SOLN_LEN and related matrix defines are correct for the number of free variables.
// MAX_EQ_SOLN_LEN = num_free_chem_pot + num_free_stable_compsets + num_free_statevars

// check_convergence, pre_solve_hook, post_solve_hook are simple and remain the same.
// The other complex functions (run_loop, solve_state, advance_state, remove_and_consolidate_phases, change_phases)
// were provided in the prior turn and their internal logic largely relies on the dimensions
// and counts that are now correctly set up by the updated init methods and struct definitions.
// For example, `solve_state` uses `soln_length` which is `spec->num_free_chemical_potentials + state->num_free_stable_compsets + spec->num_free_statevars`.
// This will correctly reflect the size of the problem.
// The MAX_ defines for SVD buffers should also be reviewed to ensure they are sufficient for the largest possible system matrix.
// MAX_SVD_DIM used MAX_PHASES for num_free_stable_compsets.
// MAX_EQ_MATRIX_ROWS = num_free_stable_phases + num_fixed_stable_cs + num_fixed_mole_frac_conds + 1
// MAX_EQ_MATRIX_COLS = num_free_chem_pot + num_free_stable_phases + num_free_statevars
// These should be consistent. For example, num_free_stable_phases can be up to MAX_PHASES.
// num_fixed_stable_cs up to MAX_PHASES. num_fixed_mole_frac_conds up to MAX_COMPONENTS.
// So MAX_EQ_MATRIX_ROWS could be approx 2*MAX_PHASES + MAX_COMPONENTS + 1.
// And MAX_EQ_MATRIX_COLS could be MAX_COMPONENTS + MAX_PHASES + MAX_STATEVARS.
// The MAX_SVD_DIM should be the larger of these two.
// MAX_SVD_DIM = (MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2)
// This seems roughly compatible if MAX_PHASES dominates MAX_FIXED_MOLE_FRACTION_CONDITIONS.
// For instance, if MAX_PHASES=64, MAX_COMPONENTS=32, MAX_STATEVARS=8:
// MAX_EQ_MATRIX_ROWS approx 2*64 + 32 + 1 = 128 + 32 + 1 = 161
// MAX_EQ_MATRIX_COLS approx 32 + 64 + 8 = 104
// MAX_SVD_DIM needs to be at least max(161, 104).
// Current MAX_SVD_DIM = 64(MP) + 32(MFMC) + 32(MC) + 8(MSV) + 2 = 138. This is NOT large enough for rows.
// Let's redefine MAX_SVD_M and MAX_SVD_N for lstsq based on these more accurate estimates.
// The SystemSpecification struct buffers (A_lstsq_copy etc.) use MAX_SVD_M and MAX_SVD_N.
// These need to be updated in that struct's definition.
// The prior definition of MAX_SVD_DIM was:
// #define MAX_SVD_DIM (MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2)
// This should be:
// #define EQ_SYS_MAX_ROWS (2 * MAX_PHASES + MAX_COMPONENTS + 1)
// #define EQ_SYS_MAX_COLS (MAX_COMPONENTS + MAX_PHASES + MAX_STATEVARS)
// #define MAX_SVD_M_EQ EQ_SYS_MAX_ROWS
// #define MAX_SVD_N_EQ EQ_SYS_MAX_COLS
// And then use these for the A_lstsq_copy, U_lstsq, V_lstsq, singular_values_lstsq, superdiag_lstsq buffers.
// I will update the #defines in SystemSpecification.

// For brevity, I'm focusing on the direct impact of removing phase-local conditions
// on struct definitions and initializations. The logical flow of those larger functions
// remains the same but operates on data structures that are now simpler.

// [Pasting the remaining functions from the previous generated code for completeness and self-contained nature of the final code block]

__device__ bool check_convergence(SystemSpecification* spec, SystemState* state) {
    // SEGMENT 38: CHECK CONVERGENCE
    gpu_debug_log(38, "Check convergence", state->condition_idx);
    
#ifdef PYCGPU_FP32EMU
    // FP32-emulation prototype: solutions are rounded to float precision each
    // iteration, so the FP64 convergence deltas (1e-10/5e-9) are unreachable.
    // These limits target the ~1e-7 resolution an actual FP32 pass could
    // deliver; a subsequent FP64 pass would polish to full tolerance.
    double ALLOWED_DELTA_Y = 1e-6;
    double ALLOWED_DELTA_PHASE_AMT = 1e-6;
    double ALLOWED_DELTA_STATEVAR = 1e-4;
#else
    double ALLOWED_DELTA_Y = 5e-09;
    double ALLOWED_DELTA_PHASE_AMT = 1e-10;
    double ALLOWED_DELTA_STATEVAR = 1e-5;
#endif
    
    if (state->condition_idx < 3) {
        gpu_debug_log_value("largest_phase_amt_change", state->largest_phase_amt_change);
        gpu_debug_log_value("largest_y_change", state->largest_y_change);
        gpu_debug_log_value("largest_statevar_change", state->largest_statevar_change);
        gpu_debug_log_value("mass_residual", state->mass_residual);
        gpu_debug_log_value("iterations_since_last_phase_change", (double)state->iterations_since_last_phase_change);
    }

    bool solution_is_feasible =
        (state->largest_phase_amt_change < ALLOWED_DELTA_PHASE_AMT) &&
        (state->largest_y_change < ALLOWED_DELTA_Y) &&
        (state->largest_statevar_change < ALLOWED_DELTA_STATEVAR) &&
        (state->mass_residual <
         #ifdef PYCGPU_FP32EMU
         fmax(spec->ALLOWED_MASS_RESIDUAL, 1e-6)
         #else
         spec->ALLOWED_MASS_RESIDUAL
         #endif
        );

    #ifdef PYCGPU_TRACE_LOOP
    if (state->condition_idx == 0) {
        printf("TRACECONV iter=%d damt=%.3e dy=%.3e dsv=%.3e mres=%.3e quiet=%d feas=%d\n",
               state->iteration, state->largest_phase_amt_change, state->largest_y_change,
               state->largest_statevar_change, state->mass_residual,
               state->iterations_since_last_phase_change, (int)solution_is_feasible);
    }
    #endif

    // CPU (minimizer.pyx check_convergence) requires >= 10 iterations since
    // the last phase change. With the ramped early steps this matters: the
    // per-iteration deltas are tiny at step 0.05-0.5, so a shorter gate
    // declares convergence before the solution is polished (measured 0.066
    // J/mol short on the ill-conditioned alfe magnetic-Hessian test).
    if (solution_is_feasible && (state->iterations_since_last_phase_change >= 10)) {
        gpu_debug_log_value("converged", 1.0);
        return true;
    }
    gpu_debug_log_value("converged", 0.0);
    return false;
}

__device__ bool pre_solve_hook(SystemSpecification* spec, SystemState* state) {
    return true;
}

__device__ bool post_solve_hook(SystemSpecification* spec, SystemState* state) {
    return true;
}

// solve_state and run_loop implementations start below

__device__ void advance_state(SystemSpecification* spec, SystemState* state, const double* equilibrium_soln, int soln_length, double step_size_param) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // SEGMENT 34: UPDATE STATE WITH STEP SIZE
    gpu_debug_log(34, "Update state with step size", state->iteration);
    gpu_debug_log_value("step_size", step_size_param);
    
    double current_step_size = step_size_param;
    double MIN_PHASE_AMOUNT = 1e-16;  // Match CPU's 1e-16 in advance_state, not 1e-10!

    // Chemical potentials are now handled in solve_state (matching CPU approach)
    // Start with phase amount updates
    int soln_idx_offset = spec->num_free_chemical_potentials; // Skip chemical potentials
    double phase_amt_step_size_limiter = current_step_size;
    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int compset_original_idx = state->free_stable_compset_indices[i];
        if (compset_original_idx < 0 || compset_original_idx >= state->num_compsets) continue;
        double delta_NP = equilibrium_soln[soln_idx_offset + i];
        if (state->phase_amt[compset_original_idx] + delta_NP < MIN_PHASE_AMOUNT) {
            if (fabs(delta_NP) > MIN_PHASE_AMOUNT) {
                phase_amt_step_size_limiter = fmin(phase_amt_step_size_limiter,
                    (MIN_PHASE_AMOUNT - state->phase_amt[compset_original_idx]) / delta_NP);
            } else if (delta_NP < 0) {
                 phase_amt_step_size_limiter = 0.0;
            }
        }
    }

    state->largest_phase_amt_change = 0.0;
    
    // SEGMENT 35: UPDATE PHASE AMOUNTS
    gpu_debug_log(35, "Update phase amounts", state->iteration);
    gpu_debug_log_value("phase_amt_step_limiter", phase_amt_step_size_limiter);
    
    #ifdef VERBOSE_DEBUG
    if (state->iteration < 10 && thread_id == 0 && phase_amt_step_size_limiter < 1.0) {
        printf("GPU DEBUG: Step size limited to %.6e at iteration %d\n", 
               phase_amt_step_size_limiter, state->iteration);
    }
    #endif
    
    
    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int compset_original_idx = state->free_stable_compset_indices[i];
        if (compset_original_idx < 0 || compset_original_idx >= state->num_compsets) continue;
        double old_amt = state->phase_amt[compset_original_idx];
        double delta = equilibrium_soln[soln_idx_offset + i];
        // The equilibrium solver returns phase amount deltas directly
        // No need to scale by 100 - this was causing phases to shrink incorrectly
        double actual_change = phase_amt_step_size_limiter * delta;
        state->phase_amt[compset_original_idx] += actual_change;
        
        // DEBUG: Print details
        #ifdef VERBOSE_DEBUG
        if (state->iteration < 10) {
            printf("  Phase %d: old=%.6e, delta=%.6e (raw), actual_change=%.6e, new=%.6e\n",
                   compset_original_idx, old_amt, delta, actual_change, state->phase_amt[compset_original_idx]);
        }
        #endif
        
        if (state->phase_amt[compset_original_idx] < MIN_PHASE_AMOUNT) {
            state->phase_amt[compset_original_idx] = MIN_PHASE_AMOUNT;
        }
        
        // NOTE: CPU advance_state does NOT write compset.NP here; NP is only
        // refreshed in recompute() as phase_amt * moles_per_formula_unit (atoms).
        if (fabs(actual_change) > state->largest_phase_amt_change) {
            state->largest_phase_amt_change = fabs(actual_change);
        }
        
        // Additional debug for phase removal issue
        #ifdef VERBOSE_DEBUG
        if (state->iteration < 50 && state->phase_amt[compset_original_idx] < 1e-6) {
            printf("GPU DEBUG: Phase %d amount became very small (%.10e) at iteration %d\n", 
                   compset_original_idx, state->phase_amt[compset_original_idx], state->iteration);
        }
        #endif
    }
    
    gpu_debug_log_value("largest_phase_amt_change", state->largest_phase_amt_change);
    
    // DEBUG: Check total phase amounts after update
    #ifdef VERBOSE_DEBUG
    double phase_amt_sum_after = 0.0;
    for (int idx = 0; idx < state->num_compsets; ++idx) {
        phase_amt_sum_after += state->phase_amt[idx];
    }
    printf("[GPU MASS BALANCE] advance_state() - after phase update: sum(phase_amt) = %.15e\n", phase_amt_sum_after);
    #endif
    
    // DO NOT normalize phase amounts in advance_state!
    // The CPU solver doesn't do this, and it prevents convergence by undoing all changes.
    // The phase amounts should be allowed to change as directed by the equilibrium solution.
    // The system amount constraint is enforced through the equilibrium matrix, not by normalization.
    soln_idx_offset += state->num_free_stable_compsets; // Start of state variable deltas

    // SEGMENT 36: UPDATE STATE VARIABLES  
    gpu_debug_log(36, "Update state variables", state->iteration);
    
    state->largest_statevar_change = 0.0;
    for(int i=0; i<spec->num_statevars; ++i) state->delta_statevars[i] = 0.0;

    for (int i = 0; i < spec->num_free_statevars; ++i) {
        int statevar_global_idx = spec->free_statevar_indices[i];
        if (statevar_global_idx < 0 || statevar_global_idx >= spec->num_statevars) continue;
        state->delta_statevars[statevar_global_idx] = equilibrium_soln[soln_idx_offset + i];
        double psc = 0.0;
        if (state->num_compsets > 0 && state->compsets[0].phase_record != nullptr) { // Check valid compset
            if (fabs(state->compsets[0].dof[statevar_global_idx]) < 1e-12) {
                 psc = (fabs(state->delta_statevars[statevar_global_idx]) > 1e-12) ? 1e30 : 0.0;
            } else {
                psc = fabs(state->delta_statevars[statevar_global_idx] / state->compsets[0].dof[statevar_global_idx]);
            }
            if (psc > state->largest_statevar_change) {
                state->largest_statevar_change = psc;
            }
        }
    }
    
    gpu_debug_log_value("largest_statevar_change", state->largest_statevar_change);
    for (int idx = 0; idx < state->num_compsets; ++idx) {
        if (state->compsets[idx].phase_record == nullptr) continue;
        for (int i = 0; i < spec->num_free_statevars; ++i) {
             int statevar_global_idx = spec->free_statevar_indices[i];
             if (statevar_global_idx < 0 || statevar_global_idx >= spec->num_statevars) continue;
             
             state->compsets[idx].dof[statevar_global_idx] += state->delta_statevars[statevar_global_idx];
        }
    }

    // DEBUG: Check phase_compositions before site fraction update
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0 && state->iteration < 5) {
        printf("GPU DEBUG: Phase compositions BEFORE site fraction update (iteration %d):\n", state->iteration);
        for (int i = 0; i < state->num_free_stable_compsets && i < 2; ++i) {
            int idx = state->free_stable_compset_indices[i];
            printf("  Phase %d: [%.6f, %.6f]\n", idx,
                   state->phase_compositions[idx * MAX_COMPONENTS + 0],
                   state->phase_compositions[idx * MAX_COMPONENTS + 1]);
        }
    }
    #endif
    
    // SEGMENT 37: UPDATE SITE FRACTIONS
    gpu_debug_log(37, "Update site fractions", state->iteration);
    
    state->largest_y_change = 0.0;
    double new_y_for_phase[MAX_DOF_PER_PHASE];

    // CPU (minimizer.pyx:1437-1453): `step_size` is ONE variable shared across the
    // whole per-compset loop — a bounds-hit halving for one phase also damps the
    // site-fraction steps of every SUBSEQUENT phase in this advance_state call.
    double site_frac_step_limiter = current_step_size;

    for (int idx = 0; idx < state->num_compsets; ++idx) {
        CompsetState* csst = &state->cs_states[idx];
        CompositionSet* compset = &state->compsets[idx];
        if (compset->phase_record == nullptr) continue;
        
        // NOTE: Do NOT skip phases with small amounts - they need to converge
        // before removal/consolidation. This matches CPU behavior.
        
        const PhaseRecord* pr = compset->phase_record;
        int num_site_fracs = pr->phase_dof;

        for(int i=0; i<num_site_fracs; ++i) csst->delta_y[i] = 0.0;

        for (int i = 0; i < num_site_fracs; ++i) {
            csst->delta_y[i] += csst->c_G[i];
            for (int sv_idx = 0; sv_idx < spec->num_statevars; ++sv_idx) {
                 if (sv_idx < 0 || sv_idx >= spec->num_statevars) continue;
                csst->delta_y[i] += csst->c_statevars[i * csst->c_statevars_cols + sv_idx] * state->delta_statevars[sv_idx];
            }
            for (int cp_idx = 0; cp_idx < spec->num_components; ++cp_idx) {
                 if (cp_idx < 0 || cp_idx >= spec->num_components) continue;
                // Use absolute chemical potentials, NOT deltas!
                // CPU code at minimizer.pyx line 1388 uses state.chemical_potentials[chempot_idx] directly
                // This matches Eq. 43 in Sundman 2015
                csst->delta_y[i] += csst->c_component[cp_idx * csst->c_component_cols + i] * state->chemical_potentials[cp_idx];
            }
            for (int cons_idx = 0; cons_idx < csst->internal_cons_length; ++cons_idx) {
                csst->delta_y[i] -= csst->full_e_matrix[(num_site_fracs + cons_idx) * csst->full_e_matrix_dim + i] * csst->internal_cons[cons_idx];
            }
        }
        
        // DEBUG: Print delta_y calculation details after consolidation
        #ifdef VERBOSE_DEBUG
        if (state->iteration >= 2 && state->iteration < 5 && idx == 0) {
            printf("GPU DEBUG: delta_y calculation for phase %d at iteration %d:\n", idx, state->iteration);
            for (int i = 0; i < num_site_fracs; ++i) {
                printf("  delta_y[%d] = %.15e\n", i, csst->delta_y[i]);
                printf("    c_G[%d] = %.15e\n", i, csst->c_G[i]);
                printf("    Chemical potential contribution: ");
                double cp_contrib = 0.0;
                for (int cp_idx = 0; cp_idx < spec->num_components; ++cp_idx) {
                    double contrib = csst->c_component[cp_idx * csst->c_component_cols + i] * state->chemical_potentials[cp_idx];
                    cp_contrib += contrib;
                    if (fabs(contrib) > 1e-15) {
                        printf("cp[%d]=%.6e ", cp_idx, contrib);
                    }
                }
                printf("(total: %.15e)\n", cp_contrib);
            }
        }
        #endif

        // CPU recomputes the floor per compset from the CURRENT (possibly already
        // halved) step size: minimum_step_size = 1e-20 * step_size (pyx:1436).
        double min_allowed_sf_step = 1e-20 * site_frac_step_limiter;
        bool exceeded_bounds_for_phase;

        do {
            exceeded_bounds_for_phase = false;
            for (int i = 0; i < num_site_fracs; ++i) {
                double current_site_frac_val = compset->dof[spec->num_statevars + i];
                new_y_for_phase[i] = current_site_frac_val + site_frac_step_limiter * csst->delta_y[i];
                if (new_y_for_phase[i] > 1.0) {
                    if ((new_y_for_phase[i] - 1.0) > 1e-11) exceeded_bounds_for_phase = true;
                    new_y_for_phase[i] = 1.0;
                } else if (new_y_for_phase[i] < MIN_SITE_FRACTION) {
                    if ((MIN_SITE_FRACTION - new_y_for_phase[i]) > 1e-11) exceeded_bounds_for_phase = true;
                    new_y_for_phase[i] = fmax(current_site_frac_val / 100.0, MIN_SITE_FRACTION);
                }
            }
            if (exceeded_bounds_for_phase) {
                site_frac_step_limiter *= 0.5;
            }
        } while (exceeded_bounds_for_phase && site_frac_step_limiter >= min_allowed_sf_step);

        double max_dy_this_cs = 0.0;
        for (int i = 0; i < num_site_fracs; ++i) {
            double old_val = compset->dof[spec->num_statevars + i];
            // Apply the final calculated new_y_for_phase value for this iteration of step_limiter
            compset->dof[spec->num_statevars + i] = new_y_for_phase[i]; // Value after potential bounding
            double change_this_y = fabs(compset->dof[spec->num_statevars + i] - old_val);
            if (change_this_y > max_dy_this_cs) max_dy_this_cs = change_this_y;
        }
        // CPU parity (minimizer.pyx:956): largest_y_change is reset INSIDE the
        // per-compset loop, so check_convergence sees only the LAST compset's
        // largest site-fraction step — a max over all compsets is NOT the
        // reference semantics. This matters: a removed compset whose internal
        // Newton oscillates (AlCuFe BCC_B2 ordering gap) otherwise blocks
        // convergence for ~50 iterations, after which change_phases samples its
        // bouncing dof at a spurious positive driving force, re-adds it, and
        // the full-step re-solve destroys the converged solution (+249 J/mol
        // with a matching phase set, or a lost second phase).
        state->largest_y_change = max_dy_this_cs;
        #ifdef PYCGPU_TRACE_LOOP
        if (state->condition_idx == 0 && max_dy_this_cs > 1e-6) {
            printf("TRACEDY iter=%d cs=%d amt=%.3e maxdy=%.3e limiter=%.3e y=", state->iteration,
                   idx, state->phase_amt[idx], max_dy_this_cs, site_frac_step_limiter);
            for (int i = 0; i < num_site_fracs; ++i)
                printf("%s%.6g", i ? "," : "", compset->dof[spec->num_statevars + i]);
            printf("\n");
        }
        #endif
    }

    gpu_debug_log_value("largest_y_change", state->largest_y_change);
}

__device__ bool remove_and_consolidate_phases(SystemSpecification* spec, SystemState* state) {
    bool phases_changed = false;
    double COMPSET_CONSOLIDATE_DISTANCE = 1e-4;

    int compset_indices_to_remove_temp[MAX_PHASES];
    int num_to_remove = 0;

    // Debug: Log entry to function
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // DEBUG removed - phase composition update before consolidation
    
    // DEBUG removed - was checking phase compositions

    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int idx1 = state->free_stable_compset_indices[i];
        if (idx1 < 0 || idx1 >= state->num_compsets) continue;
        CompositionSet* compset1 = &state->compsets[idx1];
        if (compset1->fixed || compset1->phase_record == nullptr) continue;

        bool already_marked_for_removal1 = false;
        for(int k=0; k < num_to_remove; ++k) if(compset_indices_to_remove_temp[k] == idx1) already_marked_for_removal1 = true;
        if(already_marked_for_removal1) continue;

        // Remove unstable phases (matching CPU minimizer.pyx line 1328)
        if (state->phase_amt[idx1] < 1e-10) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] Removing phase %d (iteration %d): amount %.15e < 1e-10\n",
                   idx1, state->iteration, state->phase_amt[idx1]);
            #endif
            // Check if removing this phase would leave us unable to satisfy mass balance
            // Count how many phases would remain after removal
            int phases_remaining = 0;
            for (int j = 0; j < state->num_free_stable_compsets; ++j) {
                int idx_check = state->free_stable_compset_indices[j];
                if (idx_check != idx1 && state->phase_amt[idx_check] >= 1e-10) {
                    phases_remaining++;
                }
            }
            
            // If this is the last phase that could satisfy constraints, don't remove it
            if (phases_remaining == 0 && spec->num_prescribed_mole_fraction_conditions > 0) {
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0 && state->iteration < 5) {
                    printf("  Phase %d NOT removed - last phase needed for mass balance\n", idx1);
                }
                #endif
                continue;
            }
            
            if (num_to_remove < MAX_PHASES) compset_indices_to_remove_temp[num_to_remove++] = idx1;
            state->phase_amt[idx1] = 0.0;  // CPU sets to 0 at line 1330
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && state->iteration < 5) {
                printf("  Phase %d marked for removal - amount %.2e < 1e-10\n", idx1, state->phase_amt[idx1]);
            }
            #endif
            continue;
        }

        for (int j = 0; j < state->num_free_stable_compsets; ++j) {
            if (i == j) continue;
            int idx2 = state->free_stable_compset_indices[j];
            if (idx2 < 0 || idx2 >= state->num_compsets) continue;
            
            CompositionSet* compset2 = &state->compsets[idx2];
            if (compset2->fixed || compset2->phase_record == nullptr) continue;
            if (compset1->phase_record != compset2->phase_record) continue; // Different phase types

            bool already_marked_for_removal2 = false;
            for(int k=0; k < num_to_remove; ++k) if(compset_indices_to_remove_temp[k] == idx2) already_marked_for_removal2 = true;
            if(already_marked_for_removal2) continue;

            bool should_consolidate = true;
            double max_diff = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                double diff = fabs(state->phase_compositions[idx1 * MAX_COMPONENTS + comp_idx] -
                                   state->phase_compositions[idx2 * MAX_COMPONENTS + comp_idx]);
                if (diff > max_diff) max_diff = diff;
                if (diff > COMPSET_CONSOLIDATE_DISTANCE) {
                    should_consolidate = false;
                    break;
                }
            }
            
            #ifdef VERBOSE_DEBUG
            // Debug: Log consolidation check
            if (compset1->phase_record == compset2->phase_record) {
                printf("[GPU] Iteration %d: Checking phases %d and %d (same type) for consolidation:\n",
                       state->iteration, idx1, idx2);
                printf("    Compositions: [%.6f, %.6f, %.6f] vs [%.6f, %.6f, %.6f]\n",
                       state->phase_compositions[idx1 * MAX_COMPONENTS + 0],
                       state->phase_compositions[idx1 * MAX_COMPONENTS + 1],
                       state->phase_compositions[idx1 * MAX_COMPONENTS + 2],
                       state->phase_compositions[idx2 * MAX_COMPONENTS + 0],
                       state->phase_compositions[idx2 * MAX_COMPONENTS + 1],
                       state->phase_compositions[idx2 * MAX_COMPONENTS + 2]);
                printf("    Max diff: %.8f (threshold: %.8f)\n", max_diff, COMPSET_CONSOLIDATE_DISTANCE);
                printf("    Amounts: %.6f vs %.6f\n", state->phase_amt[idx1], state->phase_amt[idx2]);
                printf("    Should consolidate: %s\n", should_consolidate ? "YES" : "NO");
            }
            #endif
            
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && state->iteration < 5) {
                // Keep original verbose debug
                if (should_consolidate && state->iteration == 0) {
                    CompositionSet* cs1 = &state->compsets[idx1];
                    CompositionSet* cs2 = &state->compsets[idx2];
                    printf("    [CONSOLIDATION] Phase %d site fractions: Y[0]=%.15e, Y[1]=%.15e\n",
                           idx1, cs1->dof[3], cs1->dof[4]);
                    printf("    [CONSOLIDATION] Phase %d site fractions: Y[0]=%.15e, Y[1]=%.15e\n",
                           idx2, cs2->dof[3], cs2->dof[4]);
                }
            }
            #endif
            
            if (should_consolidate) {
                
                // DEBUG: Site fractions at consolidation moment
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0 && state->iteration < 2) {
                    CompositionSet* cs1 = &state->compsets[idx1];
                    CompositionSet* cs2 = &state->compsets[idx2];
                    printf("[AT CONSOLIDATION] Phase %d Y=[%.15e, %.15e], amt=%.15e\n",
                           idx1, cs1->dof[3], cs1->dof[4], state->phase_amt[idx1]);
                    printf("[AT CONSOLIDATION] Phase %d Y=[%.15e, %.15e], amt=%.15e\n",
                           idx2, cs2->dof[3], cs2->dof[4], state->phase_amt[idx2]);
                }
                #endif
                if (num_to_remove < MAX_PHASES) compset_indices_to_remove_temp[num_to_remove++] = idx2;
                
                // CPU minimizer.pyx:1543 adds phase amounts directly (both in formula
                // units of the SAME phase) with a 1e-8 stability floor. No conversion
                // through moles_normalization.
                state->phase_amt[idx1] = fmax(state->phase_amt[idx1] + state->phase_amt[idx2], 1e-8);

                #ifdef VERBOSE_DEBUG
                if (thread_id == 0) {
                    printf("[CONSOLIDATION] Consolidated phases %d and %d: new amount=%.15e\n",
                           idx1, idx2, state->phase_amt[idx1]);
                }
                #endif
                
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0) {
                    // Show compositions
                    printf("  Phase %d: X=[%.15e, %.15e]\n", 
                           idx1,
                           state->phase_compositions[idx1 * MAX_COMPONENTS + 0],
                           state->phase_compositions[idx1 * MAX_COMPONENTS + 1]);
                    
                    // Also show site fractions
                    CompositionSet* cs1 = &state->compsets[idx1];
                    printf("  Phase %d site fractions: Y=[%.15e, %.15e]\n",
                           idx1, cs1->dof[3], cs1->dof[4]);
                }
                #endif
                state->phase_amt[idx2] = 0.0;
                
                // DO NOT modify site fractions or phase compositions of idx1
                // The CPU keeps them unchanged, allowing the solver to adjust them
                // in subsequent iterations to satisfy the mass balance constraint
            }
        }
    }

    if (num_to_remove > 0) {
        phases_changed = true;
        int new_free_stable_indices[MAX_PHASES];
        int new_count = 0;
        for (int i = 0; i < state->num_free_stable_compsets; ++i) {
            int current_idx = state->free_stable_compset_indices[i];
            bool is_removed = false;
            for (int j = 0; j < num_to_remove; ++j) {
                if (compset_indices_to_remove_temp[j] == current_idx) {
                    is_removed = true;
                    state->phase_amt[current_idx] = 0.0;
                    break;
                }
            }
            if (!is_removed) {
                if (new_count < MAX_PHASES) new_free_stable_indices[new_count++] = current_idx;
            }
        }
        // Match CPU behavior when all phases would be removed
        // CPU minimizer.pyx lines 1509-1517
        bool all_removed_reset = false;
        if (new_count == 0 && state->num_free_stable_compsets > 0 && num_to_remove == state->num_free_stable_compsets) {
            all_removed_reset = true;
            // Do not allow all phases to leave the system
            // Reset all phase amounts to 1 and chemical potentials to 0
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int phase_idx = state->free_stable_compset_indices[i];
                state->phase_amt[phase_idx] = 1.0;
                if (new_count < MAX_PHASES) {
                    new_free_stable_indices[new_count++] = phase_idx;
                }
            }
            
            // Reset chemical potentials to 0
            for (int i = 0; i < spec->num_components; ++i) {
                state->chemical_potentials[i] = 0.0;
            }
            
            // Force fixed chemical potentials to adopt their initial values
            for (int cp_idx = 0; cp_idx < spec->num_fixed_chemical_potentials; ++cp_idx) {
                int comp_idx = spec->fixed_chemical_potential_indices[cp_idx];
                state->chemical_potentials[comp_idx] = spec->initial_chemical_potentials[comp_idx];
            }
            
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0) {
                printf("[GPU PHASE CONSOLIDATION] All phases would be removed - resetting phase amounts to 1.0 and chemical potentials\n");
            }
            #endif
        }


        #ifdef PYCGPU_ROBUST_REMOVAL
        // Opt-in robustness experiment (mirrors PYCALPHAD_ROBUST_REMOVAL on the
        // Cython CPU): count consolidate/collapse removals toward the
        // change_phases re-add budget (times_compset_removed), so a doomed
        // add->collapse->re-add cycle on a near-duplicate compset terminates
        // after MAX_ALLOWED_TIMES_COMPSET_REMOVED attempts and the solver moves
        // on to the next candidate phase.
        if (!all_removed_reset && num_to_remove > 0) {
            for (int j = 0; j < num_to_remove; ++j) {
                int ridx = compset_indices_to_remove_temp[j];
                if (ridx >= 0 && ridx < state->num_compsets) {
                    state->times_compset_removed[ridx]++;
                    #ifdef VERBOSE_DEBUG
                    printf("[GPU ROBUST] iter=%d consolidate-removal of cs %d -> times_removed=%d\n",
                           state->iteration, ridx, state->times_compset_removed[ridx]);
                    #endif
                }
            }
        }
        #endif

        state->num_free_stable_compsets = new_count;
        for (int i = 0; i < new_count; ++i) {
            state->free_stable_compset_indices[i] = new_free_stable_indices[i];
        }
        
        // DEBUG: Print the updated free stable compsets
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("[GPU PHASE CONSOLIDATION] Updated num_free_stable_compsets from %d to %d\n", 
                   state->num_free_stable_compsets + num_to_remove, new_count);
            printf("  Remaining free phases: ");
            for (int i = 0; i < new_count; ++i) {
                printf("%d ", new_free_stable_indices[i]);
            }
            printf("\n");
        }
        #endif
    }
    
    
    return phases_changed;
}

__device__ bool change_phases(SystemSpecification* spec, SystemState* state, 
                              const DeviceGrid* grid_data, const DevicePhaseData* phase_data) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("GPU DEBUG: change_phases called - initial num_free_stable_compsets=%d\n", state->num_free_stable_compsets);
    }
    #endif
    bool phases_changed = false;
    double current_driving_forces[MAX_PHASES]; // Sized to MAX_PHASES
    state->driving_forces(spec, current_driving_forces, MAX_PHASES); // Get DFs for all possible phases

    #ifdef PYCGPU_TRACE_LOOP
    if (thread_id == 0) {
        printf("TRACEADD iter=%d df=", state->iteration);
        for (int i = 0; i < state->num_compsets; ++i)
            printf("%s%.10g", i ? "," : "", current_driving_forces[i]);
        printf(" meta_iters=");
        for (int i = 0; i < state->num_compsets; ++i)
            printf("%s%d", i ? "," : "", state->metastable_phase_iterations[i]);
        printf(" removed=");
        for (int i = 0; i < state->num_compsets; ++i)
            printf("%s%d", i ? "," : "", state->times_compset_removed[i]);
        printf("\n");
    }
    #endif

    // Match CPU minimizer.pyx lines 1391-1403 exactly
    double MIN_PHASE_AMOUNT_FOR_ADD_CHECK = 1e-9;  // CPU uses 1e-9, not MIN_PHASE_FRACTION
    int MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD = 5;
    double MIN_DRIVING_FORCE_TO_ADD = 1e-5;
    int MAX_ALLOWED_TIMES_COMPSET_REMOVED = 4;

    double current_min_phase_amount_for_removal = MIN_PHASE_AMOUNT_FOR_ADD_CHECK;
    if (state->num_free_stable_compsets > spec->max_num_free_stable_phases) {
        // Gibbs phase rule is currently being violated
        // Try forcing phases with small amounts out of the equilibrium
        current_min_phase_amount_for_removal = 1e-4;
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("GPU DEBUG: Gibbs phase rule violation: increasing MIN_PHASE_AMOUNT to %.2e\n", 
                   current_min_phase_amount_for_removal);
        }
        #endif
    }

    int compsets_to_remove_indices[MAX_PHASES];
    int num_to_remove = 0;
    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int cs_original_idx = state->free_stable_compset_indices[i];
        if (cs_original_idx < 0 || cs_original_idx >= state->num_compsets) continue;
        if (state->phase_amt[cs_original_idx] < current_min_phase_amount_for_removal && !state->compsets[cs_original_idx].fixed) {
            if (num_to_remove < MAX_PHASES) {
                compsets_to_remove_indices[num_to_remove++] = cs_original_idx;
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0) {
                    printf("GPU DEBUG: Phase %d marked for removal in change_phases - amt=%.10e < %.10e (iteration %d)\n", 
                           cs_original_idx, state->phase_amt[cs_original_idx], current_min_phase_amount_for_removal, state->iteration);
                }
                #endif
            }
        }
    }

    int compsets_to_add_indices[MAX_PHASES];
    int num_to_add = 0;
    for (int cs_idx = 0; cs_idx < state->num_compsets; ++cs_idx) {
        bool is_already_free_stable = false;
        for(int i=0; i < state->num_free_stable_compsets; ++i) if(state->free_stable_compset_indices[i] == cs_idx) is_already_free_stable = true;
        if (is_already_free_stable || state->compsets[cs_idx].fixed) continue;

        if (state->metastable_phase_iterations[cs_idx] >= MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD &&
            current_driving_forces[cs_idx] > MIN_DRIVING_FORCE_TO_ADD &&
            state->times_compset_removed[cs_idx] < MAX_ALLOWED_TIMES_COMPSET_REMOVED) {
            #ifdef DEBUG_METASTABLE
            if (condition_idx == 0) {
                printf("[GPU] Condition 0: Phase %d metastable for %d iters, df=%e, will add\n", 
                       cs_idx, state->metastable_phase_iterations[cs_idx], current_driving_forces[cs_idx]);
            }
            #endif
            if (num_to_add < MAX_PHASES) compsets_to_add_indices[num_to_add++] = cs_idx;
        }
    }

    int max_allowed_to_add_now = spec->max_num_free_stable_phases + num_to_remove - state->num_free_stable_compsets;

    if (num_to_add > 0) {
        if (max_allowed_to_add_now < 1) {
            int least_removed_cs_original_idx = -1;
            int min_times_removed_val = MAX_ALLOWED_TIMES_COMPSET_REMOVED + 1;
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int current_free_cs_idx = state->free_stable_compset_indices[i];
                if (current_free_cs_idx < 0 || current_free_cs_idx >= state->num_compsets) continue;
                bool is_candidate_for_destabilize = true;
                for(int k=0; k<num_to_remove; ++k) if(compsets_to_remove_indices[k] == current_free_cs_idx) is_candidate_for_destabilize = false;
                if (state->compsets[current_free_cs_idx].fixed) is_candidate_for_destabilize = false;
                if (is_candidate_for_destabilize) {
                    if (state->times_compset_removed[current_free_cs_idx] < min_times_removed_val) {
                        min_times_removed_val = state->times_compset_removed[current_free_cs_idx];
                        least_removed_cs_original_idx = current_free_cs_idx;
                    }
                }
            }
            if (least_removed_cs_original_idx != -1) {
                bool already_in_remove_list = false;
                for(int k=0; k<num_to_remove; ++k) if(compsets_to_remove_indices[k] == least_removed_cs_original_idx) already_in_remove_list = true;
                if(!already_in_remove_list && num_to_remove < MAX_PHASES) compsets_to_remove_indices[num_to_remove++] = least_removed_cs_original_idx;
            }
        }

        // CPU minimizer.pyx:1632-1639 always narrows the add set to exactly ONE
        // candidate: the one with the SMALLEST (still positive) driving force.
        // Ties break to the lowest compset index (candidates were gathered in
        // ascending index order and the comparison is strict).
        int best_to_add_idx = compsets_to_add_indices[0];
        for (int i = 1; i < num_to_add; ++i) {
            int candidate_idx = compsets_to_add_indices[i];
            if (current_driving_forces[candidate_idx] < current_driving_forces[best_to_add_idx]) {
                best_to_add_idx = candidate_idx;
            }
        }
        num_to_add = 1;
        compsets_to_add_indices[0] = best_to_add_idx;
    }

    int final_free_stable_indices[MAX_PHASES];
    int final_free_count = 0;
    bool current_free_set_changed_flag = false;

    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int cs_idx = state->free_stable_compset_indices[i];
        bool removed = false;
        for (int j = 0; j < num_to_remove; ++j) {
            if (compsets_to_remove_indices[j] == cs_idx) {
                removed = true;
                state->times_compset_removed[cs_idx]++;
                state->phase_amt[cs_idx] = 0.0;
                current_free_set_changed_flag = true;
                break;
            }
        }
        if (!removed) {
            if (final_free_count < MAX_PHASES) final_free_stable_indices[final_free_count++] = cs_idx;
        }
    }
    for (int i = 0; i < num_to_add; ++i) {
        int cs_idx_to_add = compsets_to_add_indices[i];
        bool already_present = false;
        for(int k=0; k < final_free_count; ++k) if(final_free_stable_indices[k] == cs_idx_to_add) already_present = true;
        if (!already_present) {
             if (final_free_count < MAX_PHASES) final_free_stable_indices[final_free_count++] = cs_idx_to_add;
             state->phase_amt[cs_idx_to_add] = fmax(state->phase_amt[cs_idx_to_add], 1e-10);
             current_free_set_changed_flag = true;
        }
    }

    if (current_free_set_changed_flag) phases_changed = true;

    // CPU minimizer.pyx:1640 stores the new index list sorted ascending.
    for (int i = 1; i < final_free_count; ++i) {
        int key = final_free_stable_indices[i];
        int j = i - 1;
        while (j >= 0 && final_free_stable_indices[j] > key) {
            final_free_stable_indices[j + 1] = final_free_stable_indices[j];
            --j;
        }
        final_free_stable_indices[j + 1] = key;
    }

    state->num_free_stable_compsets = final_free_count;
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("GPU DEBUG: change_phases - final_free_count=%d, free_stable_indices=[", final_free_count);
        for (int i = 0; i < final_free_count; ++i) {
            state->free_stable_compset_indices[i] = final_free_stable_indices[i];
            if (i > 0) printf(", ");
            printf("%d", final_free_stable_indices[i]);
        }
        printf("]\n");
    } else {
        // Still need to set the indices even for non-zero threads
        for (int i = 0; i < final_free_count; ++i) {
            state->free_stable_compset_indices[i] = final_free_stable_indices[i];
        }
    }
    #else
    // When verbose debug is off, still need to set the indices for all threads
    for (int i = 0; i < final_free_count; ++i) {
        state->free_stable_compset_indices[i] = final_free_stable_indices[i];
    }
    #endif
    
    for (int i = 0; i < final_free_count; ++i) {
        int current_idx = final_free_stable_indices[i];  // Use NEW array, not old!
        if (current_idx >=0 && current_idx < state->num_compsets) { // boundary check
            if (state->phase_amt[current_idx] < 1e-10 && !state->compsets[current_idx].fixed) {
                 state->phase_amt[current_idx] = 1e-10;
            }
        }
    }
    // Force unstable phase amounts to zero (matching CPU minimizer.pyx lines 1468-1469)
    for(int i=0; i<state->num_compsets; ++i) {
        bool is_still_free_stable = false;
        for(int j=0; j < state->num_free_stable_compsets; ++j) {
            if(state->free_stable_compset_indices[j] == i) {
                is_still_free_stable = true;
                break;
            }
        }
        if(!is_still_free_stable && !state->compsets[i].fixed) {
            state->phase_amt[i] = 0.0;
        }
    }
    return phases_changed;
}


// run_loop function removed - moved here from gpu_codegen.py

// --- GLOBAL MEMORY VERSION OF SOLVE_STATE ---
// This function implements solve_state using global memory arrays
__device__ void solve_state(
    SystemSpecification* spec,
    SystemState* state,
    double* out_equilibrium_soln,
    int soln_length,
    double* equilibrium_matrix,  // global memory
    double* equilibrium_rhs,     // global memory
    double* A_lstsq_copy,       // global memory for SVD
    double* U_lstsq,
    double* V_lstsq,
    double* singular_values_lstsq,
    double* superdiag_lstsq,
    int thread_id,              // Pass thread_id for debug output
    double* work_inv            // global memory for LU inversion scratch space
) {
    // IMPLEMENTATION: This mirrors the original solve_state but uses global memory arrays
    
    // Calculate matrix dimensions
    // Add +1 back to match CPU matrix dimensions exactly
    // CPU DOES include a system amount constraint row (with [1,1,1] for phase amounts)
    int equilibrium_matrix_rows = state->num_free_stable_compsets + 
                                 spec->num_fixed_stable_compsets + 
                                 spec->num_prescribed_mole_fraction_conditions + 1;
    // Use num_free_chemical_potentials which now equals ALL non-VA components
    // CPU uses all non-VA components as columns, not just mathematically independent ones
    // This fixes the matrix dimension mismatch (CPU 6x6 vs GPU 5x5 for ternary)
    int equilibrium_matrix_cols = spec->num_free_chemical_potentials +  // All non-VA components 
                                 state->num_free_stable_compsets + 
                                 spec->num_free_statevars;
    
    // DEBUG: Print matrix size calculation
    if (state->iteration < 5 && thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("[GPU MATRIX SIZE] Iteration %d: num_free_stable_compsets=%d, fixed=%d, mole_frac_conds=%d\n",
               state->iteration, state->num_free_stable_compsets, spec->num_fixed_stable_compsets,
               spec->num_prescribed_mole_fraction_conditions);
        printf("  Matrix dimensions: %dx%d (cols = %d + %d + %d)\n", 
               equilibrium_matrix_rows, equilibrium_matrix_cols,
               spec->num_free_chemical_potentials, state->num_free_stable_compsets, spec->num_free_statevars);
        #endif
    }
    
    // Call recompute at the beginning of solve_state, just like CPU does
    // This ensures all CompsetState arrays (masses, jacobians, energies) are up-to-date
    
    // DEBUG: Verify spec pointer before calling recompute
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: solve_state - spec=%p, spec->num_statevars=%d\n", 
               spec, spec->num_statevars);
        #endif
        if (spec->num_statevars < 0 || spec->num_statevars > 10) {
            printf("GPU ERROR: spec appears corrupted in solve_state!\n");
            printf("  spec->num_statevars=%d (0x%X)\n", spec->num_statevars, spec->num_statevars);
            printf("  spec->num_components=%d\n", spec->num_components);
            // Try to continue anyway
        }
    }
    
    #ifdef PYCGPU_PROF
    long long prof_ss_t0 = clock64();
    #endif
    state->recompute(spec, work_inv);
    #ifdef PYCGPU_PROF
    if (thread_id < PYCGPU_PROF_MAXT) g_prof_recompute[thread_id] += clock64() - prof_ss_t0;
    prof_ss_t0 = clock64();
    #endif

    // The old manual update loop is not needed since recompute handles everything
    
    // NOTE: Do NOT overwrite state->system_amount here. recompute() already set it to
    // sum(phase_amt * masses) (moles of atoms), matching CPU. Overwriting it with
    // sum(phase_amt) (formula units) makes the N-constraint converge to the wrong scale.
    
    // Manually zero the equilibrium matrix AND RHS before calling fill_equilibrium_system
    // This is needed because these arrays are in global memory and persist across iterations
    // When matrix size changes (e.g., 4x4 to 3x3 after phase consolidation), old values remain!
    for (int i = 0; i < equilibrium_matrix_rows * equilibrium_matrix_cols; ++i) {
        equilibrium_matrix[i] = 0.0;
    }
    for (int i = 0; i < equilibrium_matrix_rows; ++i) {
        equilibrium_rhs[i] = 0.0;
    }
    
    // Call fill_equilibrium_system with global memory arrays
    fill_equilibrium_system(equilibrium_matrix, equilibrium_matrix_cols,
                           equilibrium_rhs, spec, state);
    #ifdef PYCGPU_PROF
    if (thread_id < PYCGPU_PROF_MAXT) g_prof_fill[thread_id] += clock64() - prof_ss_t0;
    #endif
    
    // DEBUG: Check RHS after fill_equilibrium_system
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0 && state->iteration < 3) {
        printf("  RHS after fill_equilibrium_system: [");
        // Print ALL rows of RHS
        for (int i = 0; i < equilibrium_matrix_rows; ++i) {
            printf("%.2e", equilibrium_rhs[i]);
            if (i < equilibrium_matrix_rows - 1) printf(", ");
        }
        printf("]\n");
    }
    #endif
    
    // DEBUG: Disabled to avoid compilation issues
    // if (iteration_count == 0 && thread_id == 0) { printf("GPU DEBUG\n"); }
    
    // Use global memory arrays for SVD solve
    // This replaces the local arrays that were causing stack overflow
    
    // First copy equilibrium_matrix to A_lstsq_copy to preserve original
    int matrix_size = equilibrium_matrix_rows * equilibrium_matrix_cols;
    for (int i = 0; i < matrix_size; ++i) {
        A_lstsq_copy[i] = equilibrium_matrix[i];
    }
    
    // DEBUG: Check matrix dimensions and content before SVD
    // Note: state->iteration might be available instead of iteration_count
    if (thread_id == 0 && state->iteration < 3) {
        #ifdef VERBOSE_DEBUG
        printf("\n[EQUILIBRIUM_MATRIX_OUTPUT] GPU Iteration %d (rows=%d, cols=%d):\n", 
               state->iteration, equilibrium_matrix_rows, equilibrium_matrix_cols);
        // ALWAYS print ALL rows of the matrix
        for (int i = 0; i < equilibrium_matrix_rows; ++i) {
            printf("  Row %d: ", i);
            // Print ALL columns too
            for (int j = 0; j < equilibrium_matrix_cols; ++j) {
                printf("%+.6e ", equilibrium_matrix[i * equilibrium_matrix_cols + j]);
            }
            printf("| RHS: %+.6e\n", equilibrium_rhs[i]);
        }
        printf("  system_amount=%.6f, prescribed=%.6f\n", 
               state->system_amount, spec->prescribed_system_amount);
        #endif
    }
    
    // Call lstsq with correct signature
    // Use same tolerance as CPU (1e-16) instead of 1e-12
    #ifdef PYCGPU_PROF
    prof_ss_t0 = clock64();
    #endif
    lstsq(A_lstsq_copy, equilibrium_matrix_rows, equilibrium_matrix_cols,
          equilibrium_rhs, 1e-16,
          U_lstsq, V_lstsq, singular_values_lstsq, superdiag_lstsq);
    #ifdef PYCGPU_PROF
    if (thread_id < PYCGPU_PROF_MAXT) g_prof_lstsq[thread_id] += clock64() - prof_ss_t0;
    #endif
    #ifdef PYCGPU_FP32EMU
    pycgpu_f32_arr(equilibrium_rhs, equilibrium_matrix_cols);
    #endif
    
    // The solution should be in equilibrium_rhs after lstsq completes
    
    // DEBUG: Check if lstsq produced a non-zero solution
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0 && state->iteration < 3) {
        printf("  RHS after lstsq (solution): [");
        // Print ALL solution values
        for (int i = 0; i < equilibrium_matrix_cols; ++i) {
            printf("%.2e", equilibrium_rhs[i]);
            if (i < equilibrium_matrix_cols - 1) printf(", ");
        }
        printf("]\n");
    }
    #endif
    // Copy back to output solution
    for (int i = 0; i < soln_length && i < equilibrium_matrix_cols; ++i) {
        out_equilibrium_soln[i] = equilibrium_rhs[i];
    }
    
    // Update chemical potentials from the solution
    // The equilibrium solution contains NEW chemical potential values (not deltas)
    // This matches CPU behavior at minimizer.pyx line 1250
    for (int i = 0; i < spec->num_free_chemical_potentials; ++i) {
        int chempot_idx = spec->free_chemical_potential_indices[i];
        state->chemical_potentials[chempot_idx] = out_equilibrium_soln[i];
    }
    
    // Force fixed chemical potentials to adopt their fixed values
    for (int i = 0; i < spec->num_fixed_chemical_potentials; ++i) {
        int comp_idx = spec->fixed_chemical_potential_indices[i];
        if (comp_idx >= 0 && comp_idx < spec->num_components) {
            state->chemical_potentials[comp_idx] = spec->initial_chemical_potentials[comp_idx];
        }
    }
    
    // Calculate largest chemical potential difference for convergence check
    state->largest_chemical_potential_difference = -INFINITY;
    for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
        double diff = fabs(state->chemical_potentials[comp_idx] - state->previous_chemical_potentials[comp_idx]);
        if (diff > state->largest_chemical_potential_difference) {
            state->largest_chemical_potential_difference = diff;
        }
    }
}

// --- GLOBAL MEMORY VERSION OF RUN_LOOP ---
// This function implements the sophisticated run_loop using global memory arrays
// to avoid stack overflow while maintaining all the sophisticated solver logic
__device__ bool run_loop(
    int thread_id,              // Add thread_id parameter for debug output
    SystemSpecification* spec,
    SystemState* state,
    int max_iterations,
    const DeviceGrid* grid_data,     // Add grid data for phase search
    const DevicePhaseData* phase_data,  // Add phase data for phase search
    // Global memory arrays to replace stack arrays
    double* equilibrium_matrix,  // replaces local equilibrium matrix
    double* equilibrium_rhs,     // replaces local equilibrium RHS
    double* eq_soln,            // replaces local solution vector
    double* A_lstsq_copy,       // replaces local SVD arrays
    double* U_lstsq,
    double* V_lstsq,
    double* singular_values_lstsq,
    double* superdiag_lstsq,
    double* masses,             // replaces local masses arrays
    double* mass_jac,           // replaces local jacobian arrays
    double* x_dof,              // replaces local DOF arrays
    double* grad,               // replaces local gradient arrays
    double* hess,               // replaces local hessian arrays
    double* work_inv            // global memory for LU inversion scratch space
) {
    // IMPLEMENTATION: This mirrors the original run_loop but uses global memory arrays
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: run_loop STARTED with max_iterations=%d\n", max_iterations);
        #endif
    }
    
    double step_size = 1.0;
    bool converged = false;
    bool phases_changed_iter;

    #ifdef PYCGPU_PROF
    // Per-thread segment cycle counters (enable with PYCGPU_PROF=1; prints one
    // line per thread at loop exit). Used to attribute kernel wall time.
    long long prof_solve = 0, prof_rc = 0, prof_chg = 0, prof_adv = 0, prof_t0 = 0;
    int prof_iters = 0, prof_chg_calls = 0;
    if (thread_id < PYCGPU_PROF_MAXT) {
        g_prof_recompute[thread_id] = 0;
        g_prof_fill[thread_id] = 0;
        g_prof_lstsq[thread_id] = 0;
        g_prof_hess[thread_id] = 0;
        g_prof_inv[thread_id] = 0;
        g_prof_funcs[thread_id] = 0;
    }
    #endif

    // Use global memory for eq_soln instead of local array
    int eq_soln_len;
    
    // DEBUG: Store initial values before loop starts (removed debug_gm_history references)
    // The debug variables debug_gm_history and debug_max_steps are not available in this function scope
    
    const int DEBUG_ENABLED = 0;  // Set to 1 to enable debug output
    
    for (int iteration_count = 0; iteration_count < max_iterations; ++iteration_count) {
        if (thread_id == 0 && iteration_count % 50 == 0) {
            #ifdef VERBOSE_DEBUG
            printf("\nGPU DEBUG: Iteration %d/%d\n", iteration_count, max_iterations);
            #endif
        }
        state->iteration = iteration_count;
        phases_changed_iter = false;
        
        // DEBUG: Mark that we entered the iteration loop (removed debug_gm_history references)
        if (DEBUG_ENABLED && thread_id == 0 && iteration_count == 0) {
            #ifdef VERBOSE_DEBUG
            printf("\nGPU DEBUG: ===== ITERATION 0 (DETAILED) =====\n");
            printf("GPU DEBUG: State before iteration:\n");
            printf("GPU DEBUG:   Chemical potentials: [%.6f, %.6f]\n", 
                   state->chemical_potentials[0], state->chemical_potentials[1]);
            #endif
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG:   Number of phases: %d\n", state->num_free_stable_compsets);
            printf("GPU DEBUG:   Free stable indices: ");
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                printf("%d ", state->free_stable_compset_indices[i]);
            }
            printf("\n");
            printf("GPU DEBUG:   System amount: %.6f\n", state->system_amount);
            printf("GPU DEBUG:   Mole fractions: [%.6f, %.6f]\n", 
                   state->mole_fractions[0], state->mole_fractions[1]);
            #endif
            
            // Details for each phase
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int idx = state->free_stable_compset_indices[i];
                CompositionSet* cs = &state->compsets[idx];
                CompsetState* css = &state->cs_states[idx];
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG:   Phase %d:\n", idx);
                printf("GPU DEBUG:     NP=%.6f\n", cs->NP);
                printf("GPU DEBUG:     phase_amt=%.6f (formula units)\n", state->phase_amt[idx]);
                printf("GPU DEBUG:     energy=%.6f\n", css->energy);
                printf("GPU DEBUG:     dof=[%.15f, %.15f, %.15f]\n", 
                       cs->dof[0], cs->dof[1], cs->dof[2]);
                printf("GPU DEBUG:     phase_compositions=[%.6f, %.6f]\n",
                       state->phase_compositions[idx * MAX_COMPONENTS + 0],
                       state->phase_compositions[idx * MAX_COMPONENTS + 1]);
                #endif
                // Calculate phase_comp_sum
                double phase_comp_sum = 0.0;
                for (int j = 0; j < spec->num_components; ++j) {
                    phase_comp_sum += state->phase_compositions[idx * MAX_COMPONENTS + j];
                }
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG:     phase_comp_sum=%.6f\n", phase_comp_sum);
                printf("GPU DEBUG:     phase_amt * phase_comp_sum=%.6f\n", 
                       state->phase_amt[idx] * phase_comp_sum);
                #endif
            }
        } else if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Entered iteration loop, max_iterations=%d\n", max_iterations);
            #endif
        }
        
        // SEGMENT 21: PRE-SOLVE HOOK
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 21: Pre-solve hook (condition %d, iteration %d)\n", thread_id, iteration_count);
            #endif
        }
        
        // Call pre_solve_hook (this should be safe, no large arrays)
        bool pre_hook_result = pre_solve_hook(spec, state);
        if (!pre_hook_result) {
            // DEBUG: Mark pre_solve_hook failure (removed debug_gm_history references)
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: pre_solve_hook failed!\n");
                #endif
            }
            break;
        }
        
        // SEGMENT 22: STATE RECOMPUTATION
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 22: State recomputation (condition %d)\n", thread_id);
            printf("[GPU]   num_phases_active: %d\n", state->num_free_stable_compsets);
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int idx = state->free_stable_compset_indices[i];
                CompositionSet* cs = &state->compsets[idx];
                printf("[GPU]   phase_%d: NP=%.15e, X=[%.6f, %.6f] (reading from indices %d, %d)\n",
                       idx, state->phase_amt[idx],
                       state->phase_compositions[idx * MAX_COMPONENTS + 0],
                       state->phase_compositions[idx * MAX_COMPONENTS + 1],
                       idx * MAX_COMPONENTS + 0,
                       idx * MAX_COMPONENTS + 1);
                if (idx == 1 && thread_id == 0 && iteration_count == 0) {
                    printf("[GPU]   DEBUG: phase_compositions array around phase 1:\n");
                    for (int j = 0; j < 8; ++j) {
                        printf("    [%d] = %f\n", j, state->phase_compositions[j]);
                    }
                }
            }
            #endif
        }
        
        // NOTE: recompute is called inside solve_state, matching CPU behavior
        // Do NOT call it here to avoid double recomputation
        
        eq_soln_len = spec->num_free_chemical_potentials + state->num_free_stable_compsets + spec->num_free_statevars;
        
        // DEBUG: Store eq_soln_len calculation (removed debug_gm_history references)
        if (thread_id == 0 && iteration_count == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: eq_soln_len=%d (chem_pot=%d + compsets=%d + statevars=%d)\n", 
                   eq_soln_len, spec->num_free_chemical_potentials, 
                   state->num_free_stable_compsets, spec->num_free_statevars);
            #endif
        }
        
        if (eq_soln_len > MAX_EQ_SOLN_LEN || eq_soln_len <= 0) {
            // DEBUG: Mark eq_soln_len failure (removed debug_gm_history references)
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: eq_soln_len check failed! eq_soln_len=%d, MAX_EQ_SOLN_LEN=%d\n", 
                       eq_soln_len, MAX_EQ_SOLN_LEN);
                #endif
            }
            converged = false;
            break;
        }
        
        // DEBUG: Mark that we passed eq_soln_len check (removed debug_gm_history references)
        
        // DEBUG: Before solve_state - check state (DISABLED)
        // if (thread_id == 0) {
        //     printf("GPU DEBUG iter %d: num_compsets=%d, num_free_stable=%d\n", 
        //            iteration_count, state->num_compsets, state->num_free_stable_compsets);
        // }
        
        // SEGMENT 27-30: SOLVE STATE
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 27: Construct equilibrium system (condition %d)\n", thread_id);
            #endif
        }
        
        // Call solve_state with global memory arrays
        #ifdef PYCGPU_PROF
        prof_iters = iteration_count + 1;
        prof_t0 = clock64();
        #endif
        solve_state(spec, state, eq_soln, eq_soln_len,
                   equilibrium_matrix, equilibrium_rhs,
                   A_lstsq_copy, U_lstsq, V_lstsq,
                   singular_values_lstsq, superdiag_lstsq, thread_id,
                   work_inv);
        #ifdef PYCGPU_PROF
        prof_solve += clock64() - prof_t0;
        #endif
        
        // DEBUG: After solve_state
        if ((iteration_count < 3 || iteration_count % 50 == 0) && thread_id == 0) { 
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Equilibrium solution at iteration %d (len=%d): [", iteration_count, eq_soln_len);
            for (int i = 0; i < eq_soln_len && i < 10; ++i) {
                printf("%.6e", eq_soln[i]);
                if (i < eq_soln_len - 1) printf(", ");
            }
            if (eq_soln_len > 10) printf("...");
            printf("]\n");
            
            // Check if solution is all zeros
            bool all_zeros = true;
            for (int i = 0; i < eq_soln_len; ++i) {
                if (fabs(eq_soln[i]) > 1e-15) {
                    all_zeros = false;
                    break;
                }
            }
            if (all_zeros) {
                printf("GPU DEBUG: WARNING - Equilibrium solution is all zeros!\n");
            }
            
            printf("GPU DEBUG: After solve_state:\n");
            printf("GPU DEBUG:   Chemical potentials: [%.6f, %.6f]\n", 
                   state->chemical_potentials[0], state->chemical_potentials[1]);
            
            // Details for each phase after solve
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int idx = state->free_stable_compset_indices[i];
                CompositionSet* cs = &state->compsets[idx];
                printf("GPU DEBUG:   Phase %d:\n", idx);
                printf("GPU DEBUG:     phase_amt=%.6f (formula units)\n", state->phase_amt[idx]);
                printf("GPU DEBUG:     NP=%.6f\n", cs->NP);
                printf("GPU DEBUG:     dof=[%.15f, %.15f, %.15f]\n", 
                       cs->dof[0], cs->dof[1], cs->dof[2]);
            }
            #endif
        }
        
        // SEGMENT 33: POST SOLVE HOOK (matching CPU order)
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 33: Post solve hook\n");
            #endif
        }
        
        // Call post_solve_hook first (matching CPU behavior)
        if (!post_solve_hook(spec, state)) {
            if (thread_id < 3 && iteration_count < 3) {
                #ifdef VERBOSE_DEBUG
                printf("[GPU]   post_solve_hook_returned_false\n");
                #endif
            }
            break;
        }
        
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU]   post_solve_hook_returned_true\n");
            #endif
        }
        
        // SEGMENT 34: REMOVE AND CONSOLIDATE PHASES (before advance_state to match CPU)
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 34: Remove and consolidate phases\n");
            #endif
        }
        
        // NOTE: Phase compositions are calculated in solve_state->recompute()
        // We use those compositions for consolidation checks to match CPU behavior
        
        // Phase change operations (these should be safe, no large arrays).
        // CPU (minimizer.pyx run_loop) only removes/consolidates after 5
        // quiet iterations; removing earlier changes the trajectory on
        // shallow/degenerate surfaces (a floored compset must survive until
        // the removal step so the remaining set re-equilibrates the same
        // way — measured on Cr-Fe-Ni_shallow_bcc's two-compset BCC gap).
        #ifdef PYCGPU_PROF
        prof_t0 = clock64();
        #endif
        bool rc_phases_changed = false;
        if (state->iterations_since_last_phase_change >= 5) {
            rc_phases_changed = remove_and_consolidate_phases(spec, state);
        }
        #ifdef PYCGPU_PROF
        prof_rc += clock64() - prof_t0;
        #endif
        if (rc_phases_changed) {
            phases_changed_iter = true;
            if (thread_id < 3 && iteration_count < 3) {
                #ifdef VERBOSE_DEBUG
                printf("[GPU]   phases_removed: true\n");
                #endif
            }
        }
        
        // SEGMENT 32: CHECK CONVERGENCE
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 32: Check convergence\n");
            #endif
        }
        bool convergence_result = check_convergence(spec, state);
        if (thread_id == 0 && (iteration_count < 3 || iteration_count % 50 == 0 || convergence_result)) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Convergence check at iteration %d:\n", iteration_count);
            printf("  largest_phase_amt_change=%.2e (limit 1e-10)\n", state->largest_phase_amt_change);
            printf("  largest_y_change=%.2e (limit 5e-09)\n", state->largest_y_change);
            printf("  largest_statevar_change=%.2e (limit 1e-5)\n", state->largest_statevar_change);
            printf("  mass_residual=%.2e (limit %.2e)\n", state->mass_residual, spec->ALLOWED_MASS_RESIDUAL);
            printf("  iterations_since_last_phase_change=%d (need >=5)\n", state->iterations_since_last_phase_change);
            printf("  Converged: %s\n", convergence_result ? "YES" : "NO");
            #endif
        }
        
        if (convergence_result) {
            // Try to add phases if converged
            #ifdef PYCGPU_PROF
            prof_t0 = clock64();
            prof_chg_calls++;
            #endif
            bool chg_phases_changed = change_phases(spec, state, grid_data, phase_data);
            #ifdef PYCGPU_PROF
            prof_chg += clock64() - prof_t0;
            #endif
            if (chg_phases_changed) {
                phases_changed_iter = true;
                if (thread_id < 3 && iteration_count < 3) {
                    #ifdef VERBOSE_DEBUG
                    printf("[GPU]   phases_added: true\n");
                    #endif
                }
            }
            
            if (!phases_changed_iter) {
                // Truly converged with no phase changes
                converged = true;
                break;
            }
        }
        
        // Update phase change tracking. CPU (minimizer.pyx run_loop) is an
        // if/ELSE: the counter stays 0 through a phase-change iteration and
        // increments only on quiet iterations. Reset-then-always-increment
        // leaves the counter one ahead, so the >=5 removal and >=10
        // convergence gates fire one iteration early — early removal is
        // fatal for a freshly seeded compset whose decisive mass swing
        // happens on iteration 5 after the add (AlCuFe BCC_B2+LIQUID: the
        // seed was culled at 1e-16 right before the swing, four times, until
        // the removal budget ran out and the second phase was lost, +57 J/mol).
        if (phases_changed_iter) {
            state->iterations_since_last_phase_change = 0;
        } else {
            state->iterations_since_last_phase_change++;
        }
        // CPU minimizer.pyx:663 increments metastability counters every iteration;
        // without this, metastable_phase_iterations stays 0 and change_phases can
        // never re-add a removed phase.
        state->increment_phase_metastability_counters();
        
        // DEBUG: Before advance_state
        if (thread_id < 3 && iteration_count < 3) { 
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG iter %d: Before advance_state\n", iteration_count);
            printf("  Phase amounts: [%.6f, %.6f]\n", state->phase_amt[0], state->phase_amt[1]);
            printf("  eq_soln phase deltas: [%.6e, %.6e]\n", 
                   eq_soln[spec->num_free_chemical_potentials], 
                   eq_soln[spec->num_free_chemical_potentials + 1]);
            #endif
        }
        
        // SEGMENT 31: ADVANCE STATE (only if phases weren't changed)
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 31: Advance state\n");
            printf("[GPU]   step_size: %.6f\n", step_size);
            #endif
        }
        
        // Skip advance_state if phases changed (match CPU behavior)
        if (!phases_changed_iter) {
            // CPU (minimizer.pyx run_loop): step size ramps up over the first
            // 20 iterations of a solve — step = min(1.0, (iteration+1)/20).
            // Without the ramp, iteration-0 full steps crush freshly seeded
            // compsets to the clamp before they can equilibrate (measured on
            // issue589's 3-way FCC miscibility gap: reference converges to 3
            // compsets from the 5-vertex start, unramped backend collapses
            // to 2 and loses 643 J/mol).
            {
                double step_size_initial = 1.0 * (state->iteration + 1) / 20.0;
                step_size = fmin(1.0, step_size_initial);
            }
            // Call advance_state (this should be safe, no large arrays)
            #ifdef PYCGPU_PROF
            prof_t0 = clock64();
            #endif
            advance_state(spec, state, eq_soln, eq_soln_len, step_size);
            #ifdef PYCGPU_PROF
            prof_adv += clock64() - prof_t0;
            #endif
        } else {
            if (thread_id < 3 && iteration_count < 3) {
                #ifdef VERBOSE_DEBUG
                printf("[GPU] SKIPPING advance_state due to phase changes\n");
                #endif
            }
        }

        #ifdef PYCGPU_TRACE_LOOP
        // debug tracing twin of the reference PYCALPHAD_TRACE_LOOP print
        if (thread_id == 0) {
            printf("GPUTRACE iter=%d changed=%d amt=", iteration_count, (int)phases_changed_iter);
            for (int i = 0; i < state->num_compsets; ++i)
                printf("%s%.17g", i ? "," : "", state->phase_amt[i]);
            printf(" mu=");
            for (int i = 0; i < spec->num_components; ++i)
                printf("%s%.17g", i ? "," : "", state->chemical_potentials[i]);
            printf("\n");
        }
        #endif
        
        // DEBUG: Add detailed output after first iteration
        if (iteration_count == 0 && thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("\n[GPU TRACE] ===== AFTER ITERATION 0 =====\n");
            printf("[GPU TRACE] Chemical potentials: [");
            for (int i = 0; i < spec->num_components; ++i) {
                printf("%.15e", state->chemical_potentials[i]);
                if (i < spec->num_components - 1) printf(", ");
            }
            printf("]\n");
            printf("[GPU TRACE] System amount: %.15e\n", state->system_amount);
            printf("[GPU TRACE] Mole fractions: [");
            for (int i = 0; i < spec->num_components; ++i) {
                printf("%.15e", state->mole_fractions[i]);
                if (i < spec->num_components - 1) printf(", ");
            }
            printf("]\n");
            printf("[GPU TRACE] Mass residual: %.15e\n", state->mass_residual);
            printf("[GPU TRACE] Number of active phases: %d\n", state->num_free_stable_compsets);
            printf("[GPU TRACE] Free stable indices: [");
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                printf("%d", state->free_stable_compset_indices[i]);
                if (i < state->num_free_stable_compsets - 1) printf(", ");
            }
            printf("]\n");
            #endif
            
            #ifdef VERBOSE_DEBUG
            for (int idx = 0; idx < state->num_free_stable_compsets; ++idx) {
                int cs_idx = state->free_stable_compset_indices[idx];
                CompositionSet* compset = &state->compsets[cs_idx];
                CompsetState* csst = &state->cs_states[cs_idx];
                
                printf("\n[GPU TRACE] Phase %d:\n", cs_idx);
                printf("  NP (mole fraction): %.15e\n", compset->NP);
                printf("  phase_amt (formula units): %.15e\n", state->phase_amt[cs_idx]);
                printf("  energy: %.15e\n", csst->energy);
                printf("  phase_compositions: [");
                for (int c = 0; c < spec->num_components; ++c) {
                    printf("%.15e", state->phase_compositions[cs_idx * MAX_COMPONENTS + c]);
                    if (c < spec->num_components - 1) printf(", ");
                }
                printf("]\n");
                double phase_comp_sum = 0.0;
                for (int c = 0; c < spec->num_components; ++c) {
                    phase_comp_sum += state->phase_compositions[cs_idx * MAX_COMPONENTS + c];
                }
                printf("  phase_comp_sum: %.15e\n", phase_comp_sum);
                printf("  Site fractions: [");
                for (int sf = 0; sf < compset->phase_record->phase_dof; ++sf) {
                    printf("%.15e", compset->dof[spec->num_statevars + sf]);
                    if (sf < compset->phase_record->phase_dof - 1) printf(", ");
                }
                printf("]\n");
                printf("  State variables: [");
                for (int sv = 0; sv < spec->num_statevars; ++sv) {
                    printf("%.15e", compset->dof[sv]);
                    if (sv < spec->num_statevars - 1) printf(", ");
                }
                printf("]\n");
            }
            #endif
            
            #ifdef VERBOSE_DEBUG
            printf("\n[GPU TRACE] Convergence status:\n");
            printf("  converged: %s\n", converged ? "true" : "false");
            printf("  phases_changed: %s\n", phases_changed_iter ? "true" : "false");
            printf("  largest_phase_amt_change: %.15e\n", state->largest_phase_amt_change);
            printf("  largest_y_change: %.15e\n", state->largest_y_change);
            printf("  largest_statevar_change: %.15e\n", state->largest_statevar_change);
            printf("[GPU TRACE] ===== END ITERATION 0 =====\n\n");
            #endif
        }
    }

    #ifdef PYCGPU_PROF
    printf("[PROF] tid=%d iters=%d conv=%d solve=%.1f rc=%.1f chg=%.1f(calls=%d) adv=%.1f | recompute=%.1f fill=%.1f lstsq=%.1f hess=%.1f inv=%.1f funcs=%.1f Mcyc\n",
           thread_id, prof_iters, converged ? 1 : 0,
           prof_solve / 1e6, prof_rc / 1e6, prof_chg / 1e6, prof_chg_calls, prof_adv / 1e6,
           thread_id < PYCGPU_PROF_MAXT ? g_prof_recompute[thread_id] / 1e6 : -1.0,
           thread_id < PYCGPU_PROF_MAXT ? g_prof_fill[thread_id] / 1e6 : -1.0,
           thread_id < PYCGPU_PROF_MAXT ? g_prof_lstsq[thread_id] / 1e6 : -1.0,
           thread_id < PYCGPU_PROF_MAXT ? g_prof_hess[thread_id] / 1e6 : -1.0,
           thread_id < PYCGPU_PROF_MAXT ? g_prof_inv[thread_id] / 1e6 : -1.0,
           thread_id < PYCGPU_PROF_MAXT ? g_prof_funcs[thread_id] / 1e6 : -1.0);
    #endif

    return converged;
}

// SVD-based matrix functions using svd.c

/**
 * @brief Inverts a square matrix using SVD decomposition.
 * @param matrix Input/output matrix (N x N, stored row-major). Will be overwritten with inverse.
 * @param dim Dimension of the square matrix (N).
 * @param U Workspace for U matrix (N x N).
 * @param V Workspace for V matrix (N x N).
 * @param singular_values Workspace for singular values (N).
 * @param superdiag Workspace for super-diagonal (N).
 * @param work Additional workspace (N x N).
 */
/* legacy SVD-based invert_matrix removed; superseded by pyclap_invert_pycalphad */

/**
 * @brief Solves least squares problem Ax = b using SVD decomposition.
 * @param A Input matrix (M x N, stored row-major). Will be modified during computation.
 * @param nrows Number of rows in A (M).
 * @param ncols Number of columns in A (N).
 * @param b Input/output vector. Input: RHS vector (M), Output: solution vector (N).
 * @param tolerance Tolerance for SVD pseudo-inverse.
 * @param U Workspace for U matrix (M x N).
 * @param V Workspace for V matrix (N x N).
 * @param singular_values Workspace for singular values (N).
 * @param superdiag Workspace for super-diagonal (N).
 */
__device__ void lstsq(double* A, int nrows, int ncols, double* b, double tolerance,
                      double* U, double* V, double* singular_values, double* superdiag) {
    // Reference-LAPACK dgelsd chain (bitwise vs compiled netlib), with the
    // reference wrapper's semantics (NaN scrub -> zeros, failure -> -1e19,
    // relative rcond).  Buffers repurposed from the caller's per-thread
    // slices: V = column-major copy, singular_values = SVs, U = workspace.
    // Shapes outside the ported base case (n > 25 or non-square) cannot
    // reach the kernel — the dispatch gate bounds the system size — but if
    // they ever did, the reference failure sentinels make it loud.
    (void)tolerance;  // the chain applies dgelsd's relative rcond internally
    (void)superdiag;
    if (ncols > 120) {
        // Per-thread LAPACK workspace (the U slice) fits n <= ~120; a
        // 120-dimensional equilibrium system implies a ~60-component
        // database, far beyond anything physical.  Loud, not silent.
        for (int i = 0; i < ncols; ++i) b[i] = -1e19;
        return;
    }
    int rc = pyclap_lstsq_pycalphad(A, nrows, ncols, b, 1e-16,
                                    V, singular_values, U, (int*)0);
    if (rc == PYCLAP_ERR_DC_UNPORTED || rc == PYCLAP_ERR_NOT_SQUARE) {
        for (int i = 0; i < ncols; ++i) b[i] = -1e19;
    }
}
/* ==== jansson derivative epilogue (folded from jansson.h; the separate
 * file hit stale-read behavior on /mnt/c for newly created files) ==== */
/* jansson.h — Jansson derivative deltas at a converged equilibrium state.
 *
 * Faithful port of the reference chain (Sundman et al. 2015):
 *   state_variable_differential  (minimizer.pyx:699, Eq. 74)
 *   site_fraction_differential   (minimizer.pyx:825, Eq. 78)
 *
 * Runs as an epilogue after run_loop converges, inside the same thread and
 * on the same recompute'd SystemState, so every matrix it needs (per-compset
 * c_statevars / c_component from the e-matrix, the equilibrium-system
 * machinery) is already live.  The property-side chain rule (Eq. 73) is done
 * batched on the host: these deltas are per-condition solver outputs.
 *
 * Output layout per condition (PYJAN_OUT_STRIDE doubles):
 *   [0 .. MAX_COMPONENTS)                       delta chemical potentials
 *   [MAX_COMPONENTS .. +MAX_STATEVARS)          delta state variables
 *   [.. +MAX_PHASES)                            delta phase amounts (moles
 *                                               of formula units, matching
 *                                               state->phase_amt convention)
 *   [.. +MAX_PHASES*MAX_DOF_PER_PHASE)          delta site fractions per
 *                                               compset (compset-major)
 *   [last slot]                                 status: 1.0 ok, 0.0 failed
 */

#define PYJAN_OUT_STRIDE (MAX_COMPONENTS + MAX_STATEVARS + MAX_PHASES + \
                          MAX_PHASES * MAX_DOF_PER_PHASE + 1)

/* Reference site_fraction_differential (Eq. 78): delta_y for one compset
 * from the converged c_statevars / c_component blocks. */
__device__ static void pyjan_site_fraction_differential(
    const SystemSpecification* spec, const CompsetState* csst, int num_phase_dof,
    const double* delta_chempots, const double* delta_statevars, double* delta_y)
{
    for (int i = 0; i < num_phase_dof; ++i) {
        double acc = 0.0;
        for (int sv = 0; sv < spec->num_statevars; ++sv) {
            acc += csst->c_statevars[i * csst->c_statevars_cols + sv]
                   * delta_statevars[sv];
        }
        for (int cp = 0; cp < spec->num_components; ++cp) {
            acc += csst->c_component[cp * csst->c_component_cols + i]
                   * delta_chempots[cp];
        }
        delta_y[i] = acc;
    }
}

/* Reference state_variable_differential (Eq. 74): free the target state
 * variable, rebuild the equilibrium system with one reserved row pinning
 * delta(target) = 1, solve, and read off the deltas.
 *
 * Mutates a THREAD-LOCAL copy of the spec's statevar index arrays and
 * restores them before returning (mirroring the reference's try/finally).
 * Returns 1 on success, 0 on failure (dimension overflow / solve failure).
 */
__device__ static int pyjan_state_variable_differential(
    SystemSpecification* spec, SystemState* state, int target_statevar_index,
    double* equilibrium_matrix, double* equilibrium_rhs,
    double* U, double* V, double* singular_values, double* superdiag,
    double* delta_chemical_potentials, double* delta_statevars,
    double* delta_phase_amounts)
{
    int i, j;
    for (i = 0; i < spec->num_components; ++i) delta_chemical_potentials[i] = 0.0;
    for (i = 0; i < spec->num_statevars; ++i)  delta_statevars[i] = 0.0;
    for (i = 0; i < state->num_compsets; ++i)  delta_phase_amounts[i] = 0.0;

    /* Save original fixed/free statevar index sets. */
    int orig_fixed[MAX_STATEVARS], orig_free[MAX_STATEVARS];
    int orig_num_fixed = spec->num_fixed_statevars;
    int orig_num_free  = spec->num_free_statevars;
    for (i = 0; i < orig_num_fixed; ++i) orig_fixed[i] = spec->fixed_statevar_indices[i];
    for (i = 0; i < orig_num_free;  ++i) orig_free[i]  = spec->free_statevar_indices[i];

    /* fixed := setdiff(fixed, {target}); free := append(free, target).
     * (The reference appends, so the target is the LAST free statevar and
     * therefore the LAST column of the system — relied on below.) */
    {
        int w = 0;
        for (i = 0; i < orig_num_fixed; ++i) {
            if (spec->fixed_statevar_indices[i] != target_statevar_index) {
                spec->fixed_statevar_indices[w++] = spec->fixed_statevar_indices[i];
            }
        }
        spec->num_fixed_statevars = w;
        spec->free_statevar_indices[spec->num_free_statevars++] = target_statevar_index;
    }

    int ok = 0;
    {
        /* Reference construct_equilibrium_system(spec, state, 1):
         * rows = num_stable + num_fixed_phases + num_molefrac_conds
         *        + num_reserved(1) + 1
         * cols = free_chempots + num_stable + free_statevars(now +1)
         * and requires rows == cols (Gibbs phase rule). */
        int num_stable = state->num_free_stable_compsets;
        int num_fixed_ph = spec->num_fixed_stable_compsets;
        int num_mf = spec->num_prescribed_mole_fraction_conditions;
        int rows = num_stable + num_fixed_ph + num_mf + 1 + 1;
        int cols = spec->num_free_chemical_potentials + num_stable
                   + spec->num_free_statevars;
        if (rows == cols && rows <= MAX_SVD_M && cols <= MAX_SVD_N) {
            for (i = 0; i < rows * cols; ++i) equilibrium_matrix[i] = 0.0;
            for (i = 0; i < rows; ++i) equilibrium_rhs[i] = 0.0;
            fill_equilibrium_system(equilibrium_matrix, cols, equilibrium_rhs, spec, state);
            /* Zero the RHS again (fill writes the Newton RHS; the
             * differential wants a unit perturbation only), then pin the
             * reserved last row: matrix[-1, -1] = 1, rhs[-1] = 1. */
            for (i = 0; i < rows; ++i) equilibrium_rhs[i] = 0.0;
            for (j = 0; j < cols; ++j) equilibrium_matrix[(rows - 1) * cols + j] = 0.0;
            equilibrium_matrix[(rows - 1) * cols + (cols - 1)] = 1.0;
            equilibrium_rhs[rows - 1] = 1.0;

#ifdef PYCGPU_TRACE_LOOP
            printf("JANSPEC cond=%d rows=%d cols=%d stable=%d fixedph=%d mf=%d fcp=%d fsv=%d\n",
                   state->condition_idx, rows, cols, num_stable, num_fixed_ph, num_mf,
                   spec->num_free_chemical_potentials, spec->num_free_statevars);
            if (state->condition_idx == 1) {
                printf("JANMAT rows=%d cols=%d\n", rows, cols);
                for (i = 0; i < rows; ++i) {
                    printf("JANROW %d:", i);
                    for (j = 0; j < cols; ++j)
                        printf(" %.17g", equilibrium_matrix[i * cols + j]);
                    printf(" | %.17g\n", equilibrium_rhs[i]);
                }
            }
#endif
#ifdef PYCGPU_JANSSON_MATDUMP
            /* debug: expose the assembled system through the output buffer
             * (delta_y region, unused during diagnosis) */
            {
                double* dump = delta_phase_amounts + MAX_PHASES;  /* d_y area */
                int cap = MAX_PHASES * MAX_DOF_PER_PHASE;
                int k = 0;
                dump[k++] = (double)rows; dump[k++] = (double)cols;
                dump[k++] = (double)state->num_free_stable_compsets;
                dump[k++] = (double)state->free_stable_compset_indices[0];
                dump[k++] = (double)state->num_compsets;
                {
                    int cs0 = state->free_stable_compset_indices[0];
                    CompsetState* c0 = &state->cs_states[cs0];
                    dump[k++] = c0->masses[0];
                    dump[k++] = c0->masses[1];
                    dump[k++] = c0->grad[2];
                }
                for (i = 0; i < rows * cols && k < cap; ++i)
                    dump[k++] = equilibrium_matrix[i];
            }
#endif
            lstsq(equilibrium_matrix, rows, cols, equilibrium_rhs, 1e-16,
                  U, V, singular_values, superdiag);

            for (i = 0; i < spec->num_free_chemical_potentials; ++i) {
                int cp_idx = spec->free_chemical_potential_indices[i];
                delta_chemical_potentials[cp_idx] = equilibrium_rhs[i];
            }
            for (i = 0; i < num_stable; ++i) {
                int cs_idx = state->free_stable_compset_indices[i];
                delta_phase_amounts[cs_idx] =
                    equilibrium_rhs[spec->num_free_chemical_potentials + i];
            }
            for (i = 0; i < spec->num_free_statevars; ++i) {
                int sv_idx = spec->free_statevar_indices[i];
                delta_statevars[sv_idx] =
                    equilibrium_rhs[spec->num_free_chemical_potentials + num_stable + i];
            }
            ok = 1;
        }
    }

    /* Restore the spec (reference: finally block). */
    spec->num_fixed_statevars = orig_num_fixed;
    spec->num_free_statevars  = orig_num_free;
    for (i = 0; i < orig_num_fixed; ++i) spec->fixed_statevar_indices[i] = orig_fixed[i];
    for (i = 0; i < orig_num_free;  ++i) spec->free_statevar_indices[i]  = orig_free[i];
    return ok;
}

/* Epilogue driver: called once per condition after convergence.  Writes the
 * per-condition delta block to jansson_out (already offset per thread). */
__device__ static void pyjan_compute_deltas(
    SystemSpecification* spec, SystemState* state, int target_statevar_index,
    double* equilibrium_matrix, double* equilibrium_rhs,
    double* U, double* V, double* singular_values, double* superdiag,
    double* jansson_out)
{
    double* d_mu  = jansson_out;
    double* d_sv  = jansson_out + MAX_COMPONENTS;
    double* d_amt = jansson_out + MAX_COMPONENTS + MAX_STATEVARS;
    double* d_y   = jansson_out + MAX_COMPONENTS + MAX_STATEVARS + MAX_PHASES;
    double* status = jansson_out + (PYJAN_OUT_STRIDE - 1);

    for (int i = 0; i < PYJAN_OUT_STRIDE; ++i) jansson_out[i] = 0.0;

    int ok = pyjan_state_variable_differential(
        spec, state, target_statevar_index, equilibrium_matrix, equilibrium_rhs,
        U, V, singular_values, superdiag, d_mu, d_sv, d_amt);
    if (!ok) { *status = 0.0; return; }

#ifndef PYCGPU_JANSSON_MATDUMP
    for (int cs = 0; cs < state->num_compsets && cs < MAX_PHASES; ++cs) {
        const CompositionSet* compset = &state->compsets[cs];
        if (compset->phase_record == (const PhaseRecord*)0) continue;
        pyjan_site_fraction_differential(
            spec, &state->cs_states[cs], compset->phase_record->phase_dof,
            d_mu, d_sv, &d_y[cs * MAX_DOF_PER_PHASE]);
    }
#endif
    *status = 1.0;
}
