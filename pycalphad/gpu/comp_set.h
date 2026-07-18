#ifndef MAX_PARAMS
#define MAX_PARAMS 0
#endif

#pragma once
#include "phase_rec.h"

// Use consistent constant names with minimizer.h
#ifndef MAX_DOF_PER_PHASE
#define MAX_DOF_PER_PHASE 64
#endif

#ifndef MAX_COMPONENTS  
#define MAX_COMPONENTS 32
#endif

#ifndef MAX_STATEVARS
#define MAX_STATEVARS 8
#endif

// Phase-local conditions (conditions with a phase name, e.g. X(FCC_A1,ZN)
// or Y(LIQUID,0,ZN)): each one borders the owning compset's phase matrix
// with a constraint-jacobian row, exactly like an internal constraint
// (reference compute_phase_matrix, minimizer.pyx:98). Sized per system by
// compute_dynamic_kernel_sizes; 0 when no phase-local conditions exist.
#ifndef MAX_PHASE_LOCAL_CONDITIONS
#define MAX_PHASE_LOCAL_CONDITIONS 0
#endif
// Zero-length arrays are invalid C++; keep a 1-slot floor for the field
// declarations (num_phase_local_conditions still reads 0).
#define PYCGPU_PLC_CAP (MAX_PHASE_LOCAL_CONDITIONS > 0 ? MAX_PHASE_LOCAL_CONDITIONS : 1)

// Legacy aliases for compatibility
#define NDOF_MAX MAX_DOF_PER_PHASE
#define NELEM_MAX MAX_COMPONENTS

struct CompositionSet {
    const PhaseRecord* phase_record;
    // DOF array must be large enough for workspace state variables + phase DOF
    // With workspace having 3 state vars (N, P, T) and phase having 2 site fractions,
    // we need at least 5 elements. But MAX_DOF_PER_PHASE might be set to 4.
    // Increase the size to handle this case.
    // Trailing MAX_PARAMS slots hold runtime fit-parameter values (see
    // gpu_codegen._fit_parameter_symbols): generated functions read them as
    // x[num_statevars + phase_dof + j]. Never touched by the solver loops.
    double dof[MAX_STATEVARS + MAX_DOF_PER_PHASE + MAX_PARAMS];
    double X[MAX_COMPONENTS];
    double energy;
    double NP;
    bool fixed;
    // Phase-local conditions attached to this compset (reference:
    // CompositionSet.set_local_conditions). type 0 = mole fraction
    // (target = nonvacant component index; jacobian built from mass_jac and
    // the moles-normalization gradient), type 1 = site fraction (target =
    // site-fraction dof index; unit jacobian row).
    int num_phase_local_conditions;
    int plc_type[PYCGPU_PLC_CAP];
    int plc_target[PYCGPU_PLC_CAP];
    double plc_value[PYCGPU_PLC_CAP];

    __device__ CompositionSet() : phase_record(nullptr), energy(0.0), NP(0.0), fixed(false),
                                  num_phase_local_conditions(0) {
        for(int i = 0; i < MAX_STATEVARS + MAX_DOF_PER_PHASE; i++) dof[i] = 0.0;
        for(int i = 0; i < MAX_COMPONENTS; i++) X[i] = 0.0;
    }

    __device__ void init(const PhaseRecord* pr) {
        phase_record = pr;
    }

    __device__ void update(double* site_fracs, double phase_amt, double* state_variables, int workspace_num_statevars) {
        // With the updated energy functions that accept all state variables,
        // we now pass the full workspace state variables array directly to the energy functions.
        // The energy functions now expect [N, P, T, Y1, Y2...] format.
        
        // Store the full workspace DOF array
        // Copy all workspace state variables
        for(int i = 0; i < workspace_num_statevars; i++) {
            dof[i] = state_variables[i];
        }
        // Copy site fractions after workspace state variables
        for(int i = 0; i < phase_record->phase_dof; i++) {
            dof[workspace_num_statevars + i] = site_fracs[i];
        }
        
        NP = phase_amt;
        // Pass full workspace DOF to energy functions - they now expect workspace format
        energy = phase_record->obj(dof);
        #ifdef PYCGPU_FP32EMU
        energy = pycgpu_f32(energy);
        #endif
        
        // DEBUG: Check memory before mass_obj
        #ifdef VERBOSE_DEBUG
        if (workspace_num_statevars == 3) {
            printf("GPU DEBUG: update() before mass_obj - workspace_num_statevars still = %d\\n", workspace_num_statevars);
        }
        #endif
        
        phase_record->mass_obj(X, dof);  // Fills entire X array at once
        
        // DEBUG: Check memory after mass_obj
        #ifdef VERBOSE_DEBUG
        if (workspace_num_statevars != 3) {
            printf("GPU ERROR: update() after mass_obj - workspace_num_statevars corrupted to %d!\\n", workspace_num_statevars);
        }
        #endif
    }
    
    __device__ double calculate_phase_comp_sum(int workspace_num_statevars) {
        // Calculate sum of moles of atoms per formula unit using formulamole_obj
        // This matches CPU's formulamole_obj usage for phase normalization
        double formulamoles[MAX_COMPONENTS];
        
        // Initialize all to zero since formulamole_obj only fills nonvacant elements
        for(int i = 0; i < MAX_COMPONENTS; ++i) {
            formulamoles[i] = 0.0;
        }
        
        // Call formulamole_obj which gives moles of each component per formula unit
        if (phase_record && phase_record->formulamole_obj) {
            // With updated energy functions, pass the full DOF array directly
            phase_record->formulamole_obj(formulamoles, dof);
            #ifdef PYCGPU_FP32EMU
            pycgpu_f32_arr(formulamoles, phase_record->nonvacant_elements > 0 ? phase_record->nonvacant_elements : phase_record->num_elements);
            #endif
        }
        
        // Sum up the moles of atoms per formula unit
        double phase_sum = 0.0;
        for(int i = 0; i < phase_record->nonvacant_elements; ++i) {
            phase_sum += formulamoles[i];
        }
        
        return phase_sum;
    }
};
