#ifndef PYCGPU_HYPERPLANE_H
#define PYCGPU_HYPERPLANE_H
// Device port of pycalphad/core/hyperplane.pyx (lower-convex-hull tangent
// search). One call = one condition; embarrassingly parallel across
// conditions (one thread per condition in point_hull_kernel).
//
// Port rules (CLAUDE.md): no dynamic allocation — every scratch array is a
// stack array sized by MAX_COMPONENTS; array lengths are explicit arguments;
// 1D indexing throughout. The reference's driving_forces[num_points] buffer
// is streamed instead of stored: each point's value is accumulated in the
// SAME per-point operation order as the reference's column-wise loops, so
// the values (and the strict-< argmin) are bit-identical without the O(M)
// allocation.

#ifndef MAX_COMPONENTS
#define MAX_COMPONENTS 8
#endif

// LAPACK dgesv semantics for tiny N (unblocked dgetf2 + dgetrs), on a
// COLUMN-major A (the .pyx fills its matrices Fortran-order for dgesv).
// Bit-parity notes: pivot = FIRST index of max |A(i,j)| (idamax), column
// scaling multiplies by the reciprocal of the pivot (dgetf2's dscal path),
// dger updates the trailing submatrix column-by-column. Singular matrices
// (exact zero pivot) set every x[i] to -1e19, matching hyperplane.pyx.
__device__ static void pycgpu_hp_solve(double* A, int N, double* x)
{
    int ipiv[MAX_COMPONENTS];
    int info = 0;
    for (int j = 0; j < N; ++j) {
        int p = j;
        double amax = fabs(A[j + j * N]);
        for (int i = j + 1; i < N; ++i) {
            double v = fabs(A[i + j * N]);
            if (v > amax) { amax = v; p = i; }
        }
        ipiv[j] = p;
        if (A[p + j * N] != 0.0) {
            if (p != j) {
                for (int k = 0; k < N; ++k) {
                    double t = A[j + k * N]; A[j + k * N] = A[p + k * N]; A[p + k * N] = t;
                }
            }
            double r = 1.0 / A[j + j * N];
            for (int i = j + 1; i < N; ++i) A[i + j * N] *= r;
            for (int k = j + 1; k < N; ++k) {
                double ajk = A[j + k * N];
                if (ajk != 0.0) {
                    for (int i = j + 1; i < N; ++i) A[i + k * N] -= A[i + j * N] * ajk;
                }
            }
        } else if (info == 0) {
            info = j + 1;
        }
    }
    if (info != 0) {
        for (int i = 0; i < N; ++i) x[i] = -1e19;
        return;
    }
    for (int j = 0; j < N; ++j) {
        if (ipiv[j] != j) { double t = x[j]; x[j] = x[ipiv[j]]; x[ipiv[j]] = t; }
    }
    for (int j = 0; j < N; ++j) {          // L (unit lower) forward
        double xj = x[j];
        if (xj != 0.0) for (int i = j + 1; i < N; ++i) x[i] -= xj * A[i + j * N];
    }
    for (int j = N - 1; j >= 0; --j) {      // U back-substitution
        x[j] /= A[j + j * N];
        double xj = x[j];
        if (xj != 0.0) for (int i = 0; i < j; ++i) x[i] -= xj * A[i + j * N];
    }
}

// hyperplane_coefficients (hyperplane.pyx:59): coefficients of the plane
// through the trial simplex (+ axis constraints for fixed chempots).
// compositions: (num_points x num_components) ROW-major.
__device__ static void pycgpu_hp_plane_coefficients(
    const double* compositions, int num_components,
    const int* fixed_chempot_indices, int num_fixed_chempots,
    const int* trial_simplex, int simplex_size,
    double* out_plane_coefs)
{
    const int plane_rows = simplex_size + num_fixed_chempots; // == num_components
    double f_plane_matrix[MAX_COMPONENTS * MAX_COMPONENTS];
    for (int i = 0; i < simplex_size; ++i) {
        for (int j = 0; j < num_components; ++j) {
            f_plane_matrix[i + j * plane_rows] =
                compositions[trial_simplex[i] * num_components + j];
        }
        out_plane_coefs[i] = 1;
    }
    for (int i = 0; i < num_fixed_chempots; ++i) {
        for (int j = 0; j < num_components; ++j) {
            f_plane_matrix[i + simplex_size + j * plane_rows] = 0;
        }
        f_plane_matrix[i + simplex_size + fixed_chempot_indices[i] * plane_rows] = 1;
        out_plane_coefs[i + simplex_size] = 0;
    }
    pycgpu_hp_solve(f_plane_matrix, plane_rows, out_plane_coefs);
}

// intersecting_point (hyperplane.pyx:84): where the condition constraints
// intersect the trial simplex's plane.
__device__ static void pycgpu_hp_intersecting_point(
    const double* compositions, int num_components,
    const int* fixed_chempot_indices, int num_fixed_chempots,
    const int* trial_simplex, int simplex_size,
    const double* fixed_lincomb_molefrac_coefs,  // (num_lincomb x N) ROW-major
    const double* fixed_lincomb_molefrac_rhs, int num_lincomb,
    double* out_intersecting_point)
{
    if (simplex_size == 1) {
        for (int i = 0; i < num_components; ++i) {
            out_intersecting_point[i] =
                compositions[trial_simplex[0] * num_components + i];
        }
        return;
    }
    double constraint_matrix[MAX_COMPONENTS * MAX_COMPONENTS];
    double constraint_rhs[MAX_COMPONENTS];
    for (int i = 0; i < num_components; ++i) out_intersecting_point[i] = 0;
    double* plane_coefs = out_intersecting_point;
    pycgpu_hp_plane_coefficients(compositions, num_components,
                                 fixed_chempot_indices, num_fixed_chempots,
                                 trial_simplex, simplex_size, plane_coefs);
    for (int j = 0; j < num_components; ++j) {
        for (int i = 0; i < num_lincomb; ++i) {
            constraint_matrix[i + j * num_components] =
                fixed_lincomb_molefrac_coefs[i * num_components + j];
            constraint_rhs[i] = fixed_lincomb_molefrac_rhs[i];
        }
        constraint_matrix[num_lincomb + j * num_components] = plane_coefs[j];
        constraint_rhs[num_lincomb] = 1;
    }
    pycgpu_hp_solve(constraint_matrix, num_components, constraint_rhs);
    for (int i = 0; i < num_components; ++i) out_intersecting_point[i] = constraint_rhs[i];
}

// simplex_fractions (hyperplane.pyx:120): barycentric fractions of the
// constraint intersection point within the trial simplex.
__device__ static void pycgpu_hp_simplex_fractions(
    const double* compositions, int num_components,
    const int* fixed_chempot_indices, int num_fixed_chempots,
    const int* trial_simplex, int simplex_size,
    const double* fixed_lincomb_molefrac_coefs,
    const double* fixed_lincomb_molefrac_rhs, int num_lincomb,
    const int* free_chempot_indices,   // ascending non-fixed indices
    double* out_fractions)
{
    double f_coord_matrix[MAX_COMPONENTS * MAX_COMPONENTS];
    double target_point[MAX_COMPONENTS];
    pycgpu_hp_intersecting_point(compositions, num_components,
                                 fixed_chempot_indices, num_fixed_chempots,
                                 trial_simplex, simplex_size,
                                 fixed_lincomb_molefrac_coefs,
                                 fixed_lincomb_molefrac_rhs, num_lincomb,
                                 target_point);
    for (int j = 0; j < simplex_size; ++j) {
        for (int i = 0; i < simplex_size; ++i) {
            f_coord_matrix[j + simplex_size * i] =
                compositions[trial_simplex[i] * num_components + free_chempot_indices[j]];
        }
        out_fractions[j] = target_point[free_chempot_indices[j]];
    }
    pycgpu_hp_solve(f_coord_matrix, simplex_size, out_fractions);
}

// hyperplane (hyperplane.pyx:146). compositions/energies are one condition's
// grid sample (num_points x num_components, row-major). chemical_potentials
// is in-out (fixed entries pre-set by the caller, like the reference).
// result_fractions / result_simplex have (num_components + 1) slots.
__device__ static double pycgpu_hyperplane(
    const double* compositions, const double* energies,
    int num_points, int num_components,
    double* chemical_potentials,
    const int* fixed_chempot_indices, int num_fixed_chempots,
    const double* fixed_lincomb_molefrac_coefs,
    const double* fixed_lincomb_molefrac_rhs, int num_lincomb,
    double* result_fractions, int* result_simplex)
{
    const int simplex_size = num_components - num_fixed_chempots;
    const int max_iterations = 1000;
    int iterations = 0;
    int saved_trial = 0;
    double out_energy = 0;

    int best_guess_simplex[MAX_COMPONENTS];
    int free_chempot_indices[MAX_COMPONENTS];
    int candidate_simplex[MAX_COMPONENTS];
    double candidate_potentials[MAX_COMPONENTS];
    double smallest_fractions[MAX_COMPONENTS];
    int trial_simplices[MAX_COMPONENTS * MAX_COMPONENTS];
    double fractions[MAX_COMPONENTS * MAX_COMPONENTS];
    double f_candidate_tieline[MAX_COMPONENTS * MAX_COMPONENTS];

    // Deviation from the reference (defensive only): the .pyx leaves
    // candidate_potentials as malloc garbage until the first solve; if the
    // very first fraction test breaks out, garbage would be copied into the
    // outputs there. Zero-init so that pathological path is deterministic.
    for (int i = 0; i < MAX_COMPONENTS; ++i) candidate_potentials[i] = 0.0;

    int fixed_index = 0;
    for (int i = 0; i < num_components; ++i) {
        bool skip_index = false;
        for (int j = 0; j < num_fixed_chempots; ++j) {
            if (i == fixed_chempot_indices[j]) skip_index = true;
        }
        if (!skip_index) best_guess_simplex[fixed_index++] = i;
    }
    for (int i = 0; i < simplex_size; ++i) {
        free_chempot_indices[i] = best_guess_simplex[i];
        candidate_simplex[i] = best_guess_simplex[i];
    }
    for (int i = 0; i < simplex_size; ++i) {
        for (int j = 0; j < simplex_size; ++j) {
            trial_simplices[i * simplex_size + j] = best_guess_simplex[j];
        }
    }

    while (iterations < max_iterations) {
        iterations += 1;

        for (int trial_idx = 0; trial_idx < simplex_size; ++trial_idx) {
            for (int simplex_idx = 0; simplex_idx < simplex_size; ++simplex_idx) {
                fractions[trial_idx * simplex_size + simplex_idx] = 0;
            }
            pycgpu_hp_simplex_fractions(compositions, num_components,
                                        fixed_chempot_indices, num_fixed_chempots,
                                        &trial_simplices[trial_idx * simplex_size], simplex_size,
                                        fixed_lincomb_molefrac_coefs,
                                        fixed_lincomb_molefrac_rhs, num_lincomb,
                                        free_chempot_indices,
                                        &fractions[trial_idx * simplex_size]);
            double m = 1e300;   // _min
            for (int i = 0; i < simplex_size; ++i) {
                double v = fractions[trial_idx * simplex_size + i];
                if (v < m) m = v;
            }
            smallest_fractions[trial_idx] = m;
        }
        // argmax (strict >, init -1e30): simplex with largest smallest-fraction
        saved_trial = 0;
        {
            double highest = -1e30;
            for (int i = 0; i < simplex_size; ++i) {
                if (smallest_fractions[i] > highest) {
                    highest = smallest_fractions[i];
                    saved_trial = i;
                }
            }
        }
        if (smallest_fractions[saved_trial] < -simplex_size) break;

        for (int i = 0; i < simplex_size; ++i) {
            candidate_simplex[i] = trial_simplices[saved_trial * simplex_size + i];
        }
        for (int i = 0; i < simplex_size; ++i) {
            int idx = candidate_simplex[i];
            for (int ici = 0; ici < simplex_size; ++ici) {
                int chempot_idx = free_chempot_indices[ici];
                f_candidate_tieline[i + simplex_size * ici] =
                    compositions[idx * num_components + chempot_idx];
            }
            candidate_potentials[i] = energies[idx];
            for (int ici = 0; ici < num_fixed_chempots; ++ici) {
                int chempot_idx = fixed_chempot_indices[ici];
                candidate_potentials[i] -= chemical_potentials[chempot_idx]
                    * compositions[idx * num_components + chempot_idx];
            }
        }
        pycgpu_hp_solve(f_candidate_tieline, simplex_size, candidate_potentials);
        if (candidate_potentials[0] == -1e19) break;

        // Streamed driving-force argmin: per-point accumulation order matches
        // the reference's column-wise loops exactly (energies first, then the
        // free-chempot terms in ici order, then the fixed terms in ici
        // order), so every df value — and the strict-< argmin over ascending
        // i — is bit-identical to the stored-array version.
        for (int i = 0; i < simplex_size; ++i) best_guess_simplex[i] = candidate_simplex[i];
        for (int i = 0; i < simplex_size; ++i) {
            for (int j = 0; j < simplex_size; ++j) {
                trial_simplices[i * simplex_size + j] = best_guess_simplex[j];
            }
        }
        double lowest_df = 1e10;
        int min_df = -1;
        for (int i = 0; i < num_points; ++i) {
            double df = energies[i];
            for (int ici = 0; ici < simplex_size; ++ici) {
                df -= candidate_potentials[ici]
                    * compositions[i * num_components + free_chempot_indices[ici]];
            }
            for (int ici = 0; ici < num_fixed_chempots; ++ici) {
                int chempot_idx = fixed_chempot_indices[ici];
                df -= chemical_potentials[chempot_idx]
                    * compositions[i * num_components + chempot_idx];
            }
            if (df < lowest_df) { lowest_df = df; min_df = i; }
        }
        for (int i = 0; i < simplex_size; ++i) {
            trial_simplices[i * simplex_size + i] = min_df;
        }
        if (lowest_df > -1e-8) break;
    }

    out_energy = 0;
    for (int i = 0; i < simplex_size; ++i) {
        int idx = best_guess_simplex[i];
        out_energy += fractions[saved_trial * simplex_size + i] * energies[idx];
    }
    for (int i = 0; i < simplex_size; ++i) {
        result_fractions[i] = fractions[saved_trial * simplex_size + i];
    }
    for (int ici = 0; ici < simplex_size; ++ici) {
        chemical_potentials[free_chempot_indices[ici]] = candidate_potentials[ici];
        result_simplex[ici] = best_guess_simplex[ici];
    }
    for (int i = simplex_size; i < num_components + 1; ++i) {
        result_fractions[i] = 0.0;
        result_simplex[i] = 0;
    }
    return out_energy;
}

#endif // PYCGPU_HYPERPLANE_H
