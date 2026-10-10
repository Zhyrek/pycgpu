"""Per-condition scratch memory for the CUDA and C++ solver kernels.

Every condition solved concurrently needs its own slice of each work array
(the SVD/least-squares workspaces, the SystemState, ...). The kernels index
slice ``slot`` of array ``k`` at ``table[k] + slot * stride_k``, where the
strides are the compile-time sizes below; the order of WORK_ARRAYS is the
order of the pointer table (`struct WorkArrays` in the generated source and
`PycgpuWork` in solve_condition.h). The Metal kernel keeps this memory
thread-private and needs none of it.
"""
import numpy as np

__all__ = ['work_array_strides', 'per_condition_bytes', 'allocate_work_arrays']


def work_array_strides(sizes):
    """(name, doubles per slot) for each work array, in pointer-table order."""
    mc = int(sizes['MAX_COMPONENTS'])
    mp = int(sizes['MAX_PHASES'])
    svd = (mc + mp + int(sizes['MAX_STATEVARS'])
           + int(sizes['MAX_FIXED_MOLE_FRACTION_CONDITIONS']) + 2)
    pm = (int(sizes['MAX_DOF_PER_PHASE']) + int(sizes['MAX_INTERNAL_CONSTRAINTS'])
          + int(sizes.get('MAX_PHASE_LOCAL_CONDITIONS', 0)))
    cons = int(sizes['MAX_DOF_PER_PHASE']) + int(sizes['MAX_INTERNAL_CONSTRAINTS'])
    ndof = int(sizes['MAX_STATEVARS']) + int(sizes['MAX_DOF_PER_PHASE'])
    return (
        ('A_lstsq_copy', svd * svd),
        ('U_lstsq', svd * svd),
        ('V_lstsq', svd * svd),
        ('singular_values_lstsq', svd),
        ('superdiag_lstsq', svd),
        ('U_inv', pm * pm),
        ('V_inv', pm * pm),
        ('singular_values_inv', pm),
        ('superdiag_inv', pm),
        ('work_inv', pm * pm),
        ('x_dof', ndof),
        ('grad', ndof),
        ('hess', ndof * ndof),
        ('masses', mc),
        ('mass_jac', mc * ndof),
        ('phase_matrix', cons * cons),
        ('equilibrium_matrix', int(sizes['MAX_EQ_MATRIX_SIZE'])),
        ('equilibrium_rhs', int(sizes['MAX_EQ_MATRIX_ROWS'])),
        ('eq_soln', int(sizes['MAX_EQ_SOLN_LEN'])),
        ('system_states', int(sizes['SYSTEM_STATE_SIZE'])),
        ('delta_ms', mp * mc),
        ('phase_compositions', mp * mc),
        ('phase_amounts_per_mole_atoms', mp * mc),
    )


def per_condition_bytes(sizes):
    return 8 * sum(stride for _, stride in work_array_strides(sizes))


def allocate_work_arrays(xp, sizes, n_slots, address_of, fill=None):
    """Allocate `n_slots` slices of every work array with array module `xp`
    (numpy or cupy). Returns (arrays_by_name, pointer_table) where the table
    is a uint64 array (on the same device) of base addresses in kernel
    order; `address_of(array)` returns an array's kernel-visible address."""
    arrays = {}
    for name, stride in work_array_strides(sizes):
        arrays[name] = xp.empty((n_slots, stride), dtype=np.float64)
        if fill is not None:
            arrays[name].fill(fill)
    table = np.array([address_of(a) for a in arrays.values()], dtype=np.uint64)
    return arrays, xp.asarray(table)
