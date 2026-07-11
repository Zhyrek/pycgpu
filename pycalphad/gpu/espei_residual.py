"""Pluggable ESPEI residual: batched ZPF likelihood on the C++ backend.

Drop-in for ``espei.error_functions.zpf_error.ZPFResidual`` — same
constructor and ResidualFunction API, but ``get_likelihood`` /
``get_residuals`` run through ``BatchedZPFCalculator`` (~10x per call on
Cu-Mg-scale systems). Register it in place of the stock ZPFResidual via
ESPEI's pluggable residual registry, e.g.::

    from espei.error_functions.residual_base import residual_function_registry
    from pycalphad.gpu.espei_residual import BatchedZPFResidual
    # replace the stock ZPF residual for this run
    residual_function_registry.discovered_residual_functions = [
        BatchedZPFResidual if r.__name__ == 'ZPFResidual' else r
        for r in residual_function_registry.get_registered_residual_functions()]

``log_prob_ensemble(params_matrix)`` additionally evaluates a whole emcee
proposal batch in one set of launches (``EnsembleSampler(..., vectorize=True)``).
"""
import numpy as np

from espei.error_functions.zpf_error import ZPFResidual
from espei.utils import database_symbols_to_fit


class BatchedZPFResidual(ZPFResidual):
    def __init__(self, database, datasets, phase_models, symbols_to_fit=None,
                 weight=None):
        if symbols_to_fit is None:
            symbols_to_fit = database_symbols_to_fit(database)
        super().__init__(database, datasets, phase_models,
                         symbols_to_fit=symbols_to_fit, weight=weight)
        self._symbols_to_fit = [str(s) for s in symbols_to_fit]
        self._batched = None

    @property
    def batched(self):
        if self._batched is None:
            from pycalphad.gpu.espei_batch import BatchedZPFCalculator
            self._batched = BatchedZPFCalculator(self.zpf_data,
                                                 self._symbols_to_fit)
        return self._batched

    def get_residuals(self, parameters):
        driving_forces, weights = self.batched.driving_forces(
            np.asarray(parameters, dtype=np.float64))
        return (np.concatenate(driving_forces).tolist(),
                np.concatenate(weights).tolist())

    def get_likelihood(self, parameters):
        return self.batched.likelihood(np.asarray(parameters, dtype=np.float64),
                                       data_weight=self.weight)

    def log_prob_ensemble(self, params_matrix):
        """(n_walkers,) likelihood vector for emcee vectorize=True."""
        return self.batched.log_prob_ensemble(params_matrix,
                                              data_weight=self.weight)
