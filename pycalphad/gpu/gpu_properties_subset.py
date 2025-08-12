"""Extract a subset of properties for a specific condition."""

import numpy as np


class PropertiesSubset:
    """Wrapper that extracts properties for a specific condition from multi-condition results."""
    
    def __init__(self, properties, condition_idx, temp_idx, comp_idx, verbose=False):
        """
        Initialize with multi-condition properties and indices for the specific condition.
        
        Parameters:
        -----------
        properties : object
            The full multi-condition properties object from starting_point()
        condition_idx : int
            The flat condition index
        temp_idx : int
            Temperature index in the temperature array
        comp_idx : int or dict
            Composition index in the composition array (int for binary, dict for ternary+)
        verbose : bool
            Enable verbose output
        """
        self.properties = properties
        self.condition_idx = condition_idx
        self.temp_idx = temp_idx
        
        # Handle both single index (binary) and dict of indices (ternary+)
        if isinstance(comp_idx, dict):
            self.comp_indices = comp_idx  # Dictionary mapping component names to indices
            self.comp_idx = list(comp_idx.values())[0] if comp_idx else 0  # For backward compat
            self.is_ternary = True
        else:
            self.comp_idx = comp_idx
            self.comp_indices = None
            self.is_ternary = False
            
        self.verbose = verbose
        
        # Copy scalar attributes
        for attr in ['T', 'P', 'N']:
            if hasattr(properties, attr):
                setattr(self, attr, getattr(properties, attr))
    
    @property
    def MU(self):
        """Extract MU for the specific condition."""
        if not hasattr(self.properties, 'MU'):
            # Return empty array - let caller determine size
            return np.array([])
            
        mu_full = self.properties.MU
        
        # Handle different dimensionalities of MU
        # Common shapes:
        # - (1, 1, n_temps, n_comps, n_components) for multi-condition
        # - (1, 1, 1, 1, n_components) for single condition
        
        if self.verbose:
            print(f"[PropertiesSubset] MU shape: {mu_full.shape}, extracting for temp_idx={self.temp_idx}, comp_idx={self.comp_idx}")
        
        if mu_full.ndim == 6:
            # Ternary system: [N, P, T, X_comp1, X_comp2, component]
            if self.is_ternary:
                comp_idx_list = list(self.comp_indices.values())
                if len(comp_idx_list) == 2:
                    x_cu_idx = comp_idx_list[0]
                    x_fe_idx = comp_idx_list[1]
                    result = mu_full[0, 0, self.temp_idx, x_cu_idx, x_fe_idx, :]
                    if self.verbose:
                        print(f"[PropertiesSubset] Extracted MU from 6D: {result}")
                    return result
            # Fallback for unexpected 6D case
            return mu_full[0, 0, 0, 0, 0, :]
        elif mu_full.ndim >= 5:
            # Binary system or simpler multi-dimensional case
            # Typical indexing: [N, P, T, X, component]
            if mu_full.shape[2] > self.temp_idx and mu_full.shape[3] > self.comp_idx:
                result = mu_full[0, 0, self.temp_idx, self.comp_idx, :]
                if self.verbose:
                    print(f"[PropertiesSubset] Extracted MU from 5D: {result}")
                return result
            else:
                if self.verbose:
                    print(f"[PropertiesSubset] WARNING: Index out of bounds, using first condition")
                return mu_full[0, 0, 0, 0, :]
        else:
            # Fallback for unexpected shapes
            if self.verbose:
                print(f"[PropertiesSubset] WARNING: Unexpected MU shape {mu_full.shape}, using flat indexing")
            # For unexpected shapes, return ALL available data - no hardcoded limits
            return mu_full.flatten()
    
    @property
    def GM(self):
        """Extract GM for the specific condition."""
        if not hasattr(self.properties, 'GM'):
            return 0.0
            
        gm_full = self.properties.GM
        
        if gm_full.ndim == 5:
            # Ternary system: [N, P, T, X_comp1, X_comp2]
            if self.is_ternary:
                comp_idx_list = list(self.comp_indices.values())
                if len(comp_idx_list) == 2:
                    x_cu_idx = comp_idx_list[0]
                    x_fe_idx = comp_idx_list[1]
                    return float(gm_full[0, 0, self.temp_idx, x_cu_idx, x_fe_idx])
            # Fallback for unexpected 5D case
            return float(gm_full[0, 0, 0, 0, 0])
        elif gm_full.ndim >= 4:
            # Binary system: [N, P, T, X]
            if gm_full.shape[2] > self.temp_idx and gm_full.shape[3] > self.comp_idx:
                return float(gm_full[0, 0, self.temp_idx, self.comp_idx])
            else:
                return float(gm_full[0, 0, 0, 0])
        else:
            # Single value or unexpected shape
            return float(gm_full.flatten()[0])
    
    @property
    def NP(self):
        """Extract NP for the specific condition."""
        if not hasattr(self.properties, 'NP'):
            return np.zeros(3)  # Default for 3 phases
            
        np_full = self.properties.NP
        
        if np_full.ndim == 6:
            # Ternary system: [N, P, T, X_comp1, X_comp2, phase]
            if self.is_ternary:
                comp_idx_list = list(self.comp_indices.values())
                if len(comp_idx_list) == 2:
                    x_cu_idx = comp_idx_list[0]
                    x_fe_idx = comp_idx_list[1]
                    return np_full[0, 0, self.temp_idx, x_cu_idx, x_fe_idx, :]
            # Fallback for unexpected 6D case
            return np_full[0, 0, 0, 0, 0, :]
        elif np_full.ndim >= 5:
            # Binary system: [N, P, T, X, phase]
            if np_full.shape[2] > self.temp_idx and np_full.shape[3] > self.comp_idx:
                return np_full[0, 0, self.temp_idx, self.comp_idx, :]
            else:
                return np_full[0, 0, 0, 0, :]
        else:
            # Fallback
            return np_full.flatten()[:3]
    
    @property
    def Phase(self):
        """Extract Phase for the specific condition."""
        if not hasattr(self.properties, 'Phase'):
            return np.array([''] * 3)  # Default empty phases
            
        phase_full = self.properties.Phase
        
        if phase_full.ndim >= 5:
            # Multi-dimensional case
            if phase_full.shape[2] > self.temp_idx and phase_full.shape[3] > self.comp_idx:
                return phase_full[0, 0, self.temp_idx, self.comp_idx, :]
            else:
                return phase_full[0, 0, 0, 0, :]
        else:
            # Fallback
            return phase_full.flatten()[:3]