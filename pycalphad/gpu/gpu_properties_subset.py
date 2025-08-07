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
        comp_idx : int  
            Composition index in the composition array
        verbose : bool
            Enable verbose output
        """
        self.properties = properties
        self.condition_idx = condition_idx
        self.temp_idx = temp_idx
        self.comp_idx = comp_idx
        self.verbose = verbose
        
        # Copy scalar attributes
        for attr in ['T', 'P', 'N']:
            if hasattr(properties, attr):
                setattr(self, attr, getattr(properties, attr))
    
    @property
    def MU(self):
        """Extract MU for the specific condition."""
        if not hasattr(self.properties, 'MU'):
            return np.zeros(2)  # Default for 2 components
            
        mu_full = self.properties.MU
        
        # Handle different dimensionalities of MU
        # Common shapes:
        # - (1, 1, n_temps, n_comps, n_components) for multi-condition
        # - (1, 1, 1, 1, n_components) for single condition
        
        if self.verbose:
            print(f"[PropertiesSubset] MU shape: {mu_full.shape}, extracting for temp_idx={self.temp_idx}, comp_idx={self.comp_idx}")
        
        if mu_full.ndim >= 5:
            # Multi-dimensional case
            # Typical indexing: [N, P, T, X, component]
            if mu_full.shape[2] > self.temp_idx and mu_full.shape[3] > self.comp_idx:
                result = mu_full[0, 0, self.temp_idx, self.comp_idx, :]
                if self.verbose:
                    print(f"[PropertiesSubset] Extracted MU: {result}")
                return result
            else:
                if self.verbose:
                    print(f"[PropertiesSubset] WARNING: Index out of bounds, using first condition")
                return mu_full[0, 0, 0, 0, :]
        else:
            # Fallback for unexpected shapes
            if self.verbose:
                print(f"[PropertiesSubset] WARNING: Unexpected MU shape, using flat indexing")
            return mu_full.flatten()[:2]  # Take first 2 components
    
    @property
    def GM(self):
        """Extract GM for the specific condition."""
        if not hasattr(self.properties, 'GM'):
            return 0.0
            
        gm_full = self.properties.GM
        
        if gm_full.ndim >= 4:
            # Multi-dimensional case
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
        
        if np_full.ndim >= 5:
            # Multi-dimensional case
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