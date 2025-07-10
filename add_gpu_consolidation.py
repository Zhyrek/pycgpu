#!/usr/bin/env python3
"""Add phase consolidation to GPU code"""

# Find the right place to add consolidation
consolidation_code = '''
            // PHASE CONSOLIDATION - Match CPU behavior
            // Consolidate phases with same phase record and similar compositions
            const double COMPSET_CONSOLIDATE_DISTANCE = 1e-4;
            
            for (int idx = 0; idx < safe_num_phases; ++idx) {
                if (phase_amounts[idx] < 1e-12) continue; // Skip removed phases
                
                for (int idx2 = idx + 1; idx2 < safe_num_phases; ++idx2) {
                    if (phase_amounts[idx2] < 1e-12) continue; // Skip removed phases
                    
                    // Check if same phase type
                    if (phase_indices[idx] != phase_indices[idx2]) continue;
                    
                    // Check composition difference
                    bool should_consolidate = true;
                    double max_comp_diff = 0.0;
                    
                    // Calculate composition difference based on site fractions
                    for (int sf_idx = 0; sf_idx < MAX_DOF_PER_PHASE; ++sf_idx) {
                        double sf1 = site_fractions[idx * MAX_DOF_PER_PHASE + sf_idx];
                        double sf2 = site_fractions[idx2 * MAX_DOF_PER_PHASE + sf_idx];
                        double diff = fabs(sf1 - sf2);
                        if (diff > max_comp_diff) max_comp_diff = diff;
                        if (diff > COMPSET_CONSOLIDATE_DISTANCE) {
                            should_consolidate = false;
                            break;
                        }
                    }
                    
                    if (should_consolidate) {
                        if (tid == 0) {
                            printf("GPU: CONSOLIDATING phases %d and %d (same type, max_diff=%.6f)\\n", 
                                   idx, idx2, max_comp_diff);
                        }
                        
                        // Weighted average of site fractions
                        double total_amt = phase_amounts[idx] + phase_amounts[idx2];
                        for (int sf_idx = 0; sf_idx < MAX_DOF_PER_PHASE; ++sf_idx) {
                            double sf1 = site_fractions[idx * MAX_DOF_PER_PHASE + sf_idx];
                            double sf2 = site_fractions[idx2 * MAX_DOF_PER_PHASE + sf_idx];
                            site_fractions[idx * MAX_DOF_PER_PHASE + sf_idx] = 
                                (sf1 * phase_amounts[idx] + sf2 * phase_amounts[idx2]) / total_amt;
                        }
                        
                        // Combine amounts
                        phase_amounts[idx] += phase_amounts[idx2];
                        phase_amounts[idx2] = 0.0; // Remove phase 2
                        
                        // Update number of phases
                        safe_num_phases--;
                        
                        // Shift remaining phases down
                        for (int k = idx2; k < safe_num_phases; ++k) {
                            phase_indices[k] = phase_indices[k+1];
                            phase_amounts[k] = phase_amounts[k+1];
                            for (int sf_idx = 0; sf_idx < MAX_DOF_PER_PHASE; ++sf_idx) {
                                site_fractions[k * MAX_DOF_PER_PHASE + sf_idx] = 
                                    site_fractions[(k+1) * MAX_DOF_PER_PHASE + sf_idx];
                            }
                        }
                        
                        // Decrement idx2 to check the new phase at this position
                        idx2--;
                    }
                }
            }
            
            // Update initial_phase_data_single with consolidated phases
            initial_phase_data_single.num_phases = safe_num_phases;
'''

print("Phase consolidation code to add:")
print(consolidation_code)
print("\nThis should be added right before:")
print("// Set up condition args - copy actual state variables from Python")