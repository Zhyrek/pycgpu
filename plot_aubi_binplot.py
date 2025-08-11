#!/usr/bin/env python
"""Plot Au-Bi phase diagram using pycalphad's binplot."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pycalphad import Database, binplot, variables as v
import numpy as np

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = list(dbf.phases.keys())

print(f"Plotting Au-Bi phase diagram with phases: {phases}")

# Create binplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Define conditions for binplot
conditions = {
    v.X('BI'): (0, 1, 0.01),  # 0 to 1 in 0.01 increments
    v.T: (300, 1800, 20),      # 300 to 1800 K in 20 K increments
    v.P: 101325
}

# Create the binplot
binplot(dbf, comps, phases, conditions, plot_kwargs={'ax': ax})

# Customize plot
ax.set_xlabel('Mole Fraction Bi', fontsize=12)
ax.set_ylabel('Temperature (K)', fontsize=12)
ax.set_title('Au-Bi Phase Diagram (pycalphad binplot)', fontsize=14)
ax.grid(True, alpha=0.3)

# Save figure
plt.tight_layout()
plt.savefig('aubi_phase_diagram_binplot.png', dpi=150)
print("Saved phase diagram to aubi_phase_diagram_binplot.png")
plt.close()