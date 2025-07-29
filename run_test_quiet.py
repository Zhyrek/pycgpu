#!/usr/bin/env python
import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable debug output
import subprocess
subprocess.run(['python', 'test_alcu_zeta_fixed.py'])