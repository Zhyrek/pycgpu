from pycalphad import Database, equilibrium, variables as v

# Generate kernel
dbf = Database('NbTi.tdb')
eq = equilibrium(dbf, ['NB', 'TI', 'VA'], ['BCC_A2', 'LIQUID'], 
                 {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}, 
                 calc_opts={'pdens': 10}, gpu=True)
print("Kernel generated")