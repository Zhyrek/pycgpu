from pycalphad import Database
dbf = Database('Al-Cu-Fe.tdb')
components = ['AL','CU','FE','VA']
print('Components:', components)
print('Number of components:', len(components))
non_va = [c for c in components if c \!= 'VA']
print('Non-VA components:', non_va)
print('Number of non-VA components:', len(non_va))
