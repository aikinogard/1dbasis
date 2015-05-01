import numpy as np
from basis1d.findbasis import findbasis
from basis1d.tools import sym_xg, sym_uniform_atoms_x

n_lib = np.loadtxt('H2/n.txt')
Nt, Ng = n_lib.shape
atoms_Z = [1, 1]
dx = 0.05
Para = np.loadtxt('H2/Para.txt')

xg_lib = []
atoms_x_lib = []
for i in range(Nt):
    xg_i = sym_xg(dx, len(n_lib[i]))
    xg_lib.append(xg_i)
    atoms_x_lib.append(sym_uniform_atoms_x(xg_i, Para[i], 2))

print 'initialize class...\n'
fb = findbasis(n_lib, xg_lib, atoms_x_lib, atoms_Z, 'universal')
print 'output dens_dict'
fb.output(fobj=open('dens_dict.py', 'w'))
