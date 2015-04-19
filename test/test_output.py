from basis1d.findbasis import *

n_lib = np.loadtxt('H2/n.txt')
Nt,Ng = n_lib.shape
dx = 0.05
Para = np.loadtxt('H2/Para.txt')
mid = dx*(Ng-1)/2
atom_x_lib = np.empty((Nt,2))
atom_x_lib[:,0] = mid-Para/2
atom_x_lib[:,1] = mid+Para/2
atoms_Z = [1,1]

print 'initialize class...\n'
fb = findbasis(n_lib,dx,atom_x_lib,atoms_Z,'universal')
print 'output dens_dict'
fb.output(fobj=open('dens_dict.py','w'))