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
basis_data = \
					{1: [('s',
							[(2., 0.06462908245839719),
							 (0.8, 0.4066825517248162),
							 (0.32, 0.5468171685012185)])]
					}

print 'initialize class...\n'
fb = findbasis(n_lib,dx,atom_x_lib,atoms_Z,basis_data)
print 'fitting errors on n_lib: \n%s'%fb.error() 
print '==============='
print 'construct basis for atom position [2.,6.] from basis_data'
Smat = fb.make_bfs([2.,6.],True)
print 'overlap matrix: %s'%Smat
print 'print basis: %s\n'%fb.bfs

print '==============='
idx = 60
print 'fit %dth density in lib'
coeff,n_fit = fb.fit_density(idx)
print 'print basis: %s'%fb.bfs
print 'fitting error on %dth density: %f\n'%(idx,fb.error(idx))
fb.show_density([n_fit,fb.n_lib[idx],fb.bfs[0].grid(fb.xg),fb.bfs[1].grid(fb.xg)],True,['fit','grid','basis1','basis2'])

print '==============='
fb.optimize_bf(Z=1,shell='p',alist=np.logspace(-2,2,10),blist=np.linspace(0.1,0.9,10),Nn=2,kfold=10,doadd=True)
print 'new basis: %s'%fb.bfs
print 'new basis data: %s'%fb.basis_data
print 'fitting error on %dth density: %f\n'%(idx,fb.error(idx))