import basis1d.object as obj
import numpy as np

n_ongrid = np.loadtxt('H2/n.txt')
dx = 0.05
Para = np.loadtxt('H2/Para.txt')
mid = dx*(len(n_ongrid[0])-1)/2
coord = np.empty((len(Para),2))
coord[:,0] = mid-Para/2
coord[:,1] = mid+Para/2
coord_atom = ['h','h']
basis_info = {'h':[[0,3,np.NaN,np.NaN],[1,3,np.NaN,np.NaN],[2,1,np.NaN,np.NaN]]}

print 'build findbasis class:'
FB = obj.findbasis(n_ongrid,dx,coord,coord_atom,basis_info)

print 'optimize s orbital of h atom'
FB.optimize('h',0,np.logspace(0,2,10),np.linspace(0.3,1,4),5)
print FB.basis_info
print '\n'

print 'optimize p orbital of h atom'
FB.optimize('h',1,np.logspace(0,2,10),np.linspace(0.3,1,4),5)
print FB.basis_info
print '\n'

print 'optimize d orbital of h atom'
FB.optimize('h',2,np.logspace(0,2,10),np.linspace(0.3,1,4),5)
print FB.basis_info
print '\n'