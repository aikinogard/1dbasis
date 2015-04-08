import basis1d.object as obj
import numpy as np
import matplotlib.pyplot as plt

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
FB.optimize('h',0,np.logspace(0,1,10),np.logspace(-4,0,5),5)
print FB.basis_info
FB.show_density(40,doshow=True)
FB.show_error(doshow=True)
print '\n'

print 'optimize p orbital of h atom'
FB.optimize('h',1,np.logspace(0,1,10),np.logspace(-4,0,5),5)
print FB.basis_info
FB.show_density(40,doshow=True)
FB.show_error(doshow=True)
print '\n'

print 'optimize d orbital of h atom'
FB.optimize('h',2,np.logspace(-2,2,10),np.logspace(-8,-3,5),5)
print FB.basis_info
FB.show_density(40,doshow=True)
FB.show_error()
plt.savefig('optimize_d.pdf')
print '\n'

print 'optimized basis_info'
print FB.basis_info
print 'optimized basis functions on 40th density'
print FB.make_bf(40)



