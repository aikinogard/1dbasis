import numpy as np
from scipy.linalg import cho_factor,cho_solve,pinv
import matplotlib.pyplot as plt
import array
from basis1d.c1dints import overlap1d,fact2

sym2pow = {'s':0,'p':1,'d':2,'f':3,'g':4}

def S(a,b):
    if b.contracted:
        return sum(cb*S(pb,a) for (cb,pb) in b)
    elif a.contracted:
        return sum(ca*S(b,pa) for (ca,pa) in a)
    return a.norm*b.norm*overlap1d(a.expn,a.power,
                                 a.origin,b.expn,b.power,b.origin)

class pgbf:
	contracted = False
	def __init__(self,expn,origin=0,power=0):
		self.expn = float(expn)
		self.origin = float(origin)
		self.power = power
		self._normalize()

	def __repr__(self):
		return "pgbf(%f,%f,%d)"%(self.expn,self.origin,self.power)

	def __call__(self,x):
		"Compute the amplitude of the PGBF at point x"
		return self.grid(x)

	def grid(self,xs):
		dx = xs-self.origin
		return self.norm*dx**self.power*np.exp(-self.expn*dx**2)

	def _normalize(self):
		self.norm = np.sqrt(pow(2,2*self.power+0.5)*
                            pow(self.expn,self.power+0.5)/
                            fact2(2*self.power-1)/np.sqrt(np.pi))

class cgbf:
	contracted = True
	def __init__(self,origin=0,power=0,exps=[],coefs=[]):
		self.origin = float(origin)
		self.power = float(power)

		# cgbf is made by list of pgbf
		self.pgbfs = []
		# the coefficient of each pgbf
		self.coefs = array.array('d')
		# normalization constant of pgbf
		self.pnorms = array.array('d')
		# exponential of each pgbf
		self.pexps = array.array('d')

		for expn,coef in zip(exps,coefs):
			self.add_pgbf(expn,coef,False)

		if self.pgbfs:
			self.normalize()

	def __getitem__(self,item):
		return list(zip(self.coefs,self.pgbfs)).__getitem__(item)

	def __repr__(self):
		return "cgbf(%f,%d,%s,%s)"%(self.origin,self.power,list(self.pexps),list(self.coefs))

	def __call__(self,*args,**kwargs):
		return sum(c*p(*args,**kwargs) for c,p in self)

	def grid(self,xs):
		return sum(c*p.grid(xs) for c,p in self)

	def add_pgbf(self,expn,coef,renormalize=True):
		self.pgbfs.append(pgbf(expn,self.origin,self.power))
		self.coefs.append(coef)

		if renormalize:
			self.normalize()

		p = self.pgbfs[-1]
		self.pnorms.append(p.norm)
		self.pexps.append(p.expn)

	def normalize(self):
		Saa_sqrt = np.sqrt(S(self,self))
		for i in range(len(self.coefs)):
			self.coefs[i] /= Saa_sqrt

class findbasis:
	"""
		optimize uncontracted basis using even-tempered method 
		and convert densities on grid
		to coefficients on 1d gaussian type basis.

		n_lib: np.array, each row is a density on grid
		dx: float, grid spacing
		atom_x_lib: np.array, each row is a list of atom coordinates
		atoms_Z: np.array, list of atomic index of atoms in one chain
		basis_data = \
					{1: [('s',
							[(2., 0.06462908245839719),
							 (0.8, 0.4066825517248162),
							 (0.32, 0.5468171685012185)])]
					}
	"""
	def __init__(self,n_lib,dx,atom_x_lib,atoms_Z,basis_data):
		self.n_lib = n_lib
		self.Nt,self.Ng = n_lib.shape
		self.T = np.arange(self.Nt)
		self.dx = dx
		self.xg = np.arange(self.Ng)*dx
		self.atom_x_lib = atom_x_lib
		self.atoms_Z = atoms_Z
		self.basis_data = basis_data

	def make_bfs(self,atom_x,eval_S=False):
		self.bfs = []
		for x,Z in zip(atom_x,self.atoms_Z):
			for sym,prims in self.basis_data[Z]:
				exps = [e for e,c in prims]
				coefs = [c for e,c in prims]
				self.bfs.append(cgbf(x,sym2pow[sym],exps,coefs))
		if eval_S:
			Nbf = len(self.bfs)
			Smat = np.empty((Nbf,Nbf))
			for i in range(Nbf):
				for j in range(i,Nbf):
					out = S(self.bfs[i],self.bfs[j])
					Smat[i,j] = out
					Smat[j,i] = out
			return Smat	

	def fit_density(self,idx):
		Smat = self.make_bfs(self.atom_x_lib[idx],True)
		Nbf = len(self.bfs)
		d = np.empty(Nbf)
		for i,bf in enumerate(self.bfs):
			d[i] = self.dx*np.dot(self.n_lib[idx],bf.grid(self.xg))
		coeff = np.dot(pinv(Smat),d)

		n_fit = sum(coeff[i]*self.bfs[i].grid(self.xg) for i in range(Nbf))

		return coeff,n_fit

	def error(self,idx=None,n_fit=None):
		if idx!=None:
			if n_fit==None:
				coeff,n_fit = self.fit_density(idx)
			return np.sum((n_fit - self.n_lib[idx])**2)*self.dx
		else:
			err = np.empty(self.Nt)
			for idx in self.T:
				coeff,n_fit = self.fit_density(idx)
				err[idx] = np.sum((n_fit - self.n_lib[idx])**2)*self.dx
			return err


	def show_density(self,ns,doshow=False,names=None):
		if names:
			for i,n in enumerate(ns):
				plt.plot(self.xg,n,label=names[i])
			plt.legend(loc=1)
		else:
			for n in ns:
				plt.plot(self.xg,n)

		plt.xlabel(r'$x$')
		plt.ylabel(r'$n(x)$')
		if doshow: plt.show()

	def optimize_bf(self,Z,shell,alist,blist,Nn,kfold,doadd=True):
		if sum([1 for b in blist if b>=1])>0:
			raise ValueError('b should be smaller than 1')
		print 'optimize basis function with even-tempered method with %d-fold cross validation'%kfold
		print 'add to basis data: %s'%doadd
		print 'atomic index: %d'%Z
		print 'shell: %s'%shell
		print 'list of a: %s'%alist
		print 'list of b: %s'%blist
		print 'number of new basis: %d\n'%Nn

		T = self.T
		p = T.copy()
		np.random.shuffle(p)
		p = np.reshape(p,(kfold,-1))

		err_min = np.empty(kfold)
		a_opt = np.empty(kfold)
		b_opt = np.empty(kfold)

		for i,S in enumerate(p):
			first = True
			err_min[i] = np.NaN
			for a in alist:
				for b in blist:
					self.basis_data[Z].extend([(shell,[(a*b**m,1.)]) for m in range(Nn)])
					err = 0.
					for idx in S:
						err += self.error(idx)
					err = err/len(S)

					if first or err < err_min[i]:
						err_min[i] = err
						a_opt[i] = a
						b_opt[i] = b
						first = False
					del self.basis_data[Z][-Nn:]
			print '\tvalidation set: %s'%S
			print '\toptimized a=%f\tb=%f'%(a_opt[i],b_opt[i])
			print '\texpn:%s\n'%[a_opt[i]*b_opt[i]**m for m in range(Nn)]
		shell_data = [(shell,[(np.median(a_opt)*np.median(b_opt)**m,1.)]) for m in range(Nn)]
		print 'from cross validation:'
		print 'optimized a=%f\tb=%f'%(np.median(a_opt),np.median(b_opt))

		if np.median(a_opt)==min(alist):
			print '\toptimized a is the smallest value in alist'
		elif np.median(a_opt)==max(alist):
			print '\toptimized a is the largest value in alist'
		if np.median(b_opt)==min(blist):
			print '\toptimized b is the smallest value in blist'
		elif np.median(b_opt)==max(blist):
			print '\toptimized b is the largest value in blist'

		print 'optimized new shell: %s'%shell_data
		if doadd:
			self.basis_data[Z].extend(shell_data)
		return shell_data

	def save_basis_data(self):
		pass





