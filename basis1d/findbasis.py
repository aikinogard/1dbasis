import numpy as np
from basis1d.pgbf import pgbf
from basis1d.cgbf import cgbf
from basis1d.tools import S
from basis1d.basislib import basis
from scipy.linalg import cho_factor,cho_solve,pinv
import matplotlib.pyplot as plt

sym2pow = {'s':0,'p':1,'d':2,'f':3,'g':4}

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
		or
		basis_data = basis['minix']
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
		self.name = self.stoich()

	def stoich(self):
		from collections import Counter
		from basis1d.tools import symbol	
		cnt = Counter()
		for Z in self.atoms_Z:
			cnt[Z] += 1
		keys = sorted(cnt.keys())
		s = []
		for key in keys:
			if cnt[key] == 1:
				s.append(symbol[key])
			else:
				s.append("%s%d" % (symbol[key],cnt[key]))
		return "".join(s)

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

	def compute_nc_lib(self,T=None):
		"return the coefficients of each density in n_lib on current basis_data"
		if T==None:
			T = self.T
		self.make_bfs(self.atom_x_lib[0])
		nc_lib = np.empty((len(T),len(self.bfs)))
		for i,idx in enumerate(T):
			coeff,n_fit = self.fit_density(idx)
			nc_lib[i] = coeff
		return nc_lib








