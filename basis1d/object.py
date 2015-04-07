import basis1d.c1dints as c1dints
import numpy as np
from scipy.linalg import cho_factor,cho_solve

def GTO1d(alpha,l,Ax,xg):
	norm_factor = np.sqrt(pow(2,2*l+0.5)*pow(alpha,l+0.5)/c1dints.fact2(2*l-1)*pow(np.pi,0.5))
	return norm_factor*pow(xg-Ax,l)*np.exp(-alpha*pow(xg-Ax,2))

def make_orbital_bf(atom_angular_basis_info,Ax):
	(l,N,a,b) = atom_angular_basis_info
	if np.isnan(a+b):
		return []
	else:
		return [(a*pow(b,i),l,Ax) for i in xrange(N)]



class findbasis:
	"""
		find basis
		
		example of coord:
		coord = np.array([[1.5,2.6,1.7],[3.5,1.2,1.8],...])
		coord_atom = ['h','h','o']

		example of basis_info:
		basis_info = {'h':[(0,n1,a1,b1),(1,n2,a2,b2)],
						'o':[(0,n3,a3,b3),(1,n4,a4,b4),(2,n5,a5,b5)]}
		use n1 s orbitals and n2 p orbitals for each h atom
		use n3 s orbitals, n4 p orbitals and n5 d orbitals for each o atom
	"""
	def __init__(self,n_ongrid,dx,coord,coord_atom,basis_info):
		self.n_ongrid = n_ongrid
		self.T = np.arange(len(n_ongrid[0]))
		self.dx = dx
		self.xg = np.arange(len(n_ongrid[0]))*dx
		self.coord = coord
		self.coord_atom = coord_atom
		self.basis_info = basis_info

	def make_bf(self,idx):
		"""
			make basis functions for idx data
		"""
		bf = []
		for i,atom in enumerate(self.coord_atom):
			Ax = coord[idx][i]
			for atom_angular_basis_info in basis_info[atom]:
				bf += make_orbital_bf(atom_angular_basis_info,Ax)
		return bf

	def compute_d(self,bf,idx):
		d = np.empty(len(bf))
		for i in len(d):
			(alpha,l,Ax) = bf[i]
			d[i] = self.dx*np.dot(self.n_ongrid,GTO1d(alpha,l,Ax,self.xg))
		return d

	def compute_coeff(self,bf,idx):
		S = c1dints.overlap1d_matrix(bf)
		d = self.compute_d(bf,idx)
		Q = cho_factor(S)
		return cho_solve(Q,d)

	def show_density(self,coeff,bf):
		dens = np.zeros_like(self.xg)
		for i,c in enumerate(coeff):
			(alpha,l,Ax) = bf[i]
			dens += c*GTO1d(alpha,l,Ax,self.xg)
		return dens

	def optimize(self,atom,atom_basis_idx,alist,blist,kfold):
		"""optimize the basis with a and b in alist and blist"""
		if len(basis_info[atom])-1>atom_basis_idx:
			raise RuntimeError('atom_basis_idx to large for atom %s'%atom)

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
					self.basis_info[atom][atom_basis_idx][2] = a
					self.basis_info[atom][atom_basis_idx][3] = b
					err = 0.
					for idx in S:
						bf = self.make_bf(idx)
						coeff = self.compute_coeff(bf,idx)
						dens = self.show_density(coeff,bf)
						err += np.sum((dens-self.n_ongrid[idx])**2)*self.dx
					err = err/len(S)

					if first or err < err_min[i]:
						err_min[i] = err
						a_opt[i] = a
						b_opt[i] = b
						first = False
						
		self.basis_info[atom][atom_basis_idx][2] = np.median(a_opt)
		self.basis_info[atom][atom_basis_idx][3] = np.median(b_opt)




		

