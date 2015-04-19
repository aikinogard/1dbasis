import numpy as np
import array
from basis1d.pgbf import pgbf
from basis1d.int_tools import S

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