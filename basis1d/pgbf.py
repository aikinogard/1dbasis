import numpy as np
from basis1d.c1dints import fact2

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
