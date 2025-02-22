from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import pinv
from scipy.linalg import cho_factor, cho_solve
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.major.pad'] = 10
mpl.rcParams['ytick.major.pad'] = 10
from basis1d.tools import S
from basis1d.cgbf import cgbf
from basis1d.basislib import basis

sym2pow = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}


class DensityGrid:

    """
            This is a class including the information for 1d density on grid
    """

    def __init__(self, xg, n, atoms_x, atoms_Z, verbose=False):
        self.xg = np.array(xg)
        self.dx = xg[1] - xg[0]
        self.n = np.array(n)
        self.atoms_x = np.array(atoms_x)
        self.atoms_Z = atoms_Z
        self.verbose = verbose

    def make_bfs(self, basis_data):
        if isinstance(basis_data, str):
            try:
                # search in the basislib
                self.basis_data = basis[basis_data]
                self.basis_name = basis_data
                self.from_basislib = True
            except:
                self.basis_data = {}
                self.basis_name = basis_data
        elif isinstance(basis_data, dict):
            self.basis_data = basis_data
            self.basis_name = 'user-defined'

        self.bfs = []
        for x, Z in zip(self.atoms_x, self.atoms_Z):
            for sym, prims in self.basis_data[Z]:
                exps = [e for e, c in prims]
                coefs = [c for e, c in prims]
                self.bfs.append(cgbf(x, sym2pow[sym], exps, coefs))
            Nbf = len(self.bfs)
            self.Smat = np.empty((Nbf, Nbf))
            for i, j in combinations_with_replacement(range(Nbf), 2):
                self.Smat[i, j] = self.Smat[j, i] = S(self.bfs[i], self.bfs[j])

    def compute_c(self):
        Nbf = len(self.bfs)
        d = np.empty(Nbf)
        for i, bf in enumerate(self.bfs):
            d[i] = self.dx * np.dot(self.n, bf.grid(self.xg))
        try:
            Q = cho_factor(self.Smat + 1e-14 * np.eye(len(self.Smat)))
            self.c = cho_solve(Q, d)
        except:
            if self.verbose:
                print 'pinv is used because there is problem in cho_factor'
            self.c = np.dot(pinv(self.Smat), d)

    def grid(self, xg=None):
        if xg is None:
            xg = self.xg
        return sum(self.c[i] * self.bfs[i].grid(xg)
                   for i in range(len(self.bfs)))

    def get_error(self):
        n_fit = self.grid()
        return np.sum(np.abs(n_fit - self.n)) * self.dx

    def show(self, doshow=False, diff=False):
        if diff:
            plt.plot(self.xg, self.n - self.grid(), 'g',
                     label='density difference')
        else:
            plt.plot(self.xg, self.n, 'k', label='density on grid')
            plt.plot(self.xg, self.grid(), 'r--', label='fit density')
        if doshow:
            plt.show()
