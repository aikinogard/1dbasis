import numpy as np
from basis1d.basislib import basis
from basis1d.grid import DensityGrid


class findbasis:

    """
        optimize uncontracted basis using even-tempered method
        and convert densities on grid
        to coefficients on 1d gaussian type basis.

        n_lib: list, each row is a density on grid, the length of
                each row can be different

        xg_lib: list, each row is a grid, the length of
                each row can be different

        atoms_x_lib: list, each row is a list of atom coordinates,
                the length of each row can be different

        atoms_Z: list, list of atomic index of atoms in one chain

        basis_data = \
                    {1: [('s',
                            [(2., 0.06462908245839719),
                             (0.8, 0.4066825517248162),
                             (0.32, 0.5468171685012185)])]
                    }
        or
        basis_data = basis['minix']
    """

    def __init__(self, n_lib, xg_lib, atoms_x_lib, atoms_Z, basis_data):
        assert len(n_lib) == len(xg_lib) == len(atoms_x_lib)

        self.Nt = len(n_lib)
        self.densities = []
        for i in range(self.Nt):
            self.densities.append(
                DensityGrid(xg_lib[i], n_lib[i], atoms_x_lib[i], atoms_Z))
        self.atoms_x_lib = np.array(atoms_x_lib)
        self.atoms_Z = atoms_Z

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

    def __getitem__(self, i):
        return self.densities.__getitem__(i)

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
                s.append("%s%d" % (symbol[key], cnt[key]))
        return "".join(s)

    def fit_density(self):
        for density in self.densities:
            density.make_bfs(self.basis_data)
            density.compute_c()

    def get_error(self, T=None):
        if T is None:
            T = range(self.Nt)
        err_lib = np.empty(len(T))
        for i, idx in enumerate(T):
            err_lib[i] = self.densities[idx].get_error()
        return err_lib

    def get_nc_lib(self, T=None):
        if T is None:
            T = range(self.Nt)
        nc_lib = []
        for i, idx in enumerate(T):
            self.densities[idx].make_bfs(self.basis_data)
            self.densities[idx].compute_c()
            nc_lib.append(self.densities[idx].c)
        return np.array(nc_lib)

    def optimize_bf(self, Z, shell, alist, blist, Nn, kfold, doadd=True):
        if sum([1 for b in blist if b >= 1]) > 0:
            raise ValueError('b should be smaller than 1')
        print 'optimize basis function with even-tempered method \
            with %d-fold cross validation' % kfold
        print 'add to basis data: %s' % doadd
        print 'atomic index: %d' % Z
        print 'shell: %s' % shell
        print 'list of a: %s' % alist
        print 'list of b: %s' % blist
        print 'number of new basis: %d\n' % Nn

        p = np.arange(self.Nt)
        np.random.shuffle(p)
        p = np.reshape(p, (kfold, -1))

        err_min = np.empty(kfold)
        a_opt = np.empty(kfold)
        b_opt = np.empty(kfold)

        err_min = np.inf

        for i, S in enumerate(p):
            for a in alist:
                for b in blist:
                    self.basis_data[Z].extend(
                        [(shell, [(a * b ** m, 1.)]) for m in range(Nn)])
                    err = 0.
                    for idx in S:
                        self.densities[idx].make_bfs()
                        self.densities[idx].compute_c()
                        err += self.densities[idx].get_error()
                    err = err / len(S)

                    if err < err_min[i]:
                        err_min[i] = err
                        a_opt[i] = a
                        b_opt[i] = b
                    del self.basis_data[Z][-Nn:]
            print '\tvalidation set: %s' % S
            print '\toptimized a=%f\tb=%f' % (a_opt[i], b_opt[i])
            print '\texpn:%s\n' % [a_opt[i] * b_opt[i] ** m for m in range(Nn)]
        shell_data = [(shell, [(np.median(a_opt) * np.median(b_opt) ** m, 1.)])
                      for m in range(Nn)]
        print 'from cross validation:'
        print 'optimized a=%f\tb=%f' % (np.median(a_opt), np.median(b_opt))

        if np.median(a_opt) == min(alist):
            print '\toptimized a is the smallest value in alist'
        elif np.median(a_opt) == max(alist):
            print '\toptimized a is the largest value in alist'
        if np.median(b_opt) == min(blist):
            print '\toptimized b is the smallest value in blist'
        elif np.median(b_opt) == max(blist):
            print '\toptimized b is the largest value in blist'

        print 'optimized new shell: %s' % shell_data
        if doadd:
            self.basis_data[Z].extend(shell_data)
            if self.from_basislib:
                self.from_basislib = False
                self.basis_name = 'user-defined'
        return shell_data

    def output(self, T=None, name=None, fobj=None):
        "output density information to a .py file"
        dens_dict = {}

        if T is None:
            T = range(self.Nt)

        if name is None:
            name = '%sNt%d' % (self.stoich(), self.Nt)

        dens_dict['name'] = name

        dens_dict['atoms_Z'] = self.atoms_Z

        dens_dict['basis_name'] = self.basis_name
        dens_dict['basis_data'] = self.basis_data

        dens_dict['nc_lib'] = self.get_nc_lib(T)
        dens_dict['atoms_x_lib'] = self.atoms_x_lib[T]

        if fobj:
            from basis1d.tools import dict2str
            record = ['import numpy as np', '%s=\\' %
                      name, dict2str(dens_dict)]
            fobj.write('\n'.join(record))
        else:
            return dens_dict
