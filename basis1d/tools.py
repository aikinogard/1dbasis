import numpy as np
from basis1d.c1dints import overlap1d


def S(a, b):
    if b.contracted:
        return sum(cb * S(pb, a) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * S(b, pa) for (ca, pa) in a)
    return a.norm * b.norm * overlap1d(a.expn, a.power,
                                       a.origin, b.expn, b.power, b.origin)

symbol = [
    "X", "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn"]


def dict2str(dic, djoint=True):
    from types import StringType, DictType
    from numpy import ndarray
    lines = ['{']
    for key in dic.keys():
        if type(key) == StringType:
            lines.append("\t'%s':" % key)
        else:
            lines.append("\t%s:" % key)

        if type(dic[key]) == StringType:
            lines.append("\t\t'%s'," % dic[key])
        elif type(dic[key]) == ndarray:
            lines.append("\t\tnp.array(%s)," % dic[key].tolist())
        elif type(dic[key]) == DictType:
            subdiclines = dict2str(dic[key], djoint=False)
            for subdicline in subdiclines:
                lines.append('\t\t' + subdicline)
            lines[-1] = lines[-1] + ','
        else:
            lines.append("\t\t%s," % dic[key])
    lines.append('}')
    if djoint:
        return '\n'.join(lines)
    else:
        return lines


def sym_xg(dx, Ng):
    """
        return grid symmetric to the origin

        >>> sym_xg(0.1, 5)
        array([-0.2, -0.1,  0. ,  0.1,  0.2])
        >>> sym_xg(0.1, 6)
        array([-0.25, -0.15, -0.05,  0.05,  0.15,  0.25])
    """
    if Ng % 2:
        # if Ng is odd, put the middle point at origin
        d = (Ng - 1) / 2
        return np.arange(-d, d + 1) * dx
    else:
        # if Ng is even, origin not at grid point
        d = Ng / 2
        return np.arange(-d + 0.5, d + 0.5) * dx


def sym_uniform_atoms_x(xg, b, N):
    """
        return atoms' coordinate on grid symmetric to the origin
        this is for a uniform chain
        b is the atomic separation
        N is the number of atoms in the chain

        case 1

        >>> xg = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        >>> sym_uniform_atoms_x(xg, 0.2, 3)
        array([-0.2,  0. ,  0.2])

        case 3

        >>> sym_uniform_atoms_x(xg, 0.2, 4)
        array([-0.3, -0.1,  0.1,  0.3])

        case 4

        >>> xg = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        >>> sym_uniform_atoms_x(xg, 0.6, 4)
        array([-0.9, -0.3,  0.3,  0.9])

    """
    dx = xg[1] - xg[0]
    Ng = len(xg)
    # b on dx unit
    bg = round(b / dx)
    if np.isclose(bg, b / dx) is not True:
        raise RuntimeError("b cannot be fit into the grid")
    if N % 2:
        # if the number of atoms on uniform chain is odd:
        d = (N - 1) / 2
        if Ng % 2:
            # case 1
            return np.arange(-d, d + 1) * b
        else:
            # case 2
            raise RuntimeError("Impossible to put odd N on even Ng")
    else:
        # if the number of atoms on uniform chain is even:
        d = N / 2
        if Ng % 2:
            # case 3
            return np.arange(-d + 0.5, d + 0.5) * b
        else:
            # case 4
            assert round(b / dx) % 2
            return np.arange(-d + 0.5, d + 0.5) * b


if __name__ == '__main__':
    import doctest
    doctest.testmod()
