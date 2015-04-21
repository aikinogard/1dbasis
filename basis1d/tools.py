from basis1d.c1dints import overlap1d

def S(a,b):
    if b.contracted:
        return sum(cb*S(pb,a) for (cb,pb) in b)
    elif a.contracted:
        return sum(ca*S(b,pa) for (ca,pa) in a)
    return a.norm*b.norm*overlap1d(a.expn,a.power,
                                 a.origin,b.expn,b.power,b.origin)

symbol = [
    "X","H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",  "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl","Pb","Bi","Po","At","Rn"]

def dict2str(dic,djoint=True):
    from types import StringType, DictType
    from numpy import ndarray
    lines = ['{']
    for key in dic.keys():
        if type(key)==StringType:
            lines.append("\t'%s':"%key)
        else:
            lines.append("\t%s:"%key)

        if type(dic[key])==StringType:
            lines.append("\t\t'%s',"%dic[key])
        elif type(dic[key]) == ndarray:
            lines.append("\t\tnp.array(%s),"%dic[key].tolist())
        elif type(dic[key])==DictType:
            subdiclines = dict2str(dic[key],djoint=False)
            for subdicline in subdiclines:
                lines.append('\t\t'+subdicline)
            lines[-1] = lines[-1]+','
        else:
            lines.append("\t\t%s,"%dic[key])
    lines.append('}')
    if djoint:
        return '\n'.join(lines)
    else:
        return lines








