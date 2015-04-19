from basis1d.c1dints import overlap1d

def S(a,b):
    if b.contracted:
        return sum(cb*S(pb,a) for (cb,pb) in b)
    elif a.contracted:
        return sum(ca*S(b,pa) for (ca,pa) in a)
    return a.norm*b.norm*overlap1d(a.expn,a.power,
                                 a.origin,b.expn,b.power,b.origin)
