import numpy as np
cimport numpy as np

from math import factorial

def fact2(i):
    val = 1
    while (i>0):
        val = i*val
        i = i-2
    return val

def binomial(a,b):
    "Binomial coefficient"
    return factorial(a)/factorial(b)/factorial(a-b)

def binomial_prefactor(s,ia,ib,xpa,xpb):
    """
    The integral prefactor containing the binomial coefficients from Augspurger and Dykstra.
    >>> binomial_prefactor(0,0,0,0,0)
    1
    """
    total= 0
    for t in range(s+1):
        if s-ia <= t <= ib:
            total +=  binomial(ia,s-t)*binomial(ib,t)*\
                        pow(xpa,ia-s+t)*pow(xpb,ib-t)
    return total

def overlap1d(alpha1,l1,Ax,alpha2,l2,Bx):
    AB2 = (Ax-Bx)**2
    gamma = alpha1+alpha2
    Px = (alpha1*Ax+alpha2*Bx)/(alpha1+alpha2)
    pre = pow(np.pi/gamma,0.5)*np.exp(-alpha1*alpha2*AB2/gamma)

    wx = 0
    for i in range(1+int(np.floor(0.5*(l1+l2)))):
        wx += binomial_prefactor(2*i,l1,l2,Px-Ax,Px-Bx)*\
				fact2(2*i-1)/pow(2*gamma,i)
    return pre*wx

def overlap1d_matrix(bf):
    len_bf = len(bf)
    output = np.empty((len_bf,len_bf))
    for i in xrange(len_bf):
        (alpha1,l1,Ax) = bf[i]
        for j in xrange(i,len_bf):
            (alpha2,l2,Bx) = bf[j]
            elem = overlap1d(alpha1,l1,Ax,alpha2,l2,Bx)
            output[i,j] = elem
            output[j,i] = elem
    return output