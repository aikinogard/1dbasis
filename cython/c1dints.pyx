from __future__ import division
import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "math.h":  
    double pow(double x,double y)
    double sqrt(double x)
    double floor(double x)
    double exp(double exp)

cdef double PI = 3.14159265358979

cdef int _factorial(int i):
    cdef int val = 1
    while (i>0):
        val = i*val
        i = i-1
    return val

cdef int _fact2(int i):
    cdef int val = 1
    while (i>0):
        val = i*val
        i = i-2
    return val

def fact2(int i):
    cdef double val = _fact2(i)
    return val

@cython.cdivision(True)
cdef int _binomial(int a,int b):
    """Binomial coefficient"""
    cdef int val = 1
    cdef int i
    if b > a-b:
        b = a-b
    for i from 0 <= i < b:
        val *= (a-i)
        val /= (i+1)
    return val

cdef double _binomial_prefactor(int s,int ia,int ib,double xpa,double xpb):
    """
    The integral prefactor containing the binomial coefficients from Augspurger and Dykstra.
    >>> binomial_prefactor(0,0,0,0,0)
    1
    """
    cdef double total = 0
    cdef int t
    for t from 0 <= t < s+1:
        if s-ia <= t <= ib:
            total +=  _binomial(ia,s-t)*_binomial(ib,t)*\
                        pow(xpa,ia-s+t)*pow(xpb,ib-t)
    return total

@cython.cdivision(True)
cdef double _overlap1d(double alpha1,int l1,double Ax,double alpha2,int l2,double Bx):
    cdef int i
    cdef double AB2 = (Ax-Bx)**2
    cdef double gamma = alpha1+alpha2
    cdef double Px = (alpha1*Ax+alpha2*Bx)/(alpha1+alpha2)
    cdef double pre = pow(PI/gamma,0.5)*exp(-alpha1*alpha2*AB2/gamma)
    cdef double wx = 0
    cdef double val

    for i from 0 <= i < 1+<int>(floor(0.5*(l1+l2))):
        wx += _binomial_prefactor(2*i,l1,l2,Px-Ax,Px-Bx)*\
				_fact2(2*i-1)/pow(2*gamma,i)
    val = pre*wx*_norm_fact(alpha1,l1,Ax)*_norm_fact(alpha2,l2,Bx)
    return val

def overlap1d(double alpha1,int l1,double Ax,double alpha2,int l2,double Bx):
    cdef double val
    val = _overlap1d(alpha1,l1,Ax,alpha2,l2,Bx)
    return val

@cython.cdivision(True)
cdef double _norm_fact(double alpha,int l,double Ax):
    cdef double val
    val = sqrt(pow(2,2*l+0.5)*pow(alpha,l+0.5)/(_fact2(2*l-1)*pow(PI,0.5)))
    return val

@cython.boundscheck(False) # turn of bounds-checking for entire function
def overlap1d_matrix(list bf):
    cdef int i
    cdef int j
    cdef int len_bf = len(bf)
    cdef double elem
    cdef double alpha1
    cdef double alpha2
    cdef int l1
    cdef int l2
    cdef double Ax
    cdef double Bx

    cdef np.ndarray[DTYPE_t, ndim=2] output = np.empty((len_bf,len_bf))
    for i from 0 <= i < len_bf:
        alpha1,l1,Ax = bf[i]
        for j from i <= j < len_bf:
            alpha2,l2,Bx = bf[j]
            elem = _overlap1d(alpha1,l1,Ax,alpha2,l2,Bx)
            output[i,j] = elem
            output[j,i] = elem
    return output