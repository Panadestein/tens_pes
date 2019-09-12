"""
The present module performs the optimization of the coefficients of the SOP-FBR
representation of a 2D PES of the shape (V(x, y) = x^2 + y^2 + lambda*xy)
"""
import os
import time
import itertools
import numpy as np
from numpy.polynomial import chebyshev as cheby
import nlopt

# System paramters

NDOF = 2  # f
CHEBDIM = 4  # t_k (for the moment the same for all k)
MARRAY = np.array([5, 5])  # m_k
LMB = 1  # lambda

# Reference 2D potential


def v2d(x_dim, y_dim):
    """Returns the value of the 2D potential at the point (x_dim, y_dim)"""
    return x_dim**2 + y_dim**2 + LMB * x_dim * y_dim

# SOP-FBR


def vchpot(q_array, c_cheb, c_comb):
    """ Computes the value of the SOP-FBR potential. The workflow
        of this function can be understood as follows:

        The variables c_cheb and c_comb contain the values of the
        Chebyshev (Cljk) and interaction cofficients (Cj1...jk) respectively.

        Then we form all the possible combinations of the indexes than conform
        Cj1...jk using the values of m_k (MARRAY). This is achieved with the
        itertools function enumerate(itertools.product(*mcomb)). Iterating by
        this array we are effectively performing the nested sums.

        Second we define the hartree_product variable, that will contain the
        products of the SPPs for a given k (loop over NDOF). This products
        are made of sums of Chebyshev polynomials, wich are formed using
        the pol_tk, pol_mk and pol variables.

        The variable pol_tk contains the array of the Chebyshev coefficients
        splitted acording to the dimension of the polinomials (CHEBDIM), after
        performing this operation we have the Cljk separated by tk (CHEBDIM).
        Next the resulting array needs to be separated by k, which is achieved
        in the array pol_mk. This part may result a little bit tricky, so it
        will be further explained: for the general case (when we have different
        m_k) we have to split the array pol_tk (which is an array of subarrays
        all with size t_k) into an array of subarrays of subarrays (k, jk, tk),
        so the numpy function cumsum (cumulative sum) may come handy here. Take
        the hypotetical case of tk=2, k=3 and the m_k = [2, 3, 5], here the
        split of pol_tk must be done in 3 chunks of dimension :2, 2:5, 5: and
        every single chunk will contain (2, 3, 5) arrays of dimension 2.

        The set of indexes runing in one iteration of the sum (that is, the
        current jk's) is given by the comb variable, we will use this
        information to form the current  iteration's Chebyshev polynomials
        coefficients (stored in the variable pol). So in pol_mk, we first
        select the value of i (k) and then the current jk (comb[i]).
        The actual polynomials are created with cheby.chebval, and
        further avaluated in the requested grid point q_array[i].
    """
    vpot = 0
    mcomb = [np.arange(i) for i in MARRAY]
    pol_tk = np.array(np.split(c_cheb, c_cheb.shape[0] / CHEBDIM))
    pol_mk = np.array(np.split(pol_tk, np.cumsum(MARRAY))[0:-1])
    for idx, comb in enumerate(itertools.product(*mcomb)):
        hartree_product = 1
        for i in np.arange(NDOF):
            pol = pol_mk[i][comb[i]]
            hartree_product *= cheby.chebval(q_array[i], pol)
        vpot += c_comb[idx] * hartree_product
    return vpot


# RMS


def rho(carray, grad):
    """Computes de RMSE between V2D and VSOP-FBR
       Also prints to file relevant information
       about energies
    """
    if grad.size > 0:
        pass

    c_cheb = carray[:NCHEB]
    c_comb = carray[NCHEB::]
    e_vch = []
    for elem in G_AB:
        e_vch.append(vchpot(elem, c_cheb, c_comb))
    e_vch = np.array(e_vch)
    rms = np.sqrt(((e_vch - E_AB) ** 2).mean())

    with open("rms", "a") as file_target:
        file_target.write(str(rms) + "\n")
    with open("params_steps", "a") as param_steps:
        for elem in carray:
            param_steps.write(str(elem) + "\n")
        param_steps.write(" \n")
    with open("e_sopfbr", "a") as file_energies:
        for elem in e_vch:
            file_energies.write(str(elem) + "\n")
        file_energies.write(" \n")

    return rms

# Reference data and parameter guess input


X = np.linspace(-50, 50, num=100)
Y = np.linspace(-50, 50, num=100)
G_AB = np.concatenate((X[:, None], Y[:, None]), axis=1)
E_AB = np.vectorize(v2d)(X, Y)
np.savetxt("e_ref", E_AB)

# Total number of Chebyshev polinomial's coefficients
NCHEB = np.sum(MARRAY) * CHEBDIM

# Total number of configurations
NCOMB = np.prod(MARRAY)

# Initial Chebyshev expansion
C_CHEB = np.random.uniform(-60, 60, size=NCHEB)

# Initial Configuration interaction
C_COMB = np.random.uniform(low=0., high=1., size=NCOMB)
CARRAY = np.concatenate((C_CHEB, C_COMB))

# Fitting process

PDEV = 0.3
PARDIM = CARRAY.shape[0]
VALUE_UPPER = np.zeros(len(CARRAY))
VALUE_LOWER = np.zeros(len(CARRAY))

for j, el in enumerate(CARRAY):
    if CARRAY[j] > 0:
        VALUE_UPPER[j] = CARRAY[j] * (1.0 + PDEV)
        VALUE_LOWER[j] = CARRAY[j] * (1.0 - PDEV)
    if CARRAY[j] < 0:
        VALUE_UPPER[j] = CARRAY[j] * (1.0 - PDEV)
        VALUE_LOWER[j] = CARRAY[j] * (1.0 + PDEV)

MAXEVAL = 10
MINRMS = 0.01

OPT = nlopt.opt(nlopt.LN_BOBYQA, PARDIM)
OPT.set_lower_bounds(VALUE_LOWER)
OPT.set_upper_bounds(VALUE_UPPER)
OPT.set_min_objective(rho)
OPT.set_maxeval(MAXEVAL)
OPT.set_stopval(MINRMS)
X_OPT = OPT.optimize(CARRAY)
MINF = OPT.last_optimum_value()

np.savetxt("params_opt", X_OPT)
with open("minrms", "w") as minrms:
    minrms.write(str(MINF))

# Reducing directory entropy

TIMESTR = time.strftime("%Y%m%d_%H%M%S")  # A unique time stamp
OUT_DIR = "out_" + TIMESTR
os.makedirs(OUT_DIR)
os.rename("e_sopfbr", OUT_DIR + "/e_sopfbr")
os.rename("e_ref", OUT_DIR + "/e_ref")
os.rename("rms", OUT_DIR + "/rms")
os.rename("minrms", OUT_DIR + "/minrms")
os.rename("params_opt", OUT_DIR + "/params_opt")
os.rename("params_steps", OUT_DIR + "/params_steps")
