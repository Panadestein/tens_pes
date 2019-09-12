"""
The present module performs the optimization of the coefficients of the SOP-FBR
representation of a 2D PES of the shape (V(x, y) = x^2 + y^2 + lambda*xy)
It makes use of Numpy and Tensorly packages
"""
import os
import time
import numpy as np
from numpy.polynomial import chebyshev as cheby
import tensorly as tl
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
    """ Computes the value of the SOP-FBR potential by first
        conforming the vij(k) matrixes, then reshaping the core
        tensor, and penforming the tensor dot product.
    """
    coeff_tk = np.array(np.split(c_cheb, c_cheb.shape[0] / CHEBDIM))
    chev_coeff = np.array(np.split(coeff_tk, np.cumsum(MARRAY))[0:-1])
    v_matrixes = []
    for kdof, m_kp in enumerate(MARRAY):
        v_kp = np.zeros((q_array.shape[0], m_kp))
        for i_kp, val in enumerate(q_array):
            for j_kp in np.arange(m_kp):
                v_kp[i_kp, j_kp] = cheby.chebval(
                    val[kdof], chev_coeff[kdof][j_kp])
        v_matrixes.append(v_kp)
    v_matrixes = np.array(v_matrixes)

    prod = c_comb.reshape(MARRAY)
    for idx, elem in enumerate(v_matrixes):
        prod = tl.tenalg.mode_dot(prod, elem, idx)

    return prod


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
    e_vch = vchpot(G_AB, c_cheb, c_comb)
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


X = np.linspace(-50, 50, num=15)
Y = np.linspace(-50, 50, num=15)
G_AB = np.concatenate((X[:, None], Y[:, None]), axis=1)
E_AB = v2d(X[:, None], Y[None, :])
np.savetxt("e_ref", E_AB)

# Total number of Chebyshev polinomial's coefficients
NCHEB = np.sum(MARRAY) * CHEBDIM

# Total number of configurations
NCOMB = np.prod(MARRAY)

# Initial Chebyshev expansion
C_CHEB = np.random.uniform(-60, 60, size=NCHEB)

# Initial Configuration interaction
C_COMB = np.random.uniform(low=0., high=1., size=NCOMB)

# Total parameter array and dimension
CARRAY = np.concatenate((C_CHEB, C_COMB))
PARDIM = CARRAY.shape[0]

# Fitting process

PDEV = 0.3
VALUE_UPPER = np.zeros(len(CARRAY))
VALUE_LOWER = np.zeros(len(CARRAY))

for j, el in enumerate(CARRAY):
    if CARRAY[j] > 0:
        VALUE_UPPER[j] = CARRAY[j] * (1.0 + PDEV)
        VALUE_LOWER[j] = CARRAY[j] * (1.0 - PDEV)
    if CARRAY[j] < 0:
        VALUE_UPPER[j] = CARRAY[j] * (1.0 - PDEV)
        VALUE_LOWER[j] = CARRAY[j] * (1.0 + PDEV)

MAXEVAL = 1
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

TIMESTR = time.strftime("%Y%m%d_%H%M%S")  # A unique time stamp for out_dir
OUT_DIR = "out_" + TIMESTR
os.makedirs(OUT_DIR)
os.rename("e_sopfbr", OUT_DIR + "/e_sopfbr")
os.rename("e_ref", OUT_DIR + "/e_ref")
os.rename("rms", OUT_DIR + "/rms")
os.rename("minrms", OUT_DIR + "/minrms")
os.rename("params_opt", OUT_DIR + "/params_opt")
os.rename("params_steps", OUT_DIR + "/params_steps")
