"""
The present module performs the optimization of the coefficients of the SOP-FBR
representation of the 6D PES of the HONO cis-trans isomerization. It has
dependencies with on the Numpy, Tensorly and Scipy packages
"""
import numpy as np
from numpy.polynomial import chebyshev as cheby
import scipy.constants as sc
from scipy.optimize import minimize as mini
import tensorly as tl
import tensorly.decomposition as tldec

# System paramters

CHEBDIM = 5  # t_k (for the moment the same for all k)
CTEN_DIM = np.array([4, 4, 4, 4, 4, 4])
CCHEB_SLICE = np.cumsum(CTEN_DIM)
AUCM = sc.physical_constants['hartree-inverse meter relationship'][0] * 1e-2

# SOP-FBR


def vchpot(q_array, c_cheb, c_comb):
    """ Computes the value of the SOP-FBR potential by first
        conforming the vij(k) matrices, then reshaping
        the core tensor, and penforming the tensor dot product.
    """
    cheb_tk = np.array(np.split(c_cheb, c_cheb.shape[0] / CHEBDIM))
    cheb_tk_mk = np.array(np.split(cheb_tk, CCHEB_SLICE)[0:-1])
    v_matrices = []
    for kdof, m_kp in enumerate(CTEN_DIM):
        v_kp = np.zeros(m_kp)
        for j_kp in np.arange(m_kp):
            preroot = j_kp * [0]
            root = list(cheb_tk_mk[kdof][j_kp])
            aug_coeff = preroot + root
            v_kp[j_kp] = cheby.chebval(
                q_array[kdof], aug_coeff)
        v_matrices.append(v_kp)

    core = c_comb.reshape(CTEN_DIM)
    prod = tl.tucker_tensor.tucker_to_tensor((core, v_matrices))

    return prod


# RMS

def rho(c_cheb, c_comb):
    """Objective function of the macroiterations
    (only depends on the Chebyshev coefficients)
    """

    e_vch = []
    for elem in G_AB:
        e_vch.append(vchpot(elem, c_cheb, c_comb))
    e_vch = np.array(e_vch)

    rms = np.sqrt(((e_vch.flatten() - E_AB) ** 2).mean())

    with open("rms_rho", "a") as file_target:
        file_target.write(str(rms) + "\n")

    return rms


def rho_mat(mats, subcore, c_cheb):
    """Objective funcion for the microiterations (matrixes)"""
    resh_mat = np.split(mats.reshape(-1, 2), np.cumsum(CTEN_DIM))[0:-1]
    core = tl.tucker_tensor.tucker_to_tensor((subcore, resh_mat))
    c_comb = core.flatten()

    e_vch = []
    for elem in G_AB:
        e_vch.append(vchpot(elem, c_cheb, c_comb))
    e_vch = np.array(e_vch)

    rms = np.sqrt(((e_vch.flatten() - E_AB) ** 2).mean())

    with open("rms_rho_mat", "a") as file_target:
        file_target.write(str(rms) + "\n")

    return rms


def rho_subcore(subcore, mats, c_cheb):
    """Objective funcion for the microiterations (subcore)"""
    subcore = subcore.reshape(D0DIM)
    core = tl.tucker_tensor.tucker_to_tensor((subcore, mats))
    c_comb = core.flatten()

    e_vch = []
    for elem in G_AB:
        e_vch.append(vchpot(elem, c_cheb, c_comb))
    e_vch = np.array(e_vch)

    rms = np.sqrt(((e_vch.flatten() - E_AB) ** 2).mean())

    with open("rms_rho_subcore", "a") as file_target:
        file_target.write(str(rms) + "\n")

    return rms


# Reference data and parameter guess input

DATA = np.loadtxt('ref_ab')
G_AB = DATA[:, :-1]
E_AB = DATA[:, -1]

# Scaling data

R1MAX, R1MIN = 1.95, 2.65
R2MAX, R2MIN = 2.2, 3.6
R3MAX, R3MIN = 1.5, 2.5
T1MAX, T1MIN = -0.08715574, -0.70710678
T2MAX, T2MIN = 0.25881905, -0.76604444
PHMAX, PHMIN = 0, np.pi

G_AB[:, 0] = 2 * (G_AB[:, 0] - R2MIN) / (R2MAX - R2MIN) - 1
G_AB[:, 1] = 2 * (G_AB[:, 1] - R3MIN) / (R3MAX - R3MIN) - 1
G_AB[:, 2] = 2 * (G_AB[:, 2] - R1MIN) / (R1MAX - R1MIN) - 1
G_AB[:, 3] = 2 * (np.cos(G_AB[:, 3]) - T2MIN) / (T2MAX - T2MIN) - 1
G_AB[:, 4] = 2 * (np.cos(G_AB[:, 4]) - T1MIN) / (T1MAX - T1MIN) - 1
G_AB[:, 5] = 2 * (G_AB[:, 5] - PHMIN) / (PHMAX - PHMIN) - 1


# Total number of Chebyshev polinomial's coefficients
NCHEB = np.sum(CTEN_DIM) * CHEBDIM

# Total parameter array and dimension (cchev||ctens)
CARRAY = np.loadtxt('params_init')
PARDIM = CARRAY.shape[0]

# Subcore definitions
D0DIM = np.array([2, 2, 2, 2, 2, 2])

# Fitting process

TOL = 1
RESER = np.inf
RESER_MIN = np.inf
ITERS = 0
while RESER >= TOL:
    if ITERS == 10000:
        break

    # Optimization of factor matrices (macroiterations)
    PARAMS_OPT_CHEB = mini(rho, CARRAY[:NCHEB], args=(CARRAY[NCHEB::]),
                           method='Powell')
    CARRAY[:NCHEB] = PARAMS_OPT_CHEB.x

    # Tucker decomposition of core tensor
    CORA, MATS = tldec.tucker(CARRAY[NCHEB::].reshape(CTEN_DIM),
                              ranks=D0DIM)

    # Optimization of factor matrices (redefine functions)
    FLAT_MAT = np.concatenate(MATS)
    PARAMS_OPT_MAT = mini(rho_mat, FLAT_MAT, args=(CORA, CARRAY[:NCHEB]),
                          method='Powell')
    MATS_OPT = np.split(PARAMS_OPT_MAT.x.reshape(-1, 2),
                        np.cumsum(CTEN_DIM))[0:-1]

    # Optimization of subcore tensor
    PARAMS_OPT_D0 = mini(rho_subcore, CORA, args=(MATS_OPT, CARRAY[:NCHEB]),
                         method='Powell')
    CORA_OPT = PARAMS_OPT_D0.x.reshape(D0DIM)

    # New core tensor
    CARRAY[NCHEB::] = tl.tucker_tensor.tucker_to_tensor((CORA_OPT,
                                                         MATS_OPT)).flatten()
    # New RMSE
    RESER = PARAMS_OPT_D0.fun

    # Check evolution of RMSE
    ITERS += 1
    if RESER < RESER_MIN:
        RESER_MIN = RESER
        np.savetxt('opt_params', CARRAY)
        with open('out_reser', 'a') as outr:
            outr.write(str(ITERS) + " RMS =  " + str(RESER) + "\n")

    ITERS += 1
