from petram.phys.phys_const import mass_electron as me
from petram.phys.phys_const import q0 as q_base
from petram.phys.phys_const import epsilon0 as e0
import logging
import numpy as np
from numpy import (pi, sin, cos, exp, sqrt, log, arctan2, cross,
                   max, array, linspace, conj, transpose, arccos,
                   sum, zeros, dot, array, ascontiguousarray)
from numba import njit, void, int32, int64, float64, complex128, types
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints(
    'RF_DISPERSION_COLDPLASMA_NUMBA')


# slience log message
dprint1("Importing numba routines")
numba_logger = logging.getLogger('numba')
numba_clevel = numba_logger.level
numba_logger.setLevel(logging.WARNING)

# vacuum permittivity


qe = -q_base


iarray_ro = types.Array(int32, 1, 'C', readonly=True)
iarray2_ro = types.Array(int32, 2, 'C', readonly=True)
darray_ro = types.Array(float64, 1, 'C', readonly=True)


@njit(void(complex128[:, :], int64, int64))
def print_mat(mat, r, c):
    '''
    routine for debugging
    '''
    for i in range(r):
        for j in range(c):
            print("mat r="+str(i) + ", z="+str(j) + " :")
            print(mat[i, j])


@njit(complex128[:](float64, float64, float64, float64))
def SPD_el(w, Bnorm, dens, nu_eis):

    mass_eff = (1 + 1j*nu_eis/w)*me

    wp2 = dens * q_base**2/(mass_eff*e0)
    wc = qe * Bnorm/mass_eff

    Pterm = -wp2/w**2
    Sterm = -wp2/(w**2-wc**2)
    Dterm = wc*wp2/(w*(w**2-wc**2))

    return array([Sterm, Pterm, Dterm])


@njit(complex128[:](float64, float64, float64, float64))
def SPD_el_b(w, Bnorm, dens, wcol):

    w2 = w + 1j*wcol

    wp2 = dens * q_base**2/(me*e0)
    wc = qe * Bnorm/me

    Pterm = -wp2/w/w2
    Rterm = -wp2/w/(w2+wc)
    Lterm = -wp2/w/(w2-wc)

    Sterm = (Rterm + Lterm)/2.0
    Dterm = (Rterm - Lterm)/2.0

    return array([Sterm, Pterm, Dterm])


@njit(complex128[:](float64, float64, float64, float64, int64, float64))
def SPD_ion(w, Bnorm, dens, mass, charge, nu_ei):
    qi = charge*q_base
    mass_eff = (1 + 1j*nu_ei/w)*mass
    wp2 = dens * qi**2/(mass_eff*e0)
    wc = qi * Bnorm/mass_eff

    Pterm = -wp2/w**2
    Sterm = -wp2/(w**2-wc**2)
    Dterm = wc*wp2/(w*(w**2-wc**2))

    return array([Sterm, Pterm, Dterm])


@njit(complex128[:](float64, float64, float64, float64, int64, float64))
def SPD_ion_b(w, Bnorm, dens, mass, charge, wcol):

    qi = charge*q_base

    wp2 = dens * qi**2/(mass*e0)
    wc = qi * Bnorm/mass
    w2 = w + 1j*wcol

    Pterm = -wp2/w/w2
    Rterm = -wp2/w/(w2+wc)
    Lterm = -wp2/w/(w2-wc)

    Sterm = (Rterm + Lterm)/2.0
    Dterm = (Rterm - Lterm)/2.0

    return array([Sterm, Pterm, Dterm])


@njit(float64[:](float64[:], darray_ro, iarray_ro, float64, float64))
def f_collisions(denses, masses, charges, Te, ne):
    '''
    collisions
    '''
    nus = zeros(len(charges)+1)
    if ne == 0:
        return nus

    vt_e = sqrt(2*Te*q_base/me)
    LAMBDA = 1+12*pi*(e0*Te*q_base)**(3./2)/(q_base**3 * sqrt(ne))

    # electron-ion (electrons scattered by ions)
    for k in range(len(charges)):
        ni = denses[k]
        qi = charges[k]*q_base
        nu_ei = (qi**2 * qe**2 * ni *
                 log(LAMBDA)/(4 * pi*e0**2*me**2)/vt_e**3)
        nus[k+1] = nu_ei
    nus[0] = np.sum(nus)

    # ion_electrons (ions scattered by electrons)
    for k in range(len(charges)):
        ni = denses[k]
        qi = charges[k]*q_base
        nu_is = (qi**2 * qe**2 * ni *
                 log(LAMBDA)/(4 * pi*e0**2*me**2)/vt_e**3)*me/masses[k]
        nus[k+1] = nu_is

    return nus


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro, float64[:], float64, int32))
def _epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model):
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)

    S = 1 + 0j
    P = 1 + 0j
    D = 0j

    if col_model == 1:
        tmp = f_collisions(denses, masses, charges, Te[0], ne)
        nu_eis = np.zeros((len(masses)+1,))+np.max(tmp)
    elif col_model == 2:
        nu_eis = f_collisions(denses, masses, charges, Te[0], ne)
    elif col_model in (3, 4):
        nu_eis = Te
    else:
        nu_eis = np.array([0.]*(len(masses)+1))

    if ne > 0.:
        if col_model == 1 or col_model == 2:
            Se, Pe, De = SPD_el(w, b_norm, ne, nu_eis[0])
        else:
            Se, Pe, De = SPD_el_b(w, b_norm, ne, nu_eis[0])

        S += Se
        P += Pe
        D += De

    for dens, mass, charge, nu_ei in zip(denses, masses, charges, nu_eis[1:]):
        if dens > 0.:
            if col_model == 1 or col_model == 2:
                Si, Pi, Di = SPD_ion(w, b_norm, dens, mass, charge, nu_ei)
            else:
                Si, Pi, Di = SPD_ion_b(w, b_norm, dens, mass, charge, nu_ei)

            S += Si
            P += Pi
            D += Di

    M = array([[S,   -1j*D, 0j],
               [1j*D, S,    0j],
               [0j,   0j,   P]])

    return M


@njit(complex128[:](complex128, complex128, complex128, iarray_ro))
def adjust_terms(S, P, D, terms):

    if not terms[0]:       # S
        S *= 0.
    if not terms[1]:       # D
        D *= 0.
    if not terms[2]:       # P
        P *= 0.

    if not terms[3]:
        S = 1j*S.imag
        P = 1j*P.imag
        D = 1j*D.imag
    if not terms[4]:
        S = S.real
        P = P.real
        D = D.real
    ret = np.array([S, P, D])
    return ret


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro,
                         float64[:], float64, iarray2_ro, int32, int32))
def _epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms,
                       use_eye3, col_model):
    '''
    generalized Stix tensor
    terms defined in rf_dispersion_coldplasma
       stix_options = ("SDP", "SD", "SP", "DP", "P", "w/o xx", "None")
    '''
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)

    if col_model == 1:
        tmp = f_collisions(denses, masses, charges, Te[0], ne)
        nu_eis = np.zeros((len(masses)+1,))+np.max(tmp)
    elif col_model == 2:
        nu_eis = f_collisions(denses, masses, charges, Te[0], ne)
    elif col_model in (3, 4):
        nu_eis = Te
    else:
        nu_eis = np.array([0.]*(len(masses)+1))

    if use_eye3:
        M = array([[1.+0j,   0., 0.],
                   [0.,   1.+0j, 0.j],
                   [0.,   0.,    1.+0j]])
    else:
        M = array([[0.+0j,   0., 0.],
                   [0.,   0.+0j, 0.j],
                   [0.,   0.,    0.+0j]])

    icount = 0
    if ne > 0.:
        if col_model == 1 or col_model == 2:
            S, P, D = SPD_el(w, b_norm, ne, nu_eis[0])
        else:
            S, P, D = SPD_el_b(w, b_norm, ne, nu_eis[0])

        S, P, D = adjust_terms(S, P, D, terms[icount, :])
        M2 = array([[S, -1j*D, 0j], [1j*D, S, 0j], [0., 0j, P]])
        M += M2

    icount = 1
    for dens, mass, charge, nu_ei in zip(denses, masses, charges, nu_eis[1:]):
        if dens > 0.:
            if col_model == 1 or col_model == 2:
                S, P, D = SPD_ion(w, b_norm, dens, mass, charge, nu_ei)
            else:
                S, P, D = SPD_ion_b(w, b_norm, dens, mass, charge, nu_ei)

            # S, P, D = SPD_ion(w, b_norm, dens, mass, charge, nu_ei)
            S, P, D = adjust_terms(S, P, D, terms[icount, :])
            M2 = array([[S, -1j*D, 0j], [1j*D, S, 0j], [0., 0j, P]])
            M += M2

        icount = icount + 1

    return M

#
#  overload two functions so that Te/nu can be either array or single value.
#
def call_epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model):
    assert False, "this should not be called"

def call_epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms,
                       use_eye3, col_model):
    assert False, "this should not be called"

from numba.extending import overload
@overload(call_epsilonr_pl_cold_std)
def jit_std(w, B, denses, masses, charges, Te, ne, col_model):
    if isinstance(Te, types.Array):
        def array_impl(w, B, denses, masses, charges, Te, ne, col_model):
            Te = Te.astype(np.float64)
            return _epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model)
        return array_impl

    # Check if input is a primitive numeric type
    elif isinstance(Te, (types.Integer, types.Float)):
        def scalar_impl(w, B, denses, masses, charges, Te, ne, col_model):
            Te = np.zeros((13,), dtype=np.float64) + Te
            return _epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model)
        return scalar_impl

@overload(call_epsilonr_pl_cold_g)
def jit_g(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
    if isinstance(Te, types.Array):
        def array_impl(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
            Te = Te.astype(np.float64)
            return _epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model)
        return array_impl

    # Check if input is a primitive numeric type
    elif isinstance(Te, (types.Integer, types.Float)):
        def scalar_impl(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
            Te = np.zeros((13,), dtype=np.float64) + Te
            return _epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model)
        return scalar_impl

@njit
def epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model):
    return call_epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model)
@njit
def epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
    return call_epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model)

@njit(complex128[:, :](float64[:], complex128[:, :]))
def rotate_dielectric(B, M):
    #
    #  B : magnetic field.
    #  M : dielectric matrix.
    #
    B = ascontiguousarray(B)
    M = ascontiguousarray(M)

    def R1(ph):
        return array([[cos(ph), 0.,  sin(ph)],
                      [0,       1.,   0],
                      [-sin(ph), 0,  cos(ph)]], dtype=complex128)

    def R2(th):
        return array([[cos(th), -sin(th), 0],
                      [sin(th), cos(th), 0],
                      [0, 0,  1.]], dtype=complex128)

    #  B=(0,0,1) -> phi = 0,  th =0
    #  B=(1,0,0) -> phi = 90, th =0
    #  B=(0,1,0) -> phi = 90, th =90

    #  Bx = sin(phi) cos(th)
    #  By = sin(phi) sin(th)
    #  Bz = cos(phi)

    # B = [Bx, By, Bz]
    th = arctan2(B[1], B[0])
    ph = arctan2(B[0]*cos(th)+B[1]*sin(th), B[2])
    A = dot(R1(ph), dot(M, R1(-ph)))
    ans = dot(R2(th), dot(A, R2(-th)))

    return ans


#
# epsilonr with rotation
#
@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64[:], float64, int32))
def _epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne, col_model):
    '''
    standard SPD stix
    '''
    M = epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model)
    return rotate_dielectric(B, M)


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64[:], float64, iarray2_ro, int32, int32))
def _epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
    '''
    standard SPD stix
    '''
    M = epsilonr_pl_cold_g(w, B, denses, masses,
                           charges, Te, ne, terms, use_eye3, col_model)

    return rotate_dielectric(B, M)

def call_epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne, col_model):
    assert False, "this should not be called"

def call_epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
    assert False, "this should not be called"

@overload(call_epsilonr_pl_cold)
def jit_std_r(w, B, denses, masses, charges, Te, ne, col_model):
    if isinstance(Te, types.Array):
        def array_impl(w, B, denses, masses, charges, Te, ne, col_model):
            Te = Te.astype(np.float64)
            return _epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model)
        return array_impl

    # Check if input is a primitive numeric type
    elif isinstance(Te, (types.Integer, types.Float)):
        def scalar_impl(w, B, denses, masses, charges, Te, ne, col_model):
            Te = np.zeros((13,), dtype=np.float64) + Te
            return _epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne, col_model)
        return scalar_impl

@overload(call_epsilonr_pl_cold_generic)
def jit_g_r(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
    if isinstance(Te, types.Array):
        def array_impl(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
            Te = Te.astype(np.float64)
            return _epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model)
        return array_impl

    # Check if input is a primitive numeric type
    elif isinstance(Te, (types.Integer, types.Float)):
        def scalar_impl(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
            Te = np.zeros((13,), dtype=np.float64) + Te
            return _epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model)
        return scalar_impl

@njit
def epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne, col_model):
    return call_epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne, col_model)

@njit
def epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
    return call_epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model)


# back to the original log level
numba_logger.setLevel(numba_clevel)
