"""Numba helpers shared by local kinetic-plasma dielectric backends.

These routines deliberately do not select, or implement, a dielectric tensor
backend.  They describe the wave-vector conventions used by the RF models.
"""

import numpy as np
from numpy import array, cross, sqrt, abs
from numba import njit, complex128, float64, int64
from petram.phys.phys_const import c as speed_of_light


@njit(complex128[:](float64[:], float64, float64[:], int64, complex128[:, :]))
def eval_npara_nperp(ptx, omega, kpakpe, kpe_mode, e_cold):
    if kpe_mode == 1:  # fast wave
        npara = speed_of_light*kpakpe[0]/omega
        S = e_cold[0, 0]
        D = e_cold[0, 1]*1j
        P = e_cold[2, 2]

        nperpsq = (D**2 - (npara**2 - S)**2)/(npara**2 - S)
        nperp = sqrt(abs(nperpsq))
        #nperp = nperp.real
    elif kpe_mode == 2:  # slow wave
        npara = speed_of_light*kpakpe[0]/omega
        S = e_cold[0, 0]
        D = e_cold[0, 1]*1j
        P = e_cold[2, 2]
        nperpsq = -(npara**2 - S)*P/S
        nperp = sqrt(abs(nperpsq))
        #nperp = nperp.real
    else:
        npara = speed_of_light*kpakpe[0]/omega
        nperp = speed_of_light*kpakpe[1]/omega

    return array([npara, nperp], dtype=complex128)


#
# routines to define kpe as vector
#


@njit(float64[:](float64[:], float64, float64, float64[:], float64[:]))
def eval_kpe_std(ptx, kpara, kperp, k, b):
    #
    #   kpe vector is given by k. it just project kpevec to a plane normal to
    #   b
    #

    bn = b/sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    kn = k/sqrt(k[0]**2 + k[1]**2 + k[2]**2)
    tmp = cross(bn, kn)
    ret = -cross(bn, tmp)

    return ret


@njit(float64[:](float64[:], float64, float64, float64[:], float64[:]))
def eval_kpe_em1d(ptx, kpara, kperp, k, b):
    #
    #   kvec specifies the direction of k on r-z plane
    #
    #  k[2] is not used

    bn = b/sqrt(b[0]**2 + b[1]**2 + b[2]**2)

    kz = -(k[0]*bn[0] + k[1]*bn[1])/bn[2]
    kvec = array([k[0], k[1], kz])

    return kvec


@njit(float64[:](float64[:], float64, float64, float64[:], float64[:]))
def eval_kpe_em2da(ptx, kpara, kperp, k, b):
    #
    #   kvec specifies the direction of k on r-z plane
    #
    #  k[1] is not used

    bn = b/sqrt(b[0]**2 + b[1]**2 + b[2]**2)

    ktor = -(k[0]*bn[0] + k[2]*bn[2])/bn[1]
    kvec = array([k[0], ktor, k[2]])

    return kvec


@njit(float64[:](float64[:], float64, float64, float64[:], float64[:]))
def eval_kpe_em2d(ptx, kpara, kperp, k, b):
    #
    #   kvec specifies the direction of k on r-z plane
    #
    #  k[2] is not used

    bn = b/sqrt(b[0]**2 + b[1]**2 + b[2]**2)

    kz = -(k[0]*bn[0] + k[1]*bn[1])/bn[2]
    kvec = array([k[0], k[1], kz])

    return kvec
