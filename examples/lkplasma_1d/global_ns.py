import numpy as np
from numpy import pi, sin, cos, exp, sqrt, log, arctan2, max, array, linspace, conj, transpose
from petram.helper.variables import variable

#
order = 3
# constants   
freq = 78e6 # 41e6
w = freq * 2* pi           # omega

ny = 0.0
ky = ny*w/3e8

ntor=12
kz = ntor/0.6           # ntor = 10 at 0.6m
nz = 3e8*kz/w

B0 = 5.5
R0 = 0.68
B0R0 = B0*R0
a = 0.21

# maximum harmonics considered in hot conductivity
max_harm=5
# density
ne0 = 3e20
ne1 = 1.0e20

# temperature_e
Te0 = 2700.
Te1 = 10.
Ti0 = 2200.
Ti1 = 300. # 0.5
Ti0f = 2000.
Ti1f = 300.  # 0.5

# ion compositions
Ai, Aim = (2, 1)
Zi, Zim = (1, 1)
fraci, fracim = (0.94, 0.06)

@variable.jit.float
def bnorm_jit(x):
    return B0R0/x

@variable.jit.float
def dens_jit(x):
    rho = np.abs(x-R0)/a
    n_e = (ne0 - ne1) * (1 - rho**2) + ne1
    if rho > 1:
       return ne1*exp(-abs(rho-1)/0.2)
    return n_e

@variable.jit.float
def te_jit(x):
    rho = np.abs(x-R0)/a
    t_e = (Te0 - Te1) * (1 - rho**2) + Te1
    if t_e < Te1: return Te1
    return t_e

@variable.jit.float
def ti_jit(x):
    rho = np.abs(x-R0)/a
    t_i = (Ti0 - Ti1) * (1 - rho**2) + Ti1
    if t_i < Ti1: return Ti1
    return t_i

@variable.jit.float
def tim_jit(x):
    rho = np.abs(x-R0)/a
    t_i = (Ti0f- Ti1f) * (1 - rho**2) + Ti1f
    if t_i < Ti1f: return Ti1f
    return t_i
