
from numpy import pi, sin, cos, exp, sqrt, log, arctan2, max, array, linspace, conj, transpose
from petram.helper.variables import variable
#
order=2
# constants   
freq = 18e9 
ny = 0.0
nz = 0.0
w = freq * 2* pi           # omega
ky = ny*w/3e8
kz = nz*w/3e8

Bnorm = 0.5 #[T]

@variable.jit.float
def dens(x):
   ne = 2e19 * (-x) if x < 0 else 0.0 #[m-3]
   if ne > 5e18: 
       ne = 5e18
   return ne

