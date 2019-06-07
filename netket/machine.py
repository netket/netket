from ._C_netket.machine import *


def MPSPeriodicDiagonal(hilbert, bond_dim, symperiod=-1):
    return MPSPeriodic(hilbert, bond_dim, diag=True, symperiod=symperiod)
