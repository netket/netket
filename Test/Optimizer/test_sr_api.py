import pytest

import itertools
from functools import partial


import flax
import jax

import numpy as np
from numpy import testing

import jax.numpy as jnp
import jax.flatten_util
from jax.scipy.sparse.linalg import cg

import netket as nk
from netket.optimizer import qgt
from netket.optimizer.qgt import qgt_onthefly_logic as _sr_onthefly_logic

from .. import common

pytestmark = common.skipif_mpi


def test_deprecated_sr():
    with pytest.warns(FutureWarning):
        sr = nk.optimizer.sr.SRLazyCG()

    with pytest.warns(FutureWarning):
        sr = nk.optimizer.sr.SRLazyGMRES()
