# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Part of the code in this file is taken from
# https://github.com/Nuclear-Physics-with-Machine-Learning/JAX_QMC_Public/blob/main/LICENSE
# to which the original copyright applies
#                    GNU AFFERO GENERAL PUBLIC LICENSE


from typing import Optional, TYPE_CHECKING

import os
import logging
import socket

import jax
import numpy as np

if TYPE_CHECKING:
    import mpi4py


def autoset_default_gpu(COMM: Optional["mpi4py.MPI.Intracomm"], verbose: bool = False):
    """
    Automatically sets the default jax gpu device to be the one identified by a local rank.

    This function is used to avoid having to set `CUDA_VISIBLE_DEVICES` or equivalents when using NetKet
    in MPI mode, where every rank only uses one GPU.

    This code is only ran if we see that there is more than 1 device and it is a gpu. In other cases this
    code does not run.
    """
    devices = jax.devices()
    if len(devices) > 1 and any(d.device_kind == "gpu" for d in devices):
        local_rank = get_local_rank(COMM)
        jax.config.set("jax_default_device", devices[local_rank])
        logger = logging.getLogger()
        logger.info(
            f"Determined that rank {COMM.Get_rank()} will be using GPU[{local_rank}/{len(devices)}]"
        )


def get_local_rank(COMM: Optional["mpi4py.MPI.Intracomm"] = None, verbose=False) -> int:
    """
    Used to detect GPU rank. Tries multiple strategies.
    """
    local_rank = get_local_rank_from_env()
    if local_rank is not None:
        return local_rank
    if COMM is not None:
        return get_local_rank_from_mpi4py(COMM, verbose=verbose)


def get_local_rank_from_env() -> Optional[int]:
    """
    Correct local rank from env variables set by some MPI implementations
    (notably OpenMPI)

    Returns:
        int or None
    """
    local_rank_key_options = [
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
        "PALS_LOCAL_RANKID",
    ]

    # testable default value:
    local_rank = None
    for key in local_rank_key_options:
        if key in os.environ:
            local_rank = os.environ[key]
            logger = logging.getLogger()
            logger.info(f"Determined local rank through environment variable {key}")
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[key]
            local_rank = int(local_rank)
    return local_rank


def get_local_rank_from_mpi4py(
    COMM: "mpi4py.MPI.Intracomm", verbose: bool = False
) -> Optional[int]:
    """
    Uses MPI4PY and a list of hostnames to find the correct local rank.
    """
    # Try the last-ditch effort of home-brewed local rank deterimination
    # The strategy here is to split into sub communicators
    # Each sub communicator will be just on a single host,
    # And that communicator will assign ranks that can be interpretted
    # as local ranks.

    # To subdivide, each host will need to use a unique key.
    # We'll rely on the hostname and order them all.

    hostname = socket.gethostname()
    # host_key = host_key %
    all_hostnames = COMM.gather(hostname, root=0)

    if COMM.Get_rank() == 0:
        # Order all the hostnames, and find unique ones
        unique_hosts = np.unique(all_hostnames)
        # Numpy automatically sorts them.
    else:
        unique_hosts = None

    # Broadcast the list of hostnames:
    unique_hosts = COMM.bcast(unique_hosts, root=0)

    # Find the integer for this host in the list of hosts:
    i = int(np.where(unique_hosts == hostname)[0])
    # print(f"{hostname} found itself at index {i}")

    new_comm = COMM.Split(color=i)
    if verbose:
        print(
            f"Global rank {COMM.Get_rank()} of {COMM.Get_size()} mapped to local rank {new_comm.Get_rank()} of {new_comm.Get_size()} on host {hostname}",
            flush=True,
        )

    # The rank in the new communicator - which is host-local only - IS the local rank:
    return int(new_comm.Get_rank())
