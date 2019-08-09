# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

from __future__ import absolute_import
from collections import OrderedDict
import numpy as np
import torch

from .cxx_machine import CxxMachine

__all__ = ["Torch"]


def _get_number_parameters(m):
    r"""Returns total number of variational parameters in a torch.nn.Module."""
    return sum(
        map(lambda p: p.numel(), filter(lambda p: p.requires_grad, m.parameters()))
    )


class Torch(CxxMachine):
    def __init__(self, hilbert, module):
        super(Torch, self).__init__(hilbert)
        self._module = torch.jit.load(module) if isinstance(module, str) else module
        self._module.double()
        n_par = _get_number_parameters(self._module)

        self._n_par = lambda: n_par

    def _get_parameters(self):
        return (
            torch.cat(tuple(p.view(-1) for p in self._module.parameters()))
            .detach()
            .numpy()
        )

    def _is_holomorphic(self):
        return False

    def _set_parameters(self, p):
        if not np.all(p.imag == 0.0):
            raise ValueError("PyTorch machines have real parameters")
        p = torch.from_numpy(p.real)
        if p.numel() != self._n_par():
            raise ValueError(
                "p has wrong shape: {}; expected [{}]".format(p.size(), self._n_par())
            )
        i = 0
        for x in map(lambda x: x.view(-1), self._module.parameters()):
            x.copy_(p[i : i + len(x)])
            i += len(x)

    def _log_val(self, x, out=None):
        out = torch.from_numpy(out.view(np.float64)).view(-1, 2)
        out.copy_(self._module(torch.from_numpy(x)))

    def _der_log(self, x, out=None):
        raise NotImplementedError

    def jvp_log(self, v, delta, out=None):
        v = torch.from_numpy(v)
        if out is None:
            out = np.empty(self._n_par(), dtype=np.complex128)

        def write_to(dst):
            dst = torch.from_numpy(dst)
            i = 0
            for g in (
                x.grad.flatten() for x in self._module.parameters() if x.requires_grad
            ):
                dst[i : i + g.numel()].copy_(g)
                i += g.numel()

        def zero_grad():
            for g in (x.grad for x in self._module.parameters() if x.requires_grad):
                if g is not None:
                    g.zero_()

        grad = torch.empty(v.size(0), 2, dtype=torch.float64)

        def get_delta(is_real):
            if is_real:
                grad[:, 0] = torch.from_numpy(delta.real)
                grad[:, 1] = 0
            else:
                grad[:, 0] = 0
                grad[:, 1] = torch.from_numpy(delta.imag)
            return grad

        y = self._module(v)
        zero_grad()
        y.backward(get_delta(True), retain_graph=True)
        write_to(out.real)
        zero_grad()
        y.backward(get_delta(False))
        write_to(out.imag)
        return out

    def state_dict(self):
        return OrderedDict(
            [(k, v.detach().numpy()) for k, v in self._module.state_dict().items()]
        )
