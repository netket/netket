from .abstract_machine import AbstractMachine
import torch as _torch
import numpy as _np
import warnings


def _get_number_parameters(m):
    r"""Returns total number of variational parameters in a torch.nn.Module."""
    return sum(
        map(lambda p: p.numel(), filter(lambda p: p.requires_grad, m.parameters()))
    )


class Torch(AbstractMachine):
    def __init__(self, module, hilbert):

        self._module = _torch.jit.load(module) if isinstance(module, str) else module
        self._module.double()
        self._n_par = _get_number_parameters(self._module)

        # TODO check that module has input shape compatible with hilbert size
        super().__init__(hilbert)

    @property
    def parameters(self):
        return (
            _torch.cat(tuple(p.view(-1) for p in self._module.parameters()))
            .detach()
            .numpy()
            .astype(_np.complex128)
        )

    @parameters.setter
    def parameters(self, p):
        if not _np.all(p.imag == 0.0):
            warnings.warn(
                "PyTorch machines have real parameters, imaginary part will be discarded"
            )
        self._torch_pars = _torch.from_numpy(p.real)
        if self._torch_pars.numel() != self.n_par:
            raise ValueError(
                "p has wrong shape: {}; expected [{}]".format(
                    self._torch_pars.size(), self.n_par
                )
            )
        i = 0
        for x in map(lambda x: x.view(-1), self._module.parameters()):
            x.data.copy_(self._torch_pars[i : i + len(x)].data)
            i += len(x)

    @property
    def n_par(self):
        r"""Returns the total number of trainable parameters in the machine.
        """
        return self._n_par

    def log_val(self, x, out=None):

        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        with _torch.no_grad():
            self._t_out = _torch.from_numpy(out.view(_np.float64)).view(-1, 2)
            self._t_out.copy_(self._module(_torch.from_numpy(x)))
        return out

    def der_log(self, x, out=None):
        return NotImplementedError

    def vector_jacobian_prod(self, x, vec, out=None):

        if out is None:
            out = _np.empty(self.n_par, dtype=_np.complex128)

        def write_to(dst):
            dst = _torch.from_numpy(dst)
            i = 0
            for g in (
                p.grad.flatten() for p in self._module.parameters() if p.requires_grad
            ):
                dst[i : i + g.numel()].copy_(g)
                i += g.numel()

        def zero_grad():
            for g in (p.grad for p in self._module.parameters() if p.requires_grad):
                if g is not None:
                    g.zero_()

        vecj = _torch.empty(x.shape[0], 2, dtype=_torch.float64)

        def get_vec(is_real):
            if is_real:
                vecj[:, 0] = _torch.from_numpy(vec.real)
                vecj[:, 1] = _torch.from_numpy(vec.imag)
            else:
                vecj[:, 0] = _torch.from_numpy(vec.imag)
                vecj[:, 1] = _torch.from_numpy(-vec.real)
            return vecj

        y = self._module(_torch.from_numpy(x))
        zero_grad()
        y.backward(get_vec(True), retain_graph=True)
        write_to(out.real)
        zero_grad()
        y.backward(get_vec(False))
        write_to(out.imag)

        return out

    @property
    def is_holomorphic(self):
        r"""PyTorch models are real-valued only, thus non holomorphic.
        """
        return False

    @property
    def state_dict(self):
        from collections import OrderedDict

        return OrderedDict(
            [(k, v.detach().numpy()) for k, v in self._module.state_dict().items()]
        )
