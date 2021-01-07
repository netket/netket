from .abstract_machine import AbstractMachine
import torch as _torch
import numpy as _np
import warnings
from netket.utils import backpack_available as _backpack_available

if _backpack_available:
    from netket.utils import backpack


def _get_number_parameters(m):
    r"""Returns total number of variational parameters in a torch.nn.Module."""
    return sum(map(lambda p: p.numel(), _get_differentiable_parameters(m)))


def _get_differentiable_parameters(m):
    r"""Returns total number of variational parameters in a torch.nn.Module."""
    return filter(lambda p: p.requires_grad, m.parameters())


class Torch(AbstractMachine):
    def __init__(self, module, hilbert, dtype=float, use_backpack=False):
        self._module = _torch.jit.load(module) if isinstance(module, str) else module
        self._module.double()
        self._n_par = _get_number_parameters(self._module)
        self._parameters = list(_get_differentiable_parameters(self._module))
        self._use_backpack = use_backpack
        if self._use_backpack:
            if not _backpack_available:
                raise RuntimeError("BackPACK was not successfully imported")
            self._module = backpack.extend(self._module)
        # TODO check that module has input shape compatible with hilbert size
        super().__init__(hilbert, dtype=dtype)

    @property
    def parameters(self):
        return (
            _torch.cat(
                tuple(p.view(-1) for p in _get_differentiable_parameters(self._module))
            )
            .detach()
            .numpy()
            .astype(_np.complex128)
        )

    def assign_beta(self, beta):
        self._module.beta = beta
        return

    def save(self, filename):
        _torch.save(self._module.state_dict(), filename)
        return

    def load(self, filename):
        self._module.load_state_dict(_torch.load(filename))
        return

    @parameters.setter
    def parameters(self, p):
        if not _np.all(p.imag == 0.0):
            warnings.warn(
                "PyTorch machines have real parameters, imaginary part will be discarded"
            )
        torch_pars = _torch.from_numpy(p.real)
        if torch_pars.numel() != self._n_par:
            raise ValueError(
                "p has wrong shape: {}; expected [{}]".format(
                    torch_pars.size(), self._n_par
                )
            )
        i = 0
        for x in map(
            lambda x: x.view(-1), _get_differentiable_parameters(self._module)
        ):
            x.data.copy_(torch_pars[i : i + len(x)].data)
            i += len(x)

    @property
    def n_par(self):
        r"""Returns the total number of trainable parameters in the machine."""
        return self._n_par

    def log_val(self, x, out=None):
        if len(x.shape) == 1:
            x = x[_np.newaxis, :]
        batch_shape = x.shape[:-1]

        with _torch.no_grad():
            t_out = self._module(_torch.from_numpy(x)).numpy().view(_np.complex128)
        if out is None:
            return t_out.reshape(batch_shape)
        _np.copyto(out, t_out.reshape(-1))

        return out

    def _der_log_backpack(self, x, out):
        def jacobian():
            return _torch.cat(
                [
                    p.grad_batch.flatten(start_dim=1)
                    for p in self._module.parameters()
                    if p.requires_grad
                ],
                dim=1,
            ).numpy()

        m_sum_real = self._module(x)[:, 0].sum(dim=0)
        with backpack.backpack(backpack.extensions.BatchGrad()):
            m_sum_real.backward()
        out.real = jacobian()

        m_sum_imag = self._module(x)[:, 1].sum(dim=0)
        with backpack.backpack(backpack.extensions.BatchGrad()):
            m_sum_imag.backward()
        out.imag = jacobian()

        self._module.zero_grad()
        return out

    def der_log(self, x, out=None):
        x = x.reshape(-1, x.shape[-1])
        if out is None:
            out = _np.empty([x.shape[0], self._n_par], dtype=_np.complex128)
        x = _torch.tensor(x, dtype=self._dtype)

        if self._use_backpack:
            return self._der_log_backpack(x, out)
        else:
            m = self._module(x)
            for i in range(x.size(0)):
                dws_real = _torch.autograd.grad(
                    m[i, 0], self._parameters, retain_graph=True
                )
                dws_imag = _torch.autograd.grad(
                    m[i, 1], self._parameters, retain_graph=True
                )
                out[i, ...].real = _torch.cat([dw.flatten() for dw in dws_real]).numpy()
                out[i, ...].imag = _torch.cat([dw.flatten() for dw in dws_imag]).numpy()
            self._module.zero_grad()
            return out

    def vector_jacobian_prod(
        self, x, vec, out=None, conjugate=True, return_jacobian=False
    ):
        x = x.reshape(-1, x.shape[-1])
        vec = vec.flatten()
        if out is None:
            out = _np.empty(self._n_par, dtype=_np.complex128)

        if return_jacobian:
            jac = self.der_log(x)
            out[...] = _np.einsum("n,nk->k", vec, jac.conj())
            if not conjugate:
                out[...] = out.conj()
            return out, jac
        else:
            x = _torch.tensor(x, dtype=self._dtype)
            y = self._module(x)
            self._module.zero_grad()
            vecj = _torch.empty(x.shape[0], 2, dtype=_torch.float64)

            def gradient():
                return _torch.cat(
                    [
                        p.grad.flatten()
                        for p in self._module.parameters()
                        if p.requires_grad
                    ]
                ).numpy()

            vecj[:, 0] = _torch.from_numpy(vec.real)
            vecj[:, 1] = _torch.from_numpy(vec.imag)
            y.backward(vecj, retain_graph=True)
            out.real = gradient()
            self._module.zero_grad()

            vecj[:, 0] = _torch.from_numpy(vec.imag)
            vecj[:, 1] = _torch.from_numpy(-vec.real)
            if not conjugate:
                vecj *= -1.0
            y.backward(vecj)
            out.imag = gradient()
            self._module.zero_grad()
            return out

    @property
    def state_dict(self):
        from collections import OrderedDict

        return OrderedDict(
            [(k, v.detach().numpy()) for k, v in self._module.state_dict().items()]
        )

    def __repr__(self):
        head = super().__repr__()
        module = "\n│  ".join([""] + repr(self._module).split("\n"))
        return head + module + "\n└"


class TorchLogCosh(_torch.nn.Module):
    """
    Log(cosh) activation function for PyTorch modules
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return -input + _torch.nn.functional.softplus(2.0 * input)


class TorchView(_torch.nn.Module):
    """
    Reshaping layer for PyTorch modules
    """

    def __init__(self, shape):
        """
        Init method.
        """
        super().__init__()  # init the base class
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
