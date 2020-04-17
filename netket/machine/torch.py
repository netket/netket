from .abstract_machine import AbstractMachine
import torch as _torch
import numpy as _np
import warnings


def _get_number_parameters(m):
    r"""Returns total number of variational parameters in a torch.nn.Module."""
    return sum(map(lambda p: p.numel(), _get_differentiable_parameters(m)))


def _get_differentiable_parameters(m):
    r"""Returns total number of variational parameters in a torch.nn.Module."""
    return filter(lambda p: p.requires_grad, m.parameters())


class Torch(AbstractMachine):
    def __init__(self, module, hilbert):
        self._module = _torch.jit.load(module) if isinstance(module, str) else module
        self._module.double()
        self._n_par = _get_number_parameters(self._module)
        self._parameters = list(_get_differentiable_parameters(self._module))
        self.n_visible = hilbert.size
        # TODO check that module has input shape compatible with hilbert size
        super().__init__(hilbert)

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
        if torch_pars.numel() != self.n_par:
            raise ValueError(
                "p has wrong shape: {}; expected [{}]".format(
                    torch_pars.size(), self.n_par
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
        r"""Returns the total number of trainable parameters in the machine.
        """
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

    def der_log(self, x, out=None):
        if len(x.shape) == 1:
            x = x[_np.newaxis, :]
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        x = _torch.tensor(x, dtype=_torch.float64)
        out = x.new_empty([x.size(0), 2, self._n_par], dtype=self._parameters[0].dtype)
        m = self._module(x)

        for i in range(x.size(0)):
            dws_real = _torch.autograd.grad(
                m[i, 0], self._parameters, retain_graph=True
            )
            dws_imag = _torch.autograd.grad(
                m[i, 1], self._parameters, retain_graph=True
            )
            _torch.cat([dw.flatten() for dw in dws_real], out=out[i, 0, ...])
            _torch.cat([dw.flatten() for dw in dws_imag], out=out[i, 1, ...])

        out_complex = _np.zeros((out.size(0), out.size(2)), dtype=_np.complex128)
        out_complex = out[:, 0, :].numpy() + 1.0j * out[:, 1, :].numpy()

        return out_complex.reshape(
            tuple(list(batch_shape) + list(out_complex.shape[-1:]))
        )

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
