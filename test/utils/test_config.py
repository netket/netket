import netket as nk
import pytest

with pytest.raises(RuntimeError):
    nk.config.netket_experimental = True

assert isinstance(nk.config.netket_debug, bool)

with pytest.raises(TypeError):
    nk.config.netket_debug = 1
