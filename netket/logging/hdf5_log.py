import os
import time
import h5py
import numpy as np
from os import path as _path
from ..logging import RuntimeLog
from ..stats import Stats
from ..jax import tree_ravel


def _allkeys(obj):
    "Recursively find all keys in an h5py.Group."
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + _allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys

def _exists_hdf5(prefix):
    return _path.exists(prefix + ".hdf5")

_mode_shorthands = {
    'write': 'w',
    'append': 'a',
    'fail': 'x'
}

class HDF5Log(RuntimeLog):
    """
    HDF5 Logger, that can be passed with keyword argument `logger` to Monte
    Carlo drivers in order to serialize the output data of the simulation.

    The logger has support for scalar numbers, NumPy/JAX arrays, and netket.stats.Stats objects.
    These are stored as individual groups within a HDF5 file, under the main group `data/`:

    - scalars are stored as a group with one dataset values of shape (n_steps,) containing the logged values,
    - arrays are stored in the same way, but with values having shape (n_steps, *array_shape),
    - netket.stats.Stats are stored as a group containing each field (Mean, Variance, etc...) as a separate dataset.

    Importantly, each group has a dataset `iters`, which tracks the iteration number of the logged quantity.  

    If the model state is serialized, then it is serialized as a dataset in the group `parameters/`.
    The target of the serialization are the parameters of the variational state,
    which are stored in a dataset `parameters/value` as a flattened array.

    Data can be deserialized by calling :code:`f = h5py.File(filename, 'r')` and
    inspecting the datasets as a dictionary, i.e. :code:`f['data/energy/Mean']`
    """

    def __init__(
        self,
        output_prefix: str,
        mode: str = "write",
        save_params_every: int = 50,
        write_every: int = 50,
        save_params: bool = True,
        autoflush_cost: float = 0.005,
    ):
        """
        Construct a HDF5 Logger.

        Args:
            output_prefix: the name of the output files before the extension
            save_params_every: every how many iterations should machine parameters be
                flushed to file
            write_every: every how many iterations should data be flushed to file
            mode: Specify the behaviour in case the file already exists at this
                output_prefix. Options are
                - `[w]rite`: (default) overwrites file if it already exists;
                - `[x]` or `fail`: fails if file already exists;
            save_params: bool flag indicating whether variables of the variational state
                should be serialized at some interval. The output file is overwritten
                every time variables are saved again.
            autoflush_cost: Maximum fraction of runtime that can be dedicated to
                serializing data. Defaults to 0.005 (0.5 per cent)
        """
        super().__init__()


        if not ((mode == "write") or (mode == "append") or (mode == "fail")):
            raise ValueError(
                "Mode not recognized: should be one of `[w]rite`, `[a]ppend` or"
                "`[x]`(fail)."
            )
        mode = _mode_shorthands[mode]

        file_exists = _exists_hdf5(output_prefix)

        if file_exists and mode == "x":
            raise ValueError(
                "Output file already exists. Either delete it manually or"
                "change `output_prefix`."
            )

        dir_name = _path.dirname(output_prefix)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        self._prefix = output_prefix
        self._file_mode = mode
        self._file_open = h5py.File(self._prefix+'.hdf5', self._file_mode)

        self._write_every = write_every
        self._save_params_every = save_params_every
        self._init_step = None
        self._old_step = 0
        self._last_flush_log_step = -1
        self._steps_notflushed_log = 0
        self._steps_notflushed_params = 0
        self._save_params = save_params

        self._autoflush_cost = autoflush_cost
        self._last_flush_time = time.time()
        self._last_flush_runtime = 0.0

        self._flush_log_time = 0.0
        self._flush_params_time = 0.0  

    def __call__(self, step, log_data, variational_state):
        if self._init_step is None:
            self._init_step = step
        old_step = self._old_step
        super().__call__(step, log_data, variational_state)

        # Check if the time from the last flush is higher than the maximum
        # allowed runtime cost of flushing
        elapsed_time = time.time() - self._last_flush_time
        # On windows, the precision of `time.time` is much lower than that on Linux,
        # so `elapsed_time` may be essentially zero.
        # We add 1e-7 to avoid the zero division error.
        flush_anyway = (
            self._last_flush_runtime / (elapsed_time + 1e-7) < self._autoflush_cost
        )

        if self._init_step == step:
            self._init_dataset(log_data, variational_state)
        else:
            if (
                self._steps_notflushed_log % self._write_every == 0
                or step == old_step - 1
                or flush_anyway
            ):
                self._flush_log(step)
                print(f"step={step}: flushing log")
            if (
                self._steps_notflushed_params % self._save_params_every == 0
                or step == old_step - 1
            ):
                self._flush_params(step, variational_state)
                print(f"step={step}: flushing params")

        self._old_step = step
        self._steps_notflushed_log += 1
        self._steps_notflushed_params += 1

    def _init_dataset(self, log_data, variational_state):
        f = self._file_open
        for key, item in log_data.items():
            if hasattr(item, '_device'):
                # Convert JAX arrays into NumPy arrays
                item = np.array(item)
            if isinstance(item, Stats):
                # Stats dataclasses
                f.create_dataset('data/'+key+'/iters', data=[0], maxshape=(None,))
                data = item.to_dict()
                for subkey, value in data.items():
                    f.create_dataset('data/'+key+'/'+subkey, data=[value], maxshape=(None,))
            elif np.isscalar(item):
                # Scalars
                f.create_dataset('data/'+key+'/iters', data=[0], maxshape=(None,))
                f.create_dataset('data/'+key+'/value', data=[item], maxshape=(None,))
            elif isinstance(item, np.ndarray):
                # Arrays
                f.create_dataset('data/'+key+'/iters', data=[0], maxshape=(None,))
                f.create_dataset('data/'+key+'/value', data=[item], maxshape=(None, item.shape))
        # Parameters
        params, _ = tree_ravel(variational_state.parameters)
        f.create_dataset('parameters/iters', data=[0], maxshape=(None,))
        f.create_dataset('parameters/value', data=[params], maxshape=(None, params.shape[0]))
        f.flush()

        self._last_flush_log_step = 0
        self._steps_notflushed_log = 0
        self._steps_notflushed_params = 0

    def _flush_log(self, step):
        self._last_flush_time = time.time()

        f = self._file_open
        len_update = self._steps_notflushed_log
        for key, history in self.data.items():
            iters = 'data/'+key+'/iters'
            f[iters].resize((f[iters].shape[0]+len_update), axis=0)
            f[iters][-len_update:] = self._last_flush_log_step+np.arange(len_update)+1
            for subkey in history.keys():
                value = 'data/'+key+'/'+subkey
                f[value].resize((f[value].shape[0]+len_update), axis=0)
                f[value][-len_update:,...] = history[subkey][-len_update:]
        f.flush()

        self._last_flush_log_step = step
        self._steps_notflushed_log = 0
        self._last_flush_runtime = time.time() - self._last_flush_time
        self._flush_log_time += self._last_flush_runtime

    def _flush_params(self, step, variational_state):
        if not self._save_params:
            return
        
        _time = time.time()

        f = self._file_open
        params, _ = tree_ravel(variational_state.parameters)
        f['parameters/iters'].resize((f['parameters/iters'].shape[0]+1), axis=0)
        f['parameters/value'].resize((f['parameters/value'].shape[0]+1), axis=0)
        f['parameters/iters'][-1:] = step
        f['parameters/value'][-1:,:] = params
        f.flush()

        self._steps_notflushed_params = 0
        self._flush_params_time += time.time() - _time

    def flush(self, variational_state=None):
        """
        Writes to file the content of this logger.

        Args:
            variational_state: optionally also writes the parameters of the machine.
        """
        self._flush_log()

        if variational_state is not None:
            self._flush_params(variational_state)

    def __repr__(self):
        _str = f"HDF5Log('{self._prefix}', mode={self._file_mode}, "
        _str = _str + f"autoflush_cost={self._autoflush_cost})"
        _str = _str + "\n  Runtime cost:"
        _str = _str + f"\n  \tLog:    {self._flush_log_time}"
        _str = _str + f"\n  \tParams: {self._flush_params_time}"
        return _str
    