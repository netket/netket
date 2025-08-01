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

from typing import Any, Union, TYPE_CHECKING
from collections.abc import Iterable
from numbers import Number

import numpy as np


if TYPE_CHECKING:
    import matplotlib
    import matplotlib.figure
    import matplotlib.axes

from netket.utils.dispatch import dispatch
from netket.utils.numbers import is_scalar
from netket.utils.types import Array, DType
from netket.utils.optional_deps import import_optional_dependency


def raise_if_len_not_match(length, expected_length, string):
    if length != expected_length:
        raise ValueError(
            f"""
            Length mismatch: expected object of length {expected_length}, but
            got object of length {length} for key {string}.
            """
        )


def maybecopy(maybe_arr):
    if isinstance(maybe_arr, np.ndarray):
        return maybe_arr.copy()
    else:
        return maybe_arr


def replace_none_with_nan(item):
    # recursively replace None with np.nan in lists
    if isinstance(item, list):
        return [replace_none_with_nan(subitem) for subitem in item]
    return np.nan if item is None else item


class History:
    """
    A class to store a time-series of arbitrary data.

    An History object stores several time-series all sharing the same
    time axis. History behaves like a dictionary, where the various key
    index the values (y axis) of the time-series. The time-axis is accessed
    through the attribute `History.iters`.

    It's possible to label one time-serie (one key) as the main value, so that
    when converting to numpy array (for example for plotting with pyplot) that
    axis is automatically picked by default.

    If only one time-series is provided, without a key, then its name will
    be `value`.

    For managing multiple History objects with independent time axes or
    complex nested structures, see :class:`~netket.utils.history.HistoryDict`
    and :func:`~netket.utils.history.accum_histories_in_tree`.
    """

    __slots__ = ("_value_dict", "_value_name", "_single_value", "_keys")

    def __init__(
        self,
        values: Any = None,
        iters: list | Array | None = None,
        dtype: DType | None = None,
        iter_dtype: DType | None = None,
        main_value_name: str | None = None,
    ):
        """
        Creates a new History object.

        Values should be an arbitrary type or container to initialize the
        History with.
        By default assumes that values correspond to the first iteration `0`.
        If `values` is a list or collection of lists, an array or range should
        be passed to values with the correct length.

        Optionally it's possible to specify the dtype of the data and of the
        time-axis.

        Args:
            values: a type/container of types containing the value at the first
                iteration, or an iterable/container of iterables
                containing the values at all iterations (in the latter, values
                must also be specified). It may also be the dictionary obtained
                from another History object, when converted to dictionary.
            iters: an optional iterable of iterations at which values correspond.
                If unspecified, assumes that values are logged at only one iteration.
            dtype: If no values or iters are passed, uses this dtype to store data
                if numerical
            iter_dtype: If no values or iters are passed, uses this dtype to store
                iteration numbers
            main_value_name: If data is a dict or object with to_dict method, this
                optional string labels an entry as being the main one.
        """
        single_value = False

        if values is None and iters is None:
            values = []
            iters = []
        elif iters is None:
            # If it's a dictionary, copy it
            if isinstance(values, dict):
                values = {k: maybecopy(v) for k, v in values.items()}
                if "iters" in values:
                    iters = values["iters"]
                    del values["iters"]
            if iters is None:
                iters = 0

        if is_scalar(iters):
            iters = np.array([iters], dtype=iter_dtype)
        elif isinstance(iters, list):
            iters = [np.nan if x is None else x for x in iters]
            iters = np.array(iters, dtype=iter_dtype)

        n_elements = len(iters)

        if is_scalar(values):
            values = {"value": values}
            single_value = True

        elif hasattr(values, "__array__"):
            values = {"value": values}
            single_value = True

        elif hasattr(values, "to_compound"):
            main_value_name, values = values.to_compound()
            single_value = len(values.keys()) == 1

        elif hasattr(values, "to_dict"):
            values = values.to_dict()
            single_value = len(values.keys()) == 1

        elif isinstance(values, dict) or hasattr(values, "items"):
            single_value = len(values.keys()) == 1

        else:
            values = {"value": [values]}
            single_value = True

        if single_value:
            main_value_name = list(values.keys())[0]

        value_dict = {"iters": iters}
        keys = []
        for key, val in values.items():
            if key == "iters":
                raise ValueError("cannot have a field called iters")

            if is_scalar(val):
                raise_if_len_not_match(1, n_elements, key)
                val = np.asarray([val], dtype=dtype)

            elif hasattr(val, "__array__"):
                val = np.asarray(val, dtype=dtype)
                if n_elements == 1 and val.ndim > 0:
                    val = np.reshape(val, (1, *val.shape))

                raise_if_len_not_match(len(val), n_elements, key)

            elif isinstance(val, list):
                # see below
                # val = np.asarray(val, dtype=dtype)

                # Recursively replace None with np.nan in nested lists
                # only relevant for loading from json on disk
                val = np.asarray(replace_none_with_nan(val), dtype=dtype)
            else:
                val = [val]

            value_dict[key] = val
            keys.append(key)

        self._value_dict = value_dict
        self._value_name = main_value_name
        self._single_value = single_value
        self._keys = keys

    @property
    def main_value_name(self) -> str | None:
        """The name of the main value in this history object, if defined."""
        return self._value_name

    @property
    def iters(self) -> Array:
        return self._value_dict["iters"]

    @property
    def values(self) -> Array:
        if self._value_name is None:
            raise ValueError("No main value defined for this history object.")
        return self._value_dict[self._value_name]

    def __len__(self) -> int:
        return len(self.iters)

    def __getitem__(self, key) -> Array:
        # if its an int corresponding to an element not inside the dict,
        # treat it as accessing a slice of a single element
        if isinstance(key, int) and key not in self:
            return self._get_slice(key)

        # support slice syntax
        if isinstance(key, slice):
            return self._get_slice(key)

        return self._value_dict[key]

    def _get_slice(self, slce: slice) -> "History":
        """
        get a slice of iterations from this history object
        """
        values_sliced = {}
        for key in self.keys():
            values_sliced[key] = self[key][slce]

        iters = self.iters[slce]

        hist = History(values_sliced, iters)
        hist._single_value = self._single_value
        hist._value_name = self._value_name

        return hist

    def __contains__(self, key: str) -> bool:
        return key in self._value_dict

    def keys(self) -> list:
        return self._keys

    def append(self, val: Any, it: Number | None = None):
        """
        Append another value to this history object.

        Args:
            val: the value in the next timestep
            it: the time corresponding to this new value. If
                not defined, increment by 1.
        """
        append(self, val, it)  # type: ignore

    def get(self) -> tuple[Array, Array]:
        """
        Returns a tuple containing times and values of this history object
        """
        return self.iters, self.values

    def to_dict(self) -> dict:
        """
        Converts the history object to dict.

        Used for serialization
        """
        return self._value_dict

    def __array__(self, *args, **kwargs) -> Array:
        """
        Automatically transform this object to a numpy array when calling
        asarray, by only considering the values and neglecting the times.
        """
        return np.asarray(self.values, *args, **kwargs)

    def __iter__(self):
        """
        You can iterate the values in history object.

        Returns the Iterator object.
        """
        return iter(zip(self.iters, self.values))

    def __getattr__(self, attr):
        # Allow users to access fields with . accessor patterns
        if attr in self._value_dict:
            return self._value_dict[attr]

        raise AttributeError

    def __repr__(self):
        if len(self.iters) < 5:
            iters_repr = repr(self.iters)
        else:
            iters_repr = (
                f"[{self.iters[0]}, {self.iters[1]}, ..."
                f" {self.iters[-2]}, {self.iters[-1]}] "
                f"({len(self.iters)} steps)"
            )
        return (
            "History("
            + f"\n   keys  = {self.keys()}, "
            + f"\n   iters = {iters_repr},"
            + "\n)"
        )

    def __str__(self):
        return f"History(keys={self.keys()}, n_iters={len(self.iters)})"

    def plot(
        self,
        *args,
        fig: "matplotlib.figure.Figure | None" = None,
        show: bool = True,
        all: bool | None = None,
        xscale: str | None = None,
        yscale: str | None = "auto",
        **kwargs,
    ) -> Union["matplotlib.axes.Axes", Iterable["matplotlib.axes.Axes"]]:
        """
        Plot the history object using matplotlib.

        This function is equivalent to calling `matplotlib.pyplot.plot` on the
        values of the history object. If multiple keys are present, it will
        plot only the main value by default. If `all=True`, it will plot all
        the keys in the history object.

        When plotting a single key, this function is equivalent to calling
        :code:`plt.plot(history.iters, history[key], *args, **kwargs)`.

        For the arguments and keyword arguments, refer to the documentation of
        :func:`matplotlib.pyplot.plot`. Below we document some extra arguments.

        Args:
            fig: A matplotlib figure object to plot the data on. If not provided,
                a new figure is created.
            all: If True, plot all the keys in the history object. If False, plot
                only the main value. If None, plot only the main value if it is
                defined, otherwise plot all the keys.
            show: If True, call `plt.show()` at the end of the function. If False,
                the plot is not shown.

        """
        plt = import_optional_dependency("matplotlib.pyplot", descr="plot")

        if self.main_value_name is None and all is None:
            all = True

        if fig is None:
            fig = plt.figure()  # type: ignore

        if all:
            n_plots = len(self.keys())
        else:
            n_plots = 1

        axes = fig.subplots(n_plots, sharex=True)
        if not isinstance(axes, Iterable):
            axes = [axes]

        keys_to_plot = self.keys() if all else [self.main_value_name]

        for ax, key in zip(axes, keys_to_plot):
            ax.plot(self.iters, self[key], label=key, *args, **kwargs)
            ax.set_ylabel(key)

            if xscale is not None:
                ax.set_xscale(xscale)
            if yscale is not None:
                if yscale == "auto":
                    if (
                        np.all(self[key] > 0)
                        and (np.max(self[key]) / np.min(self[key])) > 150
                    ):
                        yscale = "log"
                    else:
                        yscale = "linear"
                ax.set_yscale(yscale)

        plt.xlabel("Iterations")  # type: ignore
        plt.legend()  # type: ignore
        if show:
            plt.show()  # type: ignore

        if n_plots == 1:
            return axes[0]
        else:
            return axes


@dispatch
def append(self: History, val: Any):
    return append(self, val, None)


@dispatch
def append(self: History, val: History, it: Any):  # noqa: E0102, F811
    if not set(self.keys()) == set(val.keys()):
        raise ValueError("cannot concatenate MVHistories with different keys")

    if it is not None:
        raise ValueError("When concatenating histories, cannot specify the iteration.")

    for key in self.keys():
        self._value_dict[key] = np.concatenate([self[key], val[key]])

    self._value_dict["iters"] = np.concatenate([self.iters, val.iters])
    return self


@dispatch
def append(self: History, values: dict, it: Any):  # noqa: E0102, F811
    for key, val in values.items():
        _vals = self._value_dict[key]

        if isinstance(_vals, list):
            _vals.append(val)
        elif isinstance(_vals, np.ndarray):
            new_shape = (len(_vals) + 1,) + _vals.shape[1:]
            # try to resize in place the buffer so that we don't reallocate
            # and if we fail, resize tby reallocating to a new buffer.
            try:
                _vals.resize(new_shape)
            except ValueError:
                _vals = np.resize(_vals, new_shape)
                self._value_dict[key] = _vals

            _vals[-1] = val
        else:
            raise TypeError(f"Unknown accumulator type {type(_vals)} for key {key}.")

    try:
        self.iters.resize(len(self.iters) + 1)
    except ValueError:
        self._value_dict["iters"] = np.resize(self.iters, (len(self.iters) + 1))

    self.iters[-1] = it
    return self


@dispatch
def append(self: History, val: Any, it: Any):  # noqa: E0102, F811
    if self._single_value and is_scalar(val) or hasattr(val, "__array__"):
        append(self, {"value": val}, it)  # type: ignore
    elif hasattr(val, "to_compound"):
        append(self, val.to_compound()[1], it)  # type: ignore
    elif hasattr(val, "to_dict"):
        append(self, val.to_dict(), it)  # type: ignore
    else:
        append(self, {"value": val}, it)  # type: ignore
    return self
