# Copyright 2024 The NetKet Authors - All rights reserved.
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

from typing import Optional, Callable

import time
import inspect
import functools
import contextlib

from rich.tree import Tree
from rich.panel import Panel


from netket.utils import struct, display

CURRENT_TIMER_STACK = []


@display.rich_repr
class Timer(struct.Pytree, mutable=True):
    """
    Measures how much time was spent in a timer, with possible
    sub-timers.

    Can be used as a context manager to time a scope, and will show
    up timers used inside of the scope if present.

    If you directly construct a timer, you cannot nest it inside another
    timer. If you are building a library function you should instead use
    :func:`netket.utils.timing.timed_scope`.

    Example:

        Time a scope

        >>> import netket as nk
        >>> import time
        >>>
        >>> with nk.utils.timing.Timer() as t:
        ...    time.sleep(1)  # This line and the ones below are indented
        ...    with nk.utils.timing.timed_scope("subfunction 1"):
        ...       time.sleep(0.5)
        ...    with nk.utils.timing.timed_scope("subfunction 2"):
        ...       time.sleep(0.25)
        >>>
        >>> t  # doctest: +SKIP
        ╭──────────────────────── Timing Information ─────────────────────────╮
        │ Total: 1.763                                                        │
        │ ├── (28.7%) | subfunction 1 : 0.505 s                               │
        │ └── (14.3%) | subfunction 2 : 0.252 s                               │
        ╰─────────────────────────────────────────────────────────────────────╯

    """

    total: float
    sub_timers: dict
    _start_time: float = 0.0

    def __init__(self):
        """
        Constructs a new timer object.

        Does not accept any argument.
        """
        self.total = 0.0
        self.sub_timers = {}

    def __rich__(self, indent=0, total_above=None):
        # The initial part of the string representation for the current object

        tree = Tree(f"Total: {self.total:.3f}")
        self._rich_walk_tree_(tree)
        return Panel(tree, title="Timing Information")

    def _rich_walk_tree_(self, tree):
        attributed = 0.0
        for key, sub_timer in self.sub_timers.items():
            if sub_timer.total / self.total > 0.01:
                percentage = 100 * (sub_timer.total / self.total)
                attributed += sub_timer.total

                sub_tree = tree.add(
                    f"({percentage:.1f}%) | {key} : {sub_timer.total:.3f} s"
                )
                sub_timer._rich_walk_tree_(sub_tree)

        # This prints an 'other' line
        # if attributed > 0 and attributed / self.total < 0.98:
        #    rest = self.total - attributed
        #    percentage = 100 * (rest / self.total)
        #    tree.add(f"({percentage:.1f}%) | other : {rest:.3f} s")

    def __iadd__(self, o):
        if not isinstance(o, Timer):
            return NotImplemented
        self.total += o.total
        for k, v in o.sub_timers.items():
            if k in self.sub_timers:
                self.sub_timers[k] += v
            else:
                self.sub_timers[k] = v
        return self

    def get_subtimer(self, name: str):
        if name not in self.sub_timers:
            self.sub_timers[name] = Timer()
        return self.sub_timers[name]

    def __enter__(self):
        global CURRENT_TIMER_STACK
        CURRENT_TIMER_STACK.append(self)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        t_end = time.perf_counter()
        current_timer = t_end - self._start_time
        self.total += current_timer
        _ = CURRENT_TIMER_STACK.pop()
        assert _ is self


@contextlib.contextmanager
def timed_scope(name: str = None, force: bool = False):
    """
    Context manager used to mark a scope to be timed individually
    by NetKet timers.

    If name is not specified, the file name and line number
    is used.

    If `force` is not specified, the timer only runs if a top-level
    timer is in use as well. If `force` is specified, the timer
    and nested timers will always run.

    Example:

        Time a section of code

        >>> import netket as nk
        >>> import time
        >>>
        >>> with nk.utils.timing.timed_scope(force=True) as t:
        ...    time.sleep(1)  # This line and the ones below are indented
        ...    with nk.utils.timing.timed_scope("subfunction 1"):
        ...       time.sleep(0.5)
        ...    with nk.utils.timing.timed_scope("subfunction 2"):
        ...       time.sleep(0.25)
        >>>
        >>> t  # doctest: +SKIP
        ╭──────────────────────── Timing Information ─────────────────────────╮
        │ Total: 1.763                                                        │
        │ ├── (28.7%) | subfunction 1 : 0.505 s                               │
        │ └── (14.3%) | subfunction 2 : 0.252 s                               │
        ╰─────────────────────────────────────────────────────────────────────╯



    Args:
        name: Name to use for the timing of this line.
        force: whether to always time, even if no top level timer
            is in use
    """
    __tracebackhide__ = True

    global CURRENT_TIMER_STACK
    if force or len(CURRENT_TIMER_STACK) > 0:
        if name is None:
            # If no name specified look 2 frames up (1 frame is
            # @contextmanager, 2 frames is caller) to get filename and
            # line number.
            caller_frame = inspect.stack()[2]
            frame = caller_frame[0]
            info = inspect.getframeinfo(frame)
            name = f"{info.filename}:{info.lineno}"

        if len(CURRENT_TIMER_STACK) == 0:
            timer = Timer()
        else:
            timer = CURRENT_TIMER_STACK[-1].get_subtimer(name)
        with timer:
            yield timer
    else:  # disabled
        yield None


def timed(fun: Callable = None, name: Optional[str] = None):
    """
    Marks the decorated function to be timed individually in
    NetKet timing scopes.

    If name is not specified, the qualified name of the
    function is used.

    The profiling is disabled if no global timer is active.

    Args:
        fun: Function to be decorated
        name: Name to use for the timing of this line.
    """
    if fun is None:
        return functools.partial(timed, name=name)

    if name is None:
        if hasattr(fun, "__qualname__"):
            name = fun.__qualname__
        else:
            name = fun.__name__

    @functools.wraps(fun)
    def timed_function(*args, **kwargs):
        __tracebackhide__ = True
        with timed_scope(name):
            return fun(*args, **kwargs)

    return timed_function


def clear_timers():
    """
    Resets all timers.
    """
    global CURRENT_TIMER_STACK
    CURRENT_TIMER_STACK.clear()
