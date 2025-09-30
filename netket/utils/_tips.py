# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
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

"""Display random tips when NetKet is imported interactively."""

import random
import sys

TIPS = [
    # Drivers
    "Prefer the new nk.driver.VMC_SR over VMC which supports minSR and SPRING.",
    "Use driver.run(..., timeit=True) to know where your dominant cost is.",
    # Sampling
    "With many Markov Chains (e.g GPUs), n_discard_per_chain>5 is often inefficient.",
    "On GPUs an efficient choice is to run ~512 Markov Chains per GPU.",
    r"If timeit=True signals high \% spent sampling n_discarded, consider lowering it.",
    # Operators
    "Avoid LocalOperator with have N-body operators (N>4). PauliStrings is ok.",
    # Loggers
    "You can load logged data with nk.utils.history.HistoryDict.from_file(data.log).",
    "You can plot data with JsonLog.data['Energy'][:-30].plot().",
    # Variational states
    "To build H|ψ⟩ use nk.vqs.apply_operator(H, vstate_ψ).",
    "log(ψ) ∈ ℜ → ψ = exp(logψ) ∈ ℜ₊ (real NN gives only positive wave-function).",
    "You can use flax.linen, flax.nnx and equinox to define neural networks.",
    # utils
    "You must cite NetKet according to our policy. Use nk.cite() to find out how.",
    "Debug multi-node HPC? `djaxrun -np 2 python Examples/Sharding/multi_process.py`",
    # extras
    "uv is a replacement for pip which helps you follow good software practices.",
    "You can disable these tips by setting export NETKET_NO_TIPS=1 in your .bashrc.",
    "😍 Love? 😡 Hate? these tips? Let us know at https://forms.gle/ej8eDGFRu2mww5mQ9",
]


def _is_interactive():
    """Check if Python is running in an interactive environment."""
    try:
        # Check if we're in IPython/Jupyter
        ipython = get_ipython()  # noqa: F821
        # IPython is available, but check if we're running a script
        # In IPython/Jupyter interactive mode, __file__ is not defined in the main namespace
        # We check the config to see if it's truly interactive
        if hasattr(ipython, "config"):
            # If IPython is running a file/script, we're not interactive
            if hasattr(ipython, "user_ns") and "__file__" in ipython.user_ns:
                return False
        # Check if we're in a Jupyter notebook or IPython shell
        return ipython.__class__.__name__ in [
            "TerminalInteractiveShell",
            "ZMQInteractiveShell",
        ]
    except NameError:
        # Check if we're in a regular Python REPL
        return hasattr(sys, "ps1")


def show_random_tip():
    """Display a random tip using rich formatting if in interactive mode."""
    if not _is_interactive():
        return

    from netket.utils import config

    if config.netket_no_tips:
        return

    import textwrap
    from rich.console import Console

    tip = random.choice(TIPS)
    console = Console()

    # Wrap text with hanging indent
    prefix = "∣NK⟩ Tip: "
    width = console.width
    wrapped = textwrap.fill(
        tip,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
    )

    # Apply styling only to the prefix
    lines = wrapped.split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            # First line: style the prefix
            console.print(f"[bold cyan]{prefix}[/bold cyan]{line[len(prefix):]}")
        else:
            # Subsequent lines: already have proper indentation
            console.print(line)
