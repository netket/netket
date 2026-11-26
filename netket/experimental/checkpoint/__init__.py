"""
The checkpointer interface of :meth:`netket.driver.AbstractDriver.run`
(``checkpointer=``): :class:`AbstractCheckpointer`, to implement one, and the
:class:`Checkpoint` it reads back.

.. warning::

    This interface is experimental and may change without a deprecation period:
    use or implement it, but do not build on it as a stable API.
"""

from netket._src.driver.checkpointer import (
    AbstractCheckpointer as AbstractCheckpointer,
    Checkpoint as Checkpoint,
)
