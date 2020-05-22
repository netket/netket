import json as _json
from os import path as _path


class CombineLogs:
    """
    Creates a Json Logger sink object, that can be passed with keyword argument `logger` to Monte
    Carlo drivers in order to serialize the outpit data of the simulation.

    Args:
        output_prefix: the name of the output files before the extension
        save_params_every: every how many iterations should machine parameters be flushed to file
        write_every: every how many iterations should data be flushed to file
        mode: Specify the behaviour in case the file already exists at this output_prefix. Options
        are
        - `[w]rite`: (default) overwrites file if it already exists;
        - `[a]ppend`: appends to the file if it exists, overwise creates a new file;
        - `[x]` or `fail`: fails if file already exists;
    """

    def __init__(
        self, *args,
    ):
        self._loggers = [lg for lg in args]

    def __call__(self, step, item, machine):
        for logger in self._loggers:
            logger(step, item, machine)

    def _flush_log(self):
        for logger in self._loggers:
            logger._flush_log()

    def _flush_params(self, machine):
        for logger in self._loggers:
            logger._flush_params(machine)

    def flush(self, machine=None):
        """
        Writes to file the content of this logger.

        :param machine: optionally also writes the parameters of the machine.
        """
        for logger in self._loggers:
            logger.flush(machine)
