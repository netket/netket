import json as _json
from os import path as _path
from netket.legacy.vmc_common import tree_map as _tree_map


def _exists_json(prefix):
    return _path.exists(prefix + ".log") or _path.exists(prefix + ".wf")


def _to_json(ob):
    to_json = getattr(ob, "to_json", None)
    if to_json is not None:
        return ob.to_json()
    else:
        return ob


class JsonLog:
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
        self, output_prefix, mode="write", save_params_every=50, write_every=50
    ):
        # Shorthands for mode
        if mode == "w":
            mode = "write"
        elif mode == "a":
            mode = "append"
        elif mode == "x":
            mode = "fail"

        if not ((mode == "write") or (mode == "append") or (mode == "fail")):
            raise ValueError(
                "Mode not recognized: should be one of `[w]rite`, `[a]ppend` or `[x]`(fail)."
            )

        file_exists = _exists_json(output_prefix)

        starting_json_content = {"Output": []}

        if file_exists and mode == "append":
            # if there is only the .wf file but not the json one, raise an error
            if not _path.exists(output_prefix + ".log"):
                raise ValueError(
                    "History file does not exists, but wavefunction file does. Please change `output_prefix or set mode=`write`."
                )

            starting_json_content = _json.load(open(output_prefix + ".log"))

        elif file_exists and mode == "fail":
            raise ValueError(
                "Output file already exists. Either delete it manually or change `output_prefix`."
            )

        self._json_out = starting_json_content
        self._prefix = output_prefix
        self._write_every = write_every
        self._save_params_every = save_params_every
        self._old_step = 0
        self._steps_notflushed_write = 0
        self._steps_notflushed_pars = 0

    def __call__(self, step, item, machine):
        item["Iteration"] = step

        self._json_out["Output"].append(item)

        if (
            self._steps_notflushed_write % self._write_every == 0
            or step == self._old_step - 1
        ):
            self._flush_log()
        if (
            self._steps_notflushed_pars % self._save_params_every == 0
            or step == self._old_step - 1
        ):
            self._flush_params(machine)

        self._old_step = step
        self._steps_notflushed_write += 1
        self._steps_notflushed_pars += 1

    def _flush_log(self):
        with open(self._prefix + ".log", "w") as outfile:
            log_data = _tree_map(_to_json, self._json_out)
            _json.dump(log_data, outfile)
            self._steps_notflushed_write = 0

    def _flush_params(self, machine):
        machine.save(self._prefix + ".wf")
        self._steps_notflushed_pars = 0

    def flush(self, machine=None):
        """
        Writes to file the content of this logger.

        :param machine: optionally also writes the parameters of the machine.
        """
        self._flush_log()

        if machine is not None:
            self._flush_params(machine)
