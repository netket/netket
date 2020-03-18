import json as _json
import netket as _nk
from os import path as _path


def _exists_json(prefix):
    return _path.exists(prefix + ".log") or _path.exists(prefix + ".wf")


class JsonLog:
    """
    Creates a Json Logger sink object, that can be passed with keyword argument `logger` to Monte
    Carlo drivers in order to serialize the outpit data of the simulation.

    Args:
        output_prefix: the name of the output files before the extension
        save_params_every: every how many iterations should machine parameters be flushed to file
        write_every: every how many iterations should data be flushed to file
        mode: Specify the behaviour in case the file already exists at this output_prefix. Options
        are `[o]verwrite` (default), `[a]ppend`, `[n]ewfile` (creates a new file with an incremental
        name) or `[f]ail`.
    """

    def __init__(
        self, output_prefix, save_params_every=50, write_every=50, mode="overwrite"
    ):
        # Shorthands for mode
        if mode == "o":
            mode = "overwrite"
        elif mode == "a":
            mode = "append"
        elif mode == "n":
            mode = "newfile"
        elif mode == "f":
            mode = "fail"

        if not (
            (mode == "overwrite")
            or (mode == "append")
            or (mode == "newfile")
            or (mode == "fail")
        ):
            raise ValueError(
                "Mode not recognized: should be one of `[o]verwrite`, `[a]ppend`, `[n]ewfile or `[f]ail`."
            )

        file_exists = _exists_json(output_prefix)

        starting_json_content = {"Output": []}

        if file_exists and mode == "newfile":
            i = 1
            # Generate a new output_prefix with an incremental counter
            while i < 1001:
                _prefix = output_prefix + "_{}".format(i)
                if not _exists_json(_prefix):
                    output_prefix = _prefix
                    break
                i = i + 1

            # give up after we tried too many files to avoid an infinite loop
            if i is 1000:
                raise IOError(
                    "More than 1000 files exist with that prefix. Please change ouput_prefix manually."
                )
        elif file_exists and mode == "append":
            # if there is only the .wf file but not the json one, raise an error
            if not _path.exists(output_prefix + ".log"):
                raise ValueError(
                    "History file does not exists, but wavefunction file does. Please change `output_prefix or set mode=`overwrite`."
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

    def __call__(self, step, item, machine):
        item["Iteration"] = step

        self._json_out["Output"].append(item)

        if step % self._write_every == 0 or step == self._old_step - 1:
            self._flush_log()
        if step % self._save_params_every == 0 or step == self._old_step - 1:
            self._flush_params(machine)

        self._old_step = step

    def _flush_log(self):
        with open(self._prefix + ".log", "w") as outfile:
            _json.dump(self._json_out, outfile)

    def _flush_params(self, machine):
        if _nk.MPI.rank() == 0:
            machine.save(self._prefix + ".wf")

    def flush(self, machine=None):
        """
        Writes to file the content of this logger.

        :param machine: optionally also writes the parameters of the machine.
        """
        self._flush_log()

        if machine is not None:
            self._flush_params(machine)
