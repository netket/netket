import json
import netket as _nk


class JsonLog:
    """
    TODO
    """

    def __init__(self, output_prefix, save_params_every=50, write_every=50):
        self._json_out = {"Output": []}
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

    def _flush_log(self):
        if _nk.MPI.rank() == 0:
            with open(self._prefix + ".log", "w") as outfile:
                json.dump(self._json_out, outfile)

    def _flush_params(self, machine):
        if _nk.MPI.rank() == 0:
            machine.save(self._prefix + ".wf")

    def flush(self, machine=None):
        self._flush_log()

        if machine is not None:
            self._flush_params(machine)
