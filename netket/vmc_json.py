import json
import netket as _nk

class _JsonLog:
    """
    TODO
    """

    def __init__(
            self, output_prefix, n_iter, obs=None, save_params_every=50, write_every=50
    ):
        self._json_out = {"Output": []}
        self._prefix = output_prefix
        self._write_every = write_every
        self._save_params_every = save_params_every
        self._n_iter = n_iter
        self._obs = obs if obs else {}

    def __call__(self, step, driver):
        item = {"Iteration": step}
        stats = driver.estimate(self._obs)
        stats["Energy"] = driver.energy
        for key, value in stats.items():
            st = value.asdict()
            st["Mean"] = st["Mean"].real
            item[key] = st

        self._json_out["Output"].append(item)

        if step % self._write_every == 0 or step == self._n_iter - 1:
            if _nk.MPI.rank() == 0:
                with open(self._prefix + ".log", "w") as outfile:
                    json.dump(self._json_out, outfile)
        if step % self._save_params_every == 0 or step == self._n_iter - 1:
            if _nk.MPI.rank() == 0:
                driver._machine.save(self._prefix + ".wf")
