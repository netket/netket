# Configuration Options

NetKet exposes a few configuration options which can be set through environment variables by doing something like
```bash
# without exporting it
NETKET_DEBUG=1 python ...

# by exporting it
export NETKET_DEBUG=1
python ...

# by setting it within python
python
>>> import os
>>> os.environ["NETKET_DEBUG"] = "1"
>>> import netket as nk
>>> print(netket.config.netket_debug)
True
```
Some configuration options can also be changed at runtime by setting it as:
```python
>>> import netket as nk
>>> nk.config.netket_debug = True
>>> ...
```

You can always query the value of an option by accessing the `nk.config` module:
```python
>>> import netket as nk
>>> print(nk.config.netket_debug)
False
>>> nk.config.netket_debug = True
>>> print(nk.config.netket_debug)
True
```

Please note that not all configurations can be set at runtime, and some will raise an error.

Options are used to activate experimental or debug functionalities or to disable some parts of netket.
Please keep in mind that all options related to experimental or internal functionalities might be removed in a future release.

# List of configuration options

```{eval-rst} 
.. list_config_options::
``