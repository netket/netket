(netket_logging_api)=
# netket.logging

```{eval-rst}
.. currentmodule:: netket.logging

```

This module contains the loggers that can be used with the optimization drivers by passing them to the `out=` keyword argument of 
{meth}`~netket.driver.AbstractVariationalDriver.run`.


```{eval-rst}
.. inheritance-diagram:: netket.logging
   :top-classes: netket.logging.RuntimeLog
   :parts: 1

```


```{eval-rst}
.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

   RuntimeLog
   JsonLog
   StateLog
   TensorBoardLog

```

In the [netket.experimental](api-experimental) module there are also some experimental loggers such as the {class}`HDF5 logger <netket.experimental.logging.HDF5Log>`
