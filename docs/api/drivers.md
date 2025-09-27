(netket_drivers_api)=
# netket.drivers

```{eval-rst}
.. currentmodule:: netket.driver

```

(drivers_abstract_classes)=
## Abstract Interface

Those are the abstract classes you can inherit from to implement your own driver

```{eval-rst}
.. currentmodule:: netket.driver

.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

   AbstractVariationalDriver
```

(drivers_concrete)=
## Concrete drivers

Those are the optimization drivers already implemented in Netket:

```{eval-rst}
.. currentmodule:: netket.driver

.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

   VMC
   VMC_SR
   SteadyState

```

## State fitting

```{eval-rst}
.. currentmodule:: netket

.. autosummary::
   :toctree: _generated/experimental/driver
   :nosignatures:

   experimental.driver.Infidelity_SR

```
