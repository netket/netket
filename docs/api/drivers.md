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

   AbstractDriver
   AbstractOptimizationDriver
   AbstractDynamicsDriver
```

For backward compatibility, {class}`~netket.driver.AbstractVariationalDriver`
remains an alias of {class}`~netket.driver.AbstractOptimizationDriver`, but new
code should use the canonical class names above.

(drivers_concrete)=
## Concrete drivers

Those are the optimization drivers already implemented in Netket.
Regarding VMC (ground-state optimization), we reccomend to use {class}`~netket.driver.VMC_SR` instead of the normal {class}`~netket.driver.VMC` in most cases.

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
.. currentmodule:: netket.driver

.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

   Infidelity_SR

```
