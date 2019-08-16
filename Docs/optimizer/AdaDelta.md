# AdaDelta
AdaDelta Optimizer.
 Like RMSProp, [AdaDelta](http://arxiv.org/abs/1212.5701) corrects the
 monotonic decay of learning rates associated with AdaGrad,
 while additionally eliminating the need to choose a global
 learning rate $$ \eta $$. The NetKet naming convention of
 the parameters strictly follows the one introduced in the original paper;
 here $$E[g^2]$$ is equivalent to the vector $$\mathbf{s}$$ from RMSProp.
 $$E[g^2]$$ and $$E[\Delta x^2]$$ are initialized as zero vectors.

 $$
 \begin{align}
 E[g^2]^\prime_k &= \rho E[g^2] + (1-\rho)G_k(\mathbf{p})^2\\
 \Delta p_k &= - \frac{\sqrt{E[\Delta x^2]+\epsilon}}{\sqrt{E[g^2]+ \epsilon}}G_k(\mathbf{p})\\
 E[\Delta x^2]^\prime_k &= \rho E[\Delta x^2] + (1-\rho)\Delta p_k^2\\
 p^\prime_k &= p_k + \Delta p_k\\
 \end{align}
 $$

## Class Constructor
Constructs a new ``AdaDelta`` optimizer.

|Argument|   Type    |           Description           |
|--------|-----------|---------------------------------|
|rho     |float=0.95 |Exponential decay rate, in [0,1].|
|epscut  |float=1e-07|Small $$\epsilon$$ cutoff.       |

### Examples
Simple AdaDelta optimizer.

```python
>>> from netket.optimizer import AdaDelta
>>> op = AdaDelta()

```



## Class Methods 
### reset
Member function resetting the internal state of the optimizer.


