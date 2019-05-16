# AmsGrad
AmsGrad Optimizer.
 In some cases, adaptive learning rate methods such as AdaMax fail
 to converge to the optimal solution because of the exponential
 moving average over past gradients. To address this problem,
 Sashank J. Reddi, Satyen Kale and Sanjiv Kumar proposed the
 AmsGrad [update algorithm](https://openreview.net/forum?id=ryQu7f-RZ).
 The update rule for $$\mathbf{v}$$ (equivalent to $$E[g^2]$$ in AdaDelta
 and $$\mathbf{s}$$ in RMSProp) is modified such that $$v^\prime_k \geq v_k$$
 is guaranteed, giving the algorithm a "long-term memory" of past gradients.
 The vectors $$\mathbf{m}$$ and $$\mathbf{v}$$ are initialized to zero, and
 are updated with the parameters $$\mathbf{p}$$:

 $$
 \begin{aligned}
 m^\prime_k &= \beta_1 m_k + (1-\beta_1)G_k(\mathbf{p})\\
 v^\prime_k &= \beta_2 v_k + (1-\beta_2)G_k(\mathbf{p})^2\\
 v^\prime_k &= \mathrm{Max}(v^\prime_k, v_k)\\
 p^\prime_k &= p_k - \frac{\eta}{\sqrt{v^\prime_k}+\epsilon}m^\prime_k
 \end{aligned}
 $$

## Class Constructor
Constructs a new ``AmsGrad`` optimizer.

|  Argument   |   Type    |         Description          |
|-------------|-----------|------------------------------|
|learning_rate|float=0.001|The learning rate $\eta$.     |
|beta1        |float=0.9  |First exponential decay rate. |
|beta2        |float=0.999|Second exponential decay rate.|
|epscut       |float=1e-07|Small epsilon cutoff.         |

### Examples
Simple AmsGrad optimizer.

```python
>>> from netket.optimizer import AmsGrad
>>> op = AmsGrad()

```



## Class Methods 
### reset
Member function resetting the internal state of the optimizer.


