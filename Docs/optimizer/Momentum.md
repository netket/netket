# Momentum
Momentum-based Optimizer.
 The momentum update incorporates an exponentially weighted moving average
 over previous gradients to speed up descent
 [Qian, N. (1999)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf).
 The momentum vector $$\mathbf{m}$$ is initialized to zero.
 Given a stochastic estimate of the gradient of the cost function
 $$G(\mathbf{p})$$, the updates for the parameter $$p_k$$ and
 corresponding component of the momentum $$m_k$$ are

 $$
 \begin{aligned}
 m^\prime_k &= \beta m_k + (1-\beta)G_k(\mathbf{p})\\
 p^\prime_k &= \eta m^\prime_k
 \end{aligned}
 $$

## Class Constructor
Constructs a new ``Momentum`` optimizer.

|  Argument   |   Type    |                    Description                     |
|-------------|-----------|----------------------------------------------------|
|learning_rate|float=0.001|The learning rate $$ \eta $$                        |
|beta         |float=0.9  |Momentum exponential decay rate, should be in [0,1].|

### Examples
Momentum optimizer.

```python
>>> from netket.optimizer import Momentum
>>> op = Momentum(learning_rate=0.01)

```



## Class Methods 
### reset
Member function resetting the internal state of the optimizer.


