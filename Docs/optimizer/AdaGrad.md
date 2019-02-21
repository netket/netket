# AdaGrad
AdaGrad Optimizer.
 In many cases, in Sgd the learning rate $$\eta$$ should
 decay as a function of training iteration to prevent overshooting
 as the optimum is approached. AdaGrad is an adaptive learning
 rate algorithm that automatically scales the learning rate with a sum
 over past gradients. The vector $$\mathbf{g}$$ is initialized to zero.
 Given a stochastic estimate of the gradient of the cost function $$G(\mathbf{p})$$,
 the updates for $$g_k$$ and the parameter $$p_k$$ are

 $$
 \begin{aligned}
 g^\prime_k &= g_k + G_k(\mathbf{p})^2\\
 p^\prime_k &= p_k - \frac{\eta}{\sqrt{g_k + \epsilon}}G_k(\mathbf{p})
 \end{aligned}
 $$

 AdaGrad has been shown to perform particularly well when
 the gradients are sparse, but the learning rate may become too small
 after many updates because the sum over the squares of past gradients is cumulative.

## Class Constructor
Constructs a new ``AdaGrad`` optimizer.

|  Argument   |   Type    |       Description        |
|-------------|-----------|--------------------------|
|learning_rate|float=0.001|Learning rate $$\eta$$.   |
|epscut       |float=1e-07|Small $$\epsilon$$ cutoff.|

### Examples
Simple AdaDelta optimizer.

```python
>>> from netket.optimizer import AdaGrad
>>> op = AdaGrad()

```



## Class Methods 
### reset
Member function resetting the internal state of the optimizer.


