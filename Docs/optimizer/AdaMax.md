# AdaMax
AdaMax Optimizer.
 AdaMax is an adaptive stochastic gradient descent method,
 and a variant of [Adam](https://arxiv.org/pdf/1412.6980.pdf) based on the infinity norm.
 In contrast to the SGD, AdaMax offers the important advantage of being much
 less sensitive to the choice of the hyper-parameters (for example, the learning rate).

 Given a stochastic estimate of the gradient of the cost function ($$ G(\mathbf{p}) $$),
 AdaMax performs an update:

 $$
 p^\prime_k = p_k + \mathcal{S}_k,
 $$

 where $$ \mathcal{S}_k $$ implicitly depends on all the history of the optimization up to the current point.
 The NetKet naming convention of the parameters strictly follows the one introduced by the authors of AdaMax.
 For an in-depth description of this method, please refer to
 [Kingma, D., & Ba, J. (2015). Adam: a method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf)
 (Algorithm 2 therein).

## Class Constructor
Constructs a new ``AdaMax`` optimizer.

|Argument|   Type    |         Description          |
|--------|-----------|------------------------------|
|alpha   |float=0.001|The step size.                |
|beta1   |float=0.9  |First exponential decay rate. |
|beta2   |float=0.999|Second exponential decay rate.|
|epscut  |float=1e-07|Small epsilon cutoff.         |

### Examples
Simple AdaMax optimizer.

```python
>>> from netket.optimizer import AdaMax
>>> op = AdaMax()

```



## Class Methods 
### reset
Member function resetting the internal state of the optimizer.


