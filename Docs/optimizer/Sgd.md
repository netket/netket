# Sgd
Simple Stochastic Gradient Descent Optimizer.
 [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 is one of the most popular optimizers in machine learning applications.
 Given a stochastic estimate of the gradient of the cost function ($$ G(\mathbf{p}) $$),
 it performs the update:

 $$
 p^\prime_k = p_k -\eta G_k(\mathbf{p}),
 $$

 where $$ \eta $$ is the so-called learning rate.
 NetKet also implements two extensions to the simple SGD,
 the first one is $$ L_2 $$ regularization,
 and the second one is the possibility to set a decay
 factor $$ \gamma \leq 1 $$ for the learning rate, such that
 at iteration $$ n $$ the learning rate is $$ \eta \gamma^n $$.

## Class Constructor
Constructs a new ``Sgd`` optimizer.

|  Argument   |  Type   |              Description              |
|-------------|---------|---------------------------------------|
|learning_rate|float    |The learning rate $$ \eta $$           |
|l2_reg       |float=0  |The amount of $$ L_2 $$ regularization.|
|decay_factor |float=1.0|The decay factor $$ \gamma $$.         |

### Examples
Simple SGD optimizer.

```python
>>> from netket.optimizer import Sgd
>>> op = Sgd(learning_rate=0.05)

```



## Class Methods 
### reset
Member function resetting the internal state of the optimizer.


