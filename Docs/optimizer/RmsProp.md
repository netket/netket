# RmsProp
RMSProp is a well-known update algorithm proposed by Geoff Hinton
 in his Neural Networks course notes [Neural Networks course notes](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
 It corrects the problem with AdaGrad by using an exponentially weighted
 moving average over past squared gradients instead of a cumulative sum.
 After initializing the vector $$\mathbf{s}$$ to zero, $$s_k$$ and t
 he parameters $$p_k$$ are updated as

 $$
 \begin{align}
 s^\prime_k = \beta s_k + (1-\beta) G_k(\mathbf{p})^2 \\
 p^\prime_k = p_k - \frac{\eta}{\sqrt{s_k}+\epsilon} G_k(\mathbf{p})
 \end{align}
 $$

## Class Constructor
Constructs a new ``RmsProp`` optimizer.

|  Argument   |   Type    |        Description         |
|-------------|-----------|----------------------------|
|learning_rate|float=0.001|The learning rate $$ \eta $$|
|beta         |float=0.9  |Exponential decay rate.     |
|epscut       |float=1e-07|Small cutoff value.         |

### Examples
RmsProp optimizer.

```python
>>> from netket.optimizer import RmsProp
>>> op = RmsProp(learning_rate=0.02)

```



## Class Methods 
### reset
Member function resetting the internal state of the optimizer.


