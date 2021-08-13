# Examples of autoregressive neural networks

In this folder there are examples of autoregressive neural networks with dense and convolutional layers.

Dense layers have a lot of parameters when the physical system is large. Convolutional layers utilize the physical system's translational symmetry and locality to reduce the number of parameters, and introduce a more meaningful metric of the parameter space to use with stochastic reconfiguration.

The fast autoregressive networks can be more efficient than the simple ones when the physical lattice and the network size are sufficiently large, such that the overhead of kernel launches is much smaller than the time for convolutions.
