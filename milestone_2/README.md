Milestone 2
===========
In Milestone 2 we  Train and run a Gaussian Processes regression on the Boston Housing Dataset.

We then Evaluate and compare the predictions using the RBF kernel and the Matern Kernel ( which is a generalization of the RBF kernel ) via 10-fold cross-validation
using MSE as the error measure.

n_restarts_optimizer : int, optional (default: 0)

The number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood.
The first run of the optimizer is performed from the kernel’s initial parameters, 
the remaining ones (if any) from thetas sampled log-uniform randomly from the space of allowed theta-values. 
If greater than 0, all bounds must be finite. 

Note that n_restarts_optimizer == 0 implies that one run is performed( set to 9 to perform 10 runs).

normalize_y : boolean, optional (default: False)
Whether the target values y are normalized, i.e., the mean of the observed target values become zero. 
This parameter should be set to True if the target values’ mean is expected to differ considerable from zero. 
When enabled, the normalization effectively modifies the GP’s prior based on the data, which contradicts the likelihood principle; 
normalization is thus disabled per default.

random_state : int, RandomState instance or None, optional (default: None)
The generator used to initialize the centers. If int, random_state is the seed used by the random number generator; 
If RandomState instance, random_state is the random number generator; 
If None, the random number generator is the RandomState instance used by np.random. (seed is set to 2018 to get consistent results) 


In the results we find that the Matern kernel performs better than the RBF kernel.

Matern kernels is a generalization of the RBF and the absolute exponential kernel parameterized by an additional parameter nu. Important intermediate values are nu=1.5 (once differentiable functions) and nu=2.5 (twice differentiable functions) The twice differentiable property of this makes Matern kernel popular in machine learning.


Author:
Yuxiang Wang 
Ashwin Kumar

