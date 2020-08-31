import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange

tfb = tfp.bijectors

def MetropolisHastings(theta,b,niters,log_prob_fn):
    """ Metropolis-Hastings MCMC random walk with Gaussian proposal
    Args:
        theta: Initial value for Markov Chain. Shape must be rank 1 tensor.
        
        b: Variance of proposal distribution. Can be either scalar or rank 1
        tensor.
        
        niters: Number of interations.
        
        log_prob_fn: Function that accepts rank one tensor as input and returns
        the log probability of the target distribution at the input.
        
    Returns:
        The acceptance rate (scalar) and samples (rank 2 tensor) in shape
        [niters+1,dim] where dim is the dimension of the target distribution."""
        
        
    assert tf.rank(theta) == 1
    assert (tf.rank(b) == 0) or (tf.size(b) == tf.size(init_val))
    n_param = tf.size(theta)
    samples = tf.TensorArray(tf.float32,size=int(niters+1), dynamic_size=False)
    samples = samples.write(0,theta)
    naccept= 0
    for i in trange(1,niters+1):
        theta_p = theta + tf.random.normal([n_param])*b
        rho = min(1, tf.math.exp(log_prob_fn(theta_p)-log_prob_fn(theta)))
        u = tf.random.uniform([1])
        if u < rho:
            naccept += 1
            theta = theta_p
        samples = samples.write(i,theta)
    acceptance_rate = naccept/niters
    
    return acceptance_rate, samples.stack()

class model_log_prob():
    """Flexible way to create a function that returns the 
    log probability of a model."""
    def __init__(self,model,y_given=None,permute=None):
        """
        Args:
            model: a model that has a method called log_prob that
            can return the log probability of an input. Input must
            be in the form of rank 1 tensor. 
            
            y_given: A rank 1 tensor stacked onto the input in
            order to calculate the joint log probability of the input
            and y_given.
            
            permute: A permutation bijector to permute the inputs
            before calculating the log probability. Useful if the 
            training data of a model is in different order of the 
            conditional probability you want to sample."""
        
        if y_given != None:
            assert tf.rank(y_given) == 1
        self.model = model
        if permute == None:
            self.permute = tfb.Identity()
        else:
            self.permute = permute
            
        if y_given == None:
            def log_prob(x):
                x = self.permute.forward(x)
                x = tf.reshape(x,[1,-1])
                return self.model.log_prob(x)
            self.log_prob = log_prob
        else:
            self.y_given = y_given
            def log_prob(x):
                xy = tf.stack([x,self.y_given],axis=1)
                xy = self.permute.forward(xy)
                return self.model.log_prob(xy)
            self.log_prob = log_prob
    @tf.function
    def __call__(self,x):
        return self.log_prob(x)