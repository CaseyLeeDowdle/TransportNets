import tensorflow as tf
import numpy as np
import time

# Next, the Metropolis hastings algorithm
def MH2d( model, y_star = 0.0, niters=10000,b=2,init_val=0.0):

    # First, we define the function that gives the log prob at a point in target space
    def log_c_target(x1, x2 = y_star):
        x = tf.constant([x1,x2],shape=[1,2])
        #log of probability density at that point
        return model.log_prob(x)

    # niters: number of interations of MCMC
    # b: variance of the proposal Gaussian density for X1
    naccept=0
    theta = init_val # initial value for the Markov chain

    samples = np.zeros(niters+1)
    samples[0] = theta
    t0 = time.time()
    for i in range(niters):
        theta_p = theta + np.random.randn()*b;
        rho = min(1, tf.math.exp(log_c_target(theta_p)-log_c_target(theta)))
        u = np.random.uniform()
        if u < rho:
            naccept += 1
            theta = theta_p
        samples[i+1] = theta
        if i % 1000 == 0:
            t1 = time.time()
            print('it:',i,'time:',t1-t0)
            t0 = t1
    acceptance_rate = naccept/niters

    return acceptance_rate, samples
