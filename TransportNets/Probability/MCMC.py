import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time

# Next, the Metropolis hastings algorithm
def MH2d(model, y_star = 0.0, niters=10000,b=2,init_val=0.0):

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

class RTO_MH:
    # try for a single sample m ~ p(m|y)
    # n =  dim of params
    # m = dim of observations
    # n_samples is the number of samples to draw
    # S = covariance matrix for observations, must be square
    def __init__(self, model, n, m, n_samples, S = None):

        self.n_samples = n_samples
        self.model = model
        self.m = m
        self.n = n
        self.S = S
        self.compile();

    def compile(self):
        latent_vector = tf.random.normal([1,self.n],mean=0.0,stddev=1.0,dtype=tf.float32)
        self.joint_sample = self.model(latent_vector)
        self.y = self.joint_sample[:,(self.n-self.m):]

        if (S != None):
            self.S = tf.eye(self.m)
        else:
            assert S.shape[0] == S.shape[1]

        self.S_obs_inv_transpose  = tf.transpose(tf.linalg.inv(self.S))

        vi = tf.random.normal([1,self.n],stddev=0.1)
        opt_results = tfp.optimizer.bfgs_minimize(
            self.H_loss_and_gradient, initial_position=vi, tolerance=1e-6)
        v_ref = opt_results.position

        jac_H = self.Jac_H(v_ref)[0,...]
        Q,R = tf.linalg.qr(jac_H)
        self.Q_T = tf.transpose(Q)

        vi = tf.random.normal([self.n_samples,self.n],stddev=0.1)
        v_prop_opt = tfp.optimizer.bfgs_minimize(
            self.v_prop_loss_and_gradient, initial_position=vi, tolerance=1e-6)
        self.v_prop = v_prop_opt.position

        self.w_v_prop = self.w(self.v_prop)
        self.v_samps = np.zeros([self.n_samples,self.n])
        self.v_samps[0,:] = v_ref


    def run(self):
        self.n_acc = 0
        for i in range(1, self.n_samples):
            t = tf.random.uniform(shape=[1])
            v = tf.reshape(tf.constant(self.v_samps[i-1,:],dtype=tf.float32),[1,self.n])
            if t < self.w_v_prop[i]/self.w(v):
                self.v_samps[i,:] = self.v_prop[i,:].numpy()
                self.n_acc += 1
            else:
                self.v_samps[i,:] = self.v_samps[i-1,:]

        print('Acceptance Rate:', self.n_acc/self.n_samples)
        return self.n_acc, self.model(self.v_samps)


    @tf.function
    def G(self, v):
        # defining matrix operations reverse of actual equation in order to allow for batch ops
        return (self.model(v)[:,(self.n-self.m):]-self.y)@self.S_obs_inv_transpose
    @tf.function
    def H(self, v):
        return tf.concat([v,self.G(v)],-1)
    @tf.function
    def H_loss_and_gradient(self, v):
        return tfp.math.value_and_gradient(
            lambda v: 0.5*tf.norm(self.H(v),axis=-1)**2, v)

    # going to use tf.batch_jacobian
    # jacobian of H wrt v_ref
    @tf.function
    def Jac_H(self, v):
        with tf.GradientTape() as g:
            g.watch(v)
            f = self.H(v)
        batch_jac = g.batch_jacobian(f,v)
        return batch_jac
    @tf.function
    def v_prop_loss_and_gradient(self, v):
        eta = tf.random.normal(shape=[self.n_samples,self.n+self.m])
        return tfp.math.value_and_gradient(
            lambda v: 0.5*tf.norm(self.Q_T@tf.reshape(self.H(v)-eta,[-1,self.m+self.n,1]),axis=[-2,-1])**2, v)

    @tf.function
    def w(self, v):
        A = tf.abs(tf.linalg.det(self.Q_T@self.Jac_H(v)))**(-1)
        x1 = -0.5*tf.norm(self.H(v),axis=-1)**2
        x2 = 0.5*tf.norm(self.Q_T@tf.reshape(self.H(v),[-1,self.m+self.n,1]),axis=[-2,-1])**2
        return A*tf.exp(x1+x2)
