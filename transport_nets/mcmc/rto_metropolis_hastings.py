import tensorflow as tf
import tensorflow_probability as tfp
import time
import numpy as np
from tqdm import trange

class RTO_MetropolisHastings():
    def __init__(self,num_params, y_obs, model, noise_std = None, S_obs = None):
        self.num_params = num_params
        self.y_obs = tf.reshape(y_obs, [1,-1])
        self.num_obs = tf.shape(self.y_obs)[-1].numpy()
        self.output_dim = self.num_params + self.num_obs
        self.model = model
        if S_obs == None:
            assert noise_std != None
            self.S_obs_inv_transpose = tf.transpose(tf.linalg.inv(noise_std*tf.eye(self.num_obs)))
        else:
            # makse sure S_obs is square matrix
            assert tf.shape(S_obs)[0] == self.num_obs & tf.shape(S_obs)[1] == self.num_obs
            self.S_obs_inv_transpose = tf.transpose(tf.linalg.inv(S_obs))
        
        vi = tf.random.normal([1,self.output_dim],stddev=0.1)
        opt_results = tfp.optimizer.bfgs_minimize(
                        self.H_loss_and_gradient, initial_position=vi, 
                        tolerance=1e-8)
        self.v_ref = opt_results.position
        jac_H = self.Jac_H(self.v_ref)[0,...]   
        Q,R = tf.linalg.qr(jac_H)
        self.Q_T = tf.transpose(Q)
                                                    
    def run(self,N_samps):
                                                    
        t0 = time.time()
        self.eta = tf.random.normal(shape=[N_samps,self.output_dim+self.num_obs])
        vi = tf.random.normal([N_samps,self.output_dim],stddev=0.1)
        v_prop_opt = tfp.optimizer.bfgs_minimize(
                self.v_prop_loss_and_gradient, initial_position=vi, tolerance=1e-8)
        v_prop = v_prop_opt.position
        w_v_prop = self.w(v_prop)
        v_samps = np.zeros([N_samps,self.output_dim])
        v_samps[0,:] = self.v_ref
        n_acc = 0
        for i in trange(1,N_samps):
            t = tf.random.uniform(shape=[1])
            v = tf.reshape(tf.constant(v_samps[i-1,:],dtype=tf.float32),[1,self.output_dim])
            if t < w_v_prop[i]/self.w(v):
                v_samps[i,:] = v_prop[i,:].numpy()
                n_acc += 1
            else:
                v_samps[i,:] = v_samps[i-1,:]
        t1 = time.time()
        self.ref_samples = v_samps
        target_samples = self.model(v_samps)                                         
        acc_rate = n_acc/N_samps
        return target_samples,acc_rate,t1-t0
                                                    
                                                    
    @tf.function
    def G(self,v):
        # defining matrix operations reverse of actual equation in order to allow for batch ops 
        return (self.model(v)[:,self.num_params:]-self.y_obs)@self.S_obs_inv_transpose
                                                    
    @tf.function
    def H(self,v):
        return tf.concat([v,self.G(v)],-1)
                                                    
    @tf.function
    def H_loss_and_gradient(self,v):
        return tfp.math.value_and_gradient(
            lambda v: 0.5*tf.norm(self.H(v),axis=-1)**2, v)
                                                    
    @tf.function
    def Jac_H(self,v):
        with tf.GradientTape() as g:
            g.watch(v)
            f = self.H(v)
        batch_jac = g.batch_jacobian(f,v)
        return batch_jac
    
    @tf.function
    def v_prop_loss_and_gradient(self,v):
        return tfp.math.value_and_gradient(
            lambda v: 0.5*tf.norm(self.Q_T@tf.reshape(self.H(v)-self.eta,
                                           [-1,self.output_dim+self.num_obs,1]),axis=[-2,-1])**2, v)
   
    @tf.function
    def w(self,v):
        A = tf.abs(tf.linalg.det(self.Q_T@self.Jac_H(v)))**(-1)
        x1 = -0.5*tf.norm(self.H(v),axis=-1)**2
        x2 = 0.5*tf.norm(self.Q_T@tf.reshape(self.H(v),[-1,self.output_dim+self.num_obs,1]),axis=[-2,-1])**2
        return A*tf.exp(x1+x2)