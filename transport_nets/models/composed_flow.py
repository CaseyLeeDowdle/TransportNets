# will have to consider multiple dimensions in x2 for future
import tensorflow.compat.v2 as tf
from tqdm import trange
import tensorflow_probability as tfp
tfd = tfp.distributions

__all__ = ['ComposedFlow']

# In future probability better to just chain bijectors from
# f and f_hat, with standard normal as base distribution for
# the TransformedDistribution
class ComposedFlow():
    def __init__(self,f,f_hat,x2_obs,output_dim,opt):
        self.f = f
        self.f_hat = f_hat
        # x2_obs must be list of rank 1
        assert tf.rank(x2_obs) == 1
        self.x2_obs = x2_obs
        self.x2_dim = len(x2_obs)
        self.output_dim = output_dim
        assert (self.output_dim - self.x2_dim) > 0
        self.x1_dim = self.output_dim - self.x2_dim
        self.opt = opt
        # Don't give user for base distribution since standard normal is assumed
        # for objective function
        self.NormalDist = tfd.MultivariateNormalDiag(loc=tf.zeros(self.output_dim), 
                                                     scale_diag=tf.ones(self.output_dim))

    @tf.function
    def train_step(self,batch_size,sigma):
        z0 = self.NormalDist.sample(batch_size)
        with tf.GradientTape() as tape:
            z1 = self.f_hat(z0)
            loss = tf.reduce_mean(self.f_hat.log_prob(z1) - self.NormalDist.log_prob(z1)
                   + 1.0/(2.0*sigma**2)*tf.norm(self.f(z1)[:,self.x1_dim:]-self.x2_obs,axis=-1)**2) 
        gradients = tape.gradient(loss, self.f_hat.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.f_hat.trainable_variables))
        return loss
    
    def train_pre_conditioner(self,epochs,batch_size,sigma):
        t = trange(epochs)
        loss_history = []
        for epoch in t:
            loss = self.train_step(batch_size,sigma)
            t.set_description("loss: %0.3f " % loss.numpy())
            t.refresh()
            loss_history.append(loss.numpy())
        return loss_history
    
    @tf.function
    def sample(self,N):
        return self.f(self.f_hat.sample(N))