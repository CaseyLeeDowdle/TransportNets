import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tqdm import trange
from transport_nets.models import real_nvp_template
from transport_nets.models import MLP_ODE

tf.enable_v2_behavior()
tfd = tfp.distributions
tfb = tfp.bijectors

class StackedFlow(tf.keras.Model):
    def __init__(self,output_dim,
                 num_layers,
                 nvp_neurons,
                 masks,
                 permutations,
                 ffjord_neurons,
                 solver = tfp.math.ode.DormandPrince(atol=1e-5),
                 trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson):
        super(StackedFlow,self).__init__(name='stacked_flow')
        assert(len(masks)) == num_layers
        assert(len(permutations)) == num_layers
        
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.nvp_neurons = nvp_neurons
        self.masks = masks
        self.permutations=permutations
        self.ffjord_neurons = ffjord_neurons
        self.ode_solve_fn = solver.solve
        self.trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson
        
        bijectors = []
        self.shift_scale_fns = []
        self.state_time_fns = []
        self.bn_bijectors = []
        self.bn_trainable_variables = []
        
        for i in range(self.num_layers):
            
            # permutation bijector
            permute_bijector = tfb.Permute(permutation=self.permutations[i])
            # nvp bijector
            coupling_nn = real_nvp_template(neuron_list=nvp_neurons)
            nvp_bijector = tfb.RealNVP(num_masked = self.masks[i], 
                                    shift_and_log_scale_fn = coupling_nn)
            self.shift_scale_fns.append(coupling_nn)
            # batch normalization bijector
            bn_bijector1 = tfb.BatchNormalization()
            _ = bn_bijector1(tf.random.normal([1,self.output_dim]))
            self.bn_bijectors.append(bn_bijector1)
            
            # ffjord bijector
            mlp_ode = MLP_ODE(output_dim=self.output_dim,neuron_list=ffjord_neurons)
            ffjord_bijector = tfb.FFJORD(state_time_derivative_fn=mlp_ode,
                                    ode_solve_fn=self.ode_solve_fn,
                                    trace_augmentation_fn=self.trace_augmentation_fn)
            self.state_time_fns.append(mlp_ode)
            
            bn_bijector2 = tfb.BatchNormalization()
            _ = bn_bijector2(tf.random.normal([1,self.output_dim]))
            self.bn_bijectors.append(bn_bijector2)
            
            bijectors += [
                          permute_bijector,
                          nvp_bijector,
                          bn_bijector1,
                          ffjord_bijector,
                          bn_bijector2]
        
        self.chained_bijectors = tfb.Chain(bijectors[::-1])
        base_loc = tf.zeros([self.output_dim],dtype=tf.float32)
        base_sigma = tf.ones([self.output_dim],dtype=tf.float32)
        self.base_distribution = tfd.MultivariateNormalDiag(base_loc, base_sigma)
        self.flow = tfd.TransformedDistribution(
                            distribution=self.base_distribution, 
                            bijector=self.chained_bijectors)
        
    @tf.function   
    def call(self,inputs):
        return self.flow.bijector.forward(inputs)
    
    @tf.function   
    def inverse(self,inputs):
        return self.flow.bijector.inverse(inputs)
    
    @tf.function   
    def sample(self,N):
        return self.flow.sample(N)
    
    @tf.function
    def prob(self, x):
        return self.flow.prob(x)

    @tf.function
    def log_prob(self, x):
        return self.flow.log_prob(x)
    
    @tf.function
    def negative_log_likelihood(self, inputs):
        return -tf.reduce_mean(self.flow.log_prob(inputs))
    
    def compile(self, optimizer):
        super(StackedFlow, self).compile()
        self.optimizer = optimizer
    
    def training_mode(self,training):
        for bij in self.bn_bijectors:
            bij.batchnorm.trainable = training
                
    @tf.function
    def train_step(self,target_sample):
        with tf.GradientTape() as tape:
            loss = self.negative_log_likelihood(target_sample)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
    def fit(self,dataset,num_epochs):
        t = trange(num_epochs)
        loss_history = []
        for epoch in t:
            for batch in dataset:
                loss = self.train_step(batch)
                t.set_description("loss: %0.3f " % loss.numpy())
                t.refresh()
                loss_history.append(loss.numpy())
        return loss_history
    