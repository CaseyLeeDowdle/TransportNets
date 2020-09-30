import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as layers
from tqdm import trange
tf.enable_v2_behavior()
tfb = tfp.bijectors
tfd = tfp.distributions

# Possibly need to write MLP_ODE as a function and wrap in tensorflow make_template
# in order to not duplicate variables 
# can still import MLP_ODE as is then define transformed distribution in jupyter notebook
# or use a function that returns the necessary transformed_distribution for wanted
# ouput_dim, num_bijectors, neuron_list

# should also compare difference between keras and directly optimizing on flow in notebook
# for the realNVP + FFJORD block triangular flow 

class MLP_ODE(tf.keras.Model):
    
    def __init__(self,output_dim, neuron_list,  name='mlp_ode'):
        super(MLP_ODE,self).__init__(name=name)
        self._neuron_list = neuron_list
        self._output_dim = output_dim
        self._modules = []
        for neurons in self._neuron_list:
            self._modules.append(layers.Dense(neurons))
            self._modules.append(layers.Activation('tanh'))
        self._modules.append(layers.Dense(self._output_dim))
        self._model = tf.keras.Sequential(self._modules)
        
    @tf.function     
    def call(self, t, inputs):
        inputs = tf.concat([tf.broadcast_to(t, inputs.shape), inputs], -1)
        return self._model(inputs)  

class FFJORD(tf.keras.Model):
    """Stacked FFJORD bijectors with Gaussian base distribution"""
    def __init__(self,output_dim,num_bijectors,neuron_list,
                 solver=tfp.math.ode.DormandPrince(atol=1e-5),
                 trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact
                 ):
        super(FFJORD, self).__init__(name='FFJORD')
        self.output_dim = output_dim
        self.num_bijectors = num_bijectors
        self.neuron_list = neuron_list
        self.ode_solve_fn = solver.solve
        self.trace_augmentation_fn = trace_augmentation_fn
        
        self.loss_fns = dict({'nll' : self.negative_log_likelihood})
        self.loss_fn_names = dict({self.negative_log_likelihood : 'Negative Log Likelihood'})
        
        bijectors = []
        for _ in range(self.num_bijectors):
            next_ffjord = tfb.FFJORD(
                      state_time_derivative_fn=MLP_ODE(self.output_dim,self.neuron_list),
                      ode_solve_fn=self.ode_solve_fn,
                      trace_augmentation_fn=self.trace_augmentation_fn)
            bijectors.append(next_ffjord)
    
        self.bijector_chain = tfb.Chain(bijectors[::-1])

        self.ref_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([self.output_dim],tf.float32))

        self.flow = tfd.TransformedDistribution(
                        distribution=self.ref_dist,
                        bijector=self.bijector_chain)
        


    @tf.function
    def train_step(self,target_sample,flow):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(target_sample)
        gradients = tape.gradient(loss,flow.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,flow.trainable_variables))
        return loss

    @tf.function
    def __call__(self, inputs):
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
    
    def compile(self, optimizer, loss_fn_name = 'nll'):
        super(FFJORD, self).compile()
        self.optimizer = optimizer
        self.loss_fn = self.loss_fns[loss_fn_name]

    def fit_custom(self,dataset,num_epochs):
        t = trange(num_epochs)
        loss_history = []
        for epoch in t:
            for batch in dataset:
                loss = self.train_step(batch,self.flow)
                t.set_description("loss: %0.3f " % loss.numpy())
                t.refresh()
            loss_history.append(loss.numpy())
        return loss_history