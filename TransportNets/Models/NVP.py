import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors

class NVP(tf.keras.models.Model):

    def __init__(self,num_masked=1,output_dim=2,num_layers=4,neuron_list=[200,200], ref_dist = None):
        super(NVP, self).__init__(name='NVP')

        # member variables
        self.output_dim = output_dim
        self.bn_trainable_vars_gamma = []
        self.bn_trainable_vars_beta = []
        self.loss_fns = dict({'nll' : self.negative_log_likelihood})
        self.loss_fn_names = dict({self.negative_log_likelihood : 'Negative Log Likelihood'})

        # s() and t()
        self.shift_and_log_scale_fn = num_layers*[None]
        for i in range(num_layers):
                self.shift_and_log_scale_fn[i] = tfb.real_nvp_default_template(
                                      hidden_layers=neuron_list)

        bijectors = []
        # for floweach layer containing bijector, batch_norm, and permutation
        for i in range(num_layers):

            bijectors.append(tfb.RealNVP(num_masked = num_masked, shift_and_log_scale_fn = self.shift_and_log_scale_fn[i]))
            if (i < (num_layers - 1)):

                # random permutation each layer
                perm_list = tf.random.shuffle(np.arange(0,output_dim))
                perm_list = tf.cast(perm_list,tf.int32)
                bijectors.append(tfb.Permute(permutation=perm_list))

                # batch norm
                bn_bijector = tfb.BatchNormalization()
                # need to do a forard pass to initialize training variables
                _ = bn_bijector(tf.random.normal([1,output_dim]))
                self.bn_trainable_vars_gamma.append(bn_bijector.trainable_variables[0])
                self.bn_trainable_vars_beta.append(bn_bijector.trainable_variables[1])
                bijectors.append(bn_bijector)


        # Chain layers together
        self.bijector_chain = tfb.Chain(bijectors=list(reversed(bijectors)))

        # Make flow
        if (ref_dist == None):
            self.ref_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([output_dim],tf.float32))
        else:
            self.ref_dist = ref_dist

        self.flow = tfd.TransformedDistribution(
            distribution=self.ref_dist,
            bijector=self.bijector_chain,
        )

    #getters and functions
    @tf.function
    def call(self, inputs):
        return self.bijector_chain.forward(inputs)
    @tf.function
    def inverse(self, inputs):
        return self.bijector_chain.inverse(inputs)
    @tf.function
    def getFlow(self, num_samples):
        return self.flow.sample(num_samples)
    @tf.function
    def transformed_log_prob(self, log_prob, x):
        return (self.bijector_chain.inverse_log_det_jacobian(x, event_ndims=self.output_dim) + log_prob(self.bijector_chain.inverse(x)))
    @tf.function
    def log_prob(self, x):
        return self.flow.log_prob(x)
    @tf.function
    def prob(self, x):
        return self.flow.prob(x)

    def batch_norm_mode(self,training):
        #training is bool
        for i,bij in enumerate(self.bijector_chain.bijectors):
            if bij.name == 'batch_normalization':
                self.bijector_chain.bijectors[i].batchnorm.trainable = training

    def compile(self, optimizer, loss_fn_name = 'nll'):
        super(NVP, self).compile()
        self.optimizer = optimizer
        self.loss_fn = self.loss_fns[loss_fn_name]

    @tf.function
    def train_step(self, x):
        # x is input from target distribution
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = self.loss_fn(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.add_metric(-tf.reduce_mean(self.flow.log_prob(x)), aggregation = 'mean', name="test")

        #loss_tracker.update_state(loss)
        return {self.loss_fn_names[self.loss_fn] : loss}

    @tf.function
    def negative_log_likelihood(self, input):
        return -tf.reduce_mean(self.flow.log_prob(input))
