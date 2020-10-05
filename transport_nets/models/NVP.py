import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import numpy as np
import h5py
import os
import contextlib

tfd = tfp.distributions
tfb = tfp.bijectors

class NVP(tf.keras.models.Model):
    def __init__(self,output_dim=2,
                 neuron_list=[200,200],
                 num_masked=None,
                 num_layers=None,
                 masks = None, 
                 permutations = None, 
                 ref_dist = None,
                 load_model = None):
        super(NVP, self).__init__(name='NVP')
        if load_model:
            self.load_model_fun(load_model)
        else:
            self.output_dim = output_dim
            self.neuron_list = neuron_list
            self.num_masked = num_masked
            self.num_layers = num_layers
            self.masks = masks
            self.permutations = permutations
        self.ref_dist = ref_dist
        # better to check Nones using their type, since numpy evaluates
        # lists as an array of bools
        if type(self.num_layers) == type(None):
            if type(self.masks) != type(None):
                self.num_layers = len(self.masks)
            else:
                assert type(self.permutations) != type(None)
                self.num_layers = len(self.permutations)
        if type(self.permutations) == type(None):
            self.permutations = []
            permutations_given = False
        else:
            permutations_given = True
        self.bn_trainable_vars_gamma = []
        self.bn_trainable_vars_beta = []
        self.loss_fns = dict({'nll' : self.negative_log_likelihood})
        self.loss_fn_names = dict({self.negative_log_likelihood : 
                                   'Negative Log Likelihood'})
        # s() and t()
        self.shift_and_log_scale_fn = self.num_layers*[None]
        for i in range(self.num_layers):
                self.shift_and_log_scale_fn[i] = tfb.real_nvp_default_template(
                                      hidden_layers=self.neuron_list)

        bijectors = []
        # for floweach layer containing bijector, batch_norm, and permutation
        for i in range(self.num_layers):
            # default to fixed mask length defined by num_masked
            if type(self.masks) == type(None):
                bijectors.append(tfb.RealNVP(
                    num_masked = self.num_masked, 
                    shift_and_log_scale_fn =self.shift_and_log_scale_fn[i]))
            else:
                bijectors.append(tfb.RealNVP(
                    num_masked = self.masks[i], 
                    shift_and_log_scale_fn = self.shift_and_log_scale_fn[i]))
            if i < (self.num_layers - 1):
                if not permutations_given:
                    # default to random permutation each layer
                    perm_list = tf.random.shuffle(np.arange(0,self.output_dim))
                    perm_list = tf.cast(perm_list,tf.int32)
                    # want to store permutations for reproducible model
                    self.permutations.append(perm_list)
                    bijectors.append(tfb.Permute(permutation=perm_list))
                else:
                    # else append custom permutation from list
                    bijectors.append(tfb.Permute(permutation=self.permutations[i]))

                # batch norm
                bn_bijector = tfb.BatchNormalization()
                # need to do a forard pass to initialize training variables
                _ = bn_bijector(tf.random.normal([1,self.output_dim]))
                self.bn_trainable_vars_gamma.append(bn_bijector.trainable_variables[0])
                self.bn_trainable_vars_beta.append(bn_bijector.trainable_variables[1])
                bijectors.append(bn_bijector)


        # Chain layers together
        self.bijector_chain = tfb.Chain(bijectors=list(reversed(bijectors)))

        # Make flow
        if ref_dist == None:
            self.ref_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([self.output_dim],tf.float32))
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
    def sample(self, num_samples):
        return self.flow.sample(num_samples)
    @tf.function
    def transformed_log_prob(self, log_prob, x):
        return (self.bijector_chain.inverse_log_det_jacobian(x, event_ndims=self.output_dim) 
                    + log_prob(self.bijector_chain.inverse(x)))
    @tf.function
    def log_prob(self,x):
        return self.flow.log_prob(x)
    @tf.function
    def prob(self, x):
        return self.flow.prob(x)
    
    @tf.function
    def negative_log_likelihood(self, inputs):
        return -tf.reduce_mean(self.flow.log_prob(inputs))

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
            loss = self.loss_fn(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.add_metric(-tf.reduce_mean(self.flow.log_prob(x)), 
                        aggregation = 'mean', name="test")
        #loss_tracker.update_state(loss)
        return {self.loss_fn_names[self.loss_fn] : loss}
    
    def save_model(self,filename):
        with contextlib.suppress(FileNotFoundError):
            os.remove(filename)
        f = h5py.File(filename, 'w')
        model_group = f.create_group('model')
        model_dict = {'output_dim':self.output_dim,
                      'neuron_list':self.neuron_list,
                      'num_masked':self.num_masked,
                      'num_layers':self.num_layers,
                      'masks':self.masks,
                      'permutations':self.permutations}
        for key,val in model_dict.items():
            if type(val) is not type(None):
                model_group[key] = val
        f.close()
        
    def load_model_fun(self,file_name):
        assert os.path.isfile(file_name)
        attributes = ['output_dim',
                     'neuron_list',
                     'num_masked',
                     'num_layers',
                     'masks',
                     'permutations']
        f = h5py.File(file_name,'r')
        for key in attributes:
            if key in f['model']:
                setattr(self,key,f['model'].__getitem__(key)[()])
            else:
                setattr(self,key,None)
        f.close()

# simplified version of Tensorflow Probability template
def real_nvp_template(neuron_list,name=None):
    with tf.name_scope(name or 'real_nvp_template'):
          
        def _fn(x,output_units,**condition_kwargs):
            for neurons in neuron_list:
                x = tf1.layers.dense(x,neurons)
                x = tf.nn.relu(x)
            x = tf1.layers.dense(x,2*output_units)
            
            shift, logscale = tf.split(x, 2, axis=-1)
            return shift, logscale
    
    return tf1.make_template('real_nvp_template', _fn)