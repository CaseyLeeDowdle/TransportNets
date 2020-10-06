import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from transport_nets.bijectors import BananaMap

__all__ = ['BananaFlow']


def BananaFlow(param_tuple):
    """TransformedDistribution with 2-D standard normal base
    distribution and BanananaMap for bijector"""
    bananaMap = BananaMap(param_tuple)
    bFlow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(
            loc=tf.zeros([2]),scale_diag=tf.ones([2])),
            bijector=bananaMap)
    return bFlow