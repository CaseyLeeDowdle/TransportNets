import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'BananaMap',
    'BananaFlow'
]

class BananaMap(tfb.Bijector):
    """Pushes Gaussian reference distribution to Banana target distribution.
    The non-rotated forward BananaMap T: R -> X, R=[r1,r2], X=[x1,x2]
    is defined to be
    
    x1 = a1*r1
    x2 = a2*r1^2 + a3*r2
    
    where a1,a2,a3 are real-valued scalars."""
    
    
    def __init__(self,param_tuple,validate_args=False,name='BananaMap'):
        """Creates BananaMap bijector.
        
        Args:
            param_tuple: Python tuple in form (a1,a2,a3,theta) where
            a1,a2,a3 are scalars of the BananaMap, T: R -> X defined in the 
            class docstring and theta is a counter-clockwise rotation of the 
            Banana distribution. For example, for rotation matrix M and forward
            BananaMap T,
            M o T(r), 
            T applies forward BananaMap to Gaussian distribution and M then 
            rotates the Banana distribution."""
        super(BananaMap, self).__init__(
         validate_args=validate_args,
         forward_min_event_ndims=1,
         name=name)
        
        self.a1,self.a2,self.a3,self.theta = param_tuple
        
    def _forward(self, r):
        x1 = self.a1*r[:,:1]
        x2 = self.a2*r[:,:1]**2 + self.a3*r[:,1:2]
        x1Rot = tf.cos(self.theta)*x1 - tf.sin(self.theta)*x2 #radians
        x2Rot = tf.sin(self.theta)*x1 + tf.cos(self.theta)*x2
        
        return tf.concat([x1Rot,x2Rot],axis=1)
    
    def _inverse(self, x):
        
        x1Rot = tf.cos(self.theta)*x[:,:1] + tf.sin(self.theta)*x[:,1:2]
        x2Rot = -tf.sin(self.theta)*x[:,:1] + tf.cos(self.theta)*x[:,1:2]
        r1 = (1./self.a1)*x1Rot
        r2 = (1.0/self.a3)*(x2Rot-self.a2*r1**2)
        
        return tf.concat([r1,r2],axis=1)
    
    def _inverse_log_det_jacobian(self,x):
        
        return -self._forward_log_det_jacobian(self._inverse(x))
        
    def _forward_log_det_jacobian(self,r):
        
        return tf.math.log(tf.abs(self.a1*self.a3))*tf.ones([r.shape[0]])
    
def BananaFlow(param_tuple):
    """TransformedDistribution with 2-D standard normal base
    distribution and BanananaMap for bijector"""
    bananaMap = BananaMap(param_tuple)
    bFlow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(
            loc=tf.zeros([2]),scale_diag=tf.ones([2])),
            bijector=bananaMap)
    return bFlow