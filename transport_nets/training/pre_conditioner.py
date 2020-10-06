# will have to consider multiple dimensions in x2 for future

class PreConditioner():
    def __init__(self,f,f_hat,x2_obs,opt)
NormalDist = tfd.MultivariateNormalDiag(loc=[0.0,0.0], scale_diag=[1.0,1.0]) #standard normal distribution 
@tf.function
def pre_cond_train(f,f_hat,x2_obs,sigma,batch_size,opt):
    z0 = NormalDist.sample(batch_size)
    with tf.GradientTape() as tape:
        z1 = f_hat(z0)
        loss = tf.reduce_mean(f_hat.log_prob(z1) - NormalDist.log_prob(z1)
               + 1.0/(2.0*sigma**2)*tf.norm(f(z1)[:,1:2]-x2_obs,axis=-1)**2) 
    gradients = tape.gradient(loss, f_hat.trainable_variables)
    opt.apply_gradients(zip(gradients, f_hat.trainable_variables))
    return loss