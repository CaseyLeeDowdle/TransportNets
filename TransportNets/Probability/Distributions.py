from abc import abstractmethod
import numpy as np
import tensorflow as tf

class Distribution:
    """ Base class for defining distributions. """

    def __init__(self, dim):
      self.dim = dim

    @abstractmethod
    def Sample(self, N):
        """ Implemented by children to generate a matrix with N samples of the
            distribution.  Each row of the returned matrix contains a single
            sample.  The result is a numpy array.
        """
        return None


    @abstractmethod
    def Density(self, x):
        """ Evaluates the density of this distribution at one or more points.
            Each row in x is a point.  Returns an Nx1 numpy matrix containing
            the density values.
        """
        return None


class BananaDistribution(Distribution):

    def __init__(self, a, b, offset=None):
        super(BananaDistribution,self).__init__(2)

        if(offset is None):
            offset = np.zeros((2))

        assert(len(offset)==2)

        self.a = a
        self.b = b
        self.offset = offset


    def GaussToTarget(self,r):
        """ Transforms points in the Gaussian reference space to points in the
            Banana target space.  Each row of r is a point.
        """
        x1_base = self.a*r[:,0:1]
        x1 = x1_base + self.offset[0]
        x2 = self.offset[1] + r[:,1:2]/self.a + self.b*x1_base*x1_base
        return np.hstack([x1,x2])

    def TargetToGauss(self,x):
        """ Transforms points in the target space to the Gaussian reference space."""
        xdiff = (x[:,0:1] - self.offset[0])
        r1 = xdiff/self.a
        r2 = self.a*(x[:,1:2] - self.offset[1] - self.b*xdiff*xdiff)
        return np.hstack([r1,r2])

    def Sample(self,N):
        r = np.random.randn(N,2).astype('f')
        return self.GaussToTarget(r)

    def Density(self, x):
        r = self.TargetToGauss(x)
        return (0.5/np.pi) * np.exp(-0.5*np.sum(r*r,axis=1)).reshape(-1,1)


class MixtureDistribution(Distribution):

    def __init__(self, dists, weights):
        super(MixtureDistribution,self).__init__(dists[0].dim)

        self.numComps = len(dists)
        assert(len(weights)==self.numComps)

        self.dists = dists
        self.weights = weights / np.sum(weights) # make sure the weights sum to one

    def Sample(self,N):
        # Pick random components
        inds = np.random.choice(list(range(self.numComps)), N, p=self.weights)
        samps = np.zeros((N,self.dim),dtype='f')

        for i in range(N):
            samps[i,:] = self.dists[inds[i]].Sample(1)

        return samps

    def Density(self, x):
        allDens = np.dstack([dist.Density(x) for dist in self.dists])
        print(allDens.shape, self.weights.shape)
        return np.tensordot(allDens, self.weights,axes=((2),(0)))



def RotatedBanana(N,angle,a1=1.0,a2=0.5,a3=0.1,start_height=0):
    angle = np.radians(angle)
    r = tf.random.normal([N,2],mean=0.0,stddev=1.0)
    x1 = a1*r[:,:1]
    x2 = a2*r[:,:1]**2 + a3*r[:,1:2]
    x1Rot = np.cos(angle)*x1 - np.sin(angle)*x2
    x2Rot = np.sin(angle)*x1 + np.cos(angle)*x2
    return tf.concat([x1Rot,x2Rot + start_height],axis=1)
