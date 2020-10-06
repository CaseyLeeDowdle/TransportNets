from abc import abstractmethod
import numpy as np
import tensorflow as tf

__all__ = ['MixtureDistribution']

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