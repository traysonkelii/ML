from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
from toolkit.perceptron_learner import PerceptronLearner

class IrisLearner(PerceptronLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    labels = []

    def __init__(self):
      self.one_two_w = []
      self.zero_one_w = []
      pass

    def normalize(self,v):
      norm = np.linalg.norm(v)
      if norm == 0: 
        return v
      return v / norm

    def iris(self, n_data):

      one_two_data = n_data[np.where(n_data[:,5] != 0)] # 1 vs 2
      zero_one_data = n_data[np.where(n_data[:,5] != 2)] # 0 vs 1

      self.one_two_w = self.normalize(self.cycle(one_two_data)) # finds 1 or 2
      self.zero_one_w = self.normalize(self.cycle(zero_one_data)) # finds 0 or 1

    def train(self, features, labels):
        n_data = self.clean_data(features.data, labels.data)
        self.iris(n_data)
        """
        :type features: Matrix
        :type labels: Matrix
        """

        return

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        f = np.array(features)
        f = np.append(f,[1])
        one_two = np.around(np.dot(f,self.one_two_w), decimals=2)
        zero_one = np.around(np.dot(f,self.zero_one_w), decimals=2)

        if zero_one < 0:
          labels += [0]
        elif one_two > 9:
          labels += [2]
        else:
          labels += [1]
