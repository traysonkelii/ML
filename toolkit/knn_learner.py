from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import log
from math import ceil
import numpy as np
from scipy import stats
import copy 


class KnnLearner(SupervisedLearner):

    def __init__(self):
      pass

    data = []

    def concat_data(self, features, labels):
      n_features = np.array(features)
      n_labels = np.array(labels)
      data = np.hstack((n_features, n_labels))
      return data

    def shuffle(self, n_data):
      np.random.shuffle(n_data)
      return n_data
    
   


    def train(self, features, labels):
      n_data = self.concat_data(features.data, labels.data)
      self.data = n_data
      return

    def euclidean(self, point_1, point_2):
      
      pass

    def knn(self, input):

      pass

    def predict(self, features, labels):
      del labels[:]
      """
      :type features: [float]
      :type labels: [float]
      """
      answer = 0  
      labels.append(answer) 
