from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import log
from math import ceil
import math
import numpy as np
from scipy import stats
import copy 


class KnnLearner(SupervisedLearner):

    def __init__(self):
      pass

    data = []
    bssf = 0

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
      #Below is the code for my experiment
      # print(len(self.data))

      # count = 0
      # percent = int(n_data.shape[0] * .005)
      # test_points = int(n_data.shape[0] * .3)
      # change = False
      # self.shuffle(n_data)
      # subset = n_data[0:percent,:]
      # itt = 0
      # while count < 6:
      #   itt += 1
      #   new_set = self.shuffle(n_data)

      #   if(change):
      #     subset = new_set[0:percent,:]
        
      #   points_holder = new_set[0:test_points,:]
      #   points = points_holder[:,:-1]
      #   correct = 0
      #   for i in range(len(points)):
      #     features = subset[:,:-1]
      #     features = points[i] - features
      #     features = features * features
      #     features = np.sum(features, axis=1)
      #     features = features ** 1/2
      #     final = np.argpartition(features, 5)
      #     final = final[:5]
      #     votes = new_set[final,:-1]
      #     unique, counts = np.unique(votes, return_counts=True)
      #     most = np.argmax(counts) 
      #     answer = unique[most]
      #     if answer == points_holder[i,-1]:
      #       correct += 1
      #   total = correct / len(points)
      #   print("Total",total)
      #   print("BSSF",self.bssf)
      #   print("count",count)
      #   print("\n")
      #   if total <= self.bssf:
      #     change = False
      #     count += 1
      #   else:
      #     change = True
      #     count = 0
      #     self.data = subset
      #     self.bssf = total


      # print("Itterations", itt)
      # print(len(self.data))
      return

    def euclidean_closest(self, points, k):
      features = self.data[:,:-1]
      features = points - features
      features = features * features
      features = np.sum(features, axis=1)
      features = features ** 1/2
      final = np.argpartition(features, k)
      final = final[:k]
      return final, features

    def HEOM(self, points, k):

      features = self.data[:,:-1]
      for i in range(features.shape[1]):
        unknown = np.where(features[:,i] > 10000000000)[0]
        for j in unknown:
          features[j][i] =  1

      nominal = []  
      continuous = []
      for i in range(features.shape[1]):
        if len(np.unique(features[:,i])) < 20:
          nominal.append(i)
        else:
          continuous.append(i)
      nominal = np.array(nominal)
      continuous = np.array(continuous)

      ranges = {}
      for i in continuous:
        max_val = np.amax(features[:,i])
        min_val = np.amin(features[:,i])
        range_val = max_val - min_val
        ranges[i] = range_val

      for i in range(len(points)):
        if points[i] > 10000000000:
          points[i] = 1
      
      for i in ranges:
        points[i] = points[i]/ranges[i]

      for i in range(len(points)):
        if i in nominal:
          features[:,i] = np.where(features[:,i] == points[i], 1, 0)
        else:
          features[:,i] = abs(features[:,i] - points[i])/ranges[i]

      features = points - features
      features = features * features
      features = np.sum(features, axis=1)
      features = features ** 1/2
      final = np.argpartition(features, k)
      final = final[:k]
      votes = self.data[final,-1]
      unique, counts = np.unique(votes, return_counts=True)
      most = np.argmax(counts) 
      return unique[most]

  

    def clean_data(self):
      features = self.data[:,:-1]
      for i in range(features.shape[1]):
        mean = np.mean(features[:,i])
        print(mean)
        unknown = np.where(features[:,i] > 100000000)[0]
        for j in unknown:
          print("wooooo")

    def knn(self, points, k, weighted=False):
      closest, distances = self.euclidean_closest(points, k)
      votes = self.data[closest,-1]
      unique, counts = np.unique(votes, return_counts=True)
      if weighted:
        weights = distances[closest]
        weights = weights**2
        for i in range(len(weights)):
          if weights[i] == 0:
            weights[i] = 0.0000000001
        weights = 1/weights
        weighted_votes = []
        for i in unique:
          temp = np.where(i == votes)[0]
          val = np.sum(weights[temp])
          weighted_votes.append((val,i))
        weighted_votes = np.array(weighted_votes)
        most = np.argmax(weighted_votes[:,0])
        return weighted_votes[most][1]
      else:
        most = np.argmax(counts) 
        return unique[most]
    
    def knn_reg(self, points, k, weighted=False):
      closest, distances = self.euclidean_closest(points, k)
      votes = self.data[closest,-1]
      if weighted:
        weights = distances[closest]
        weights = weights**2
        for i in range(len(weights)):
          if weights[i] == 0:
            weights[i] = 0.0000000001
        votes = votes/weights
        new_weights = 1/weights
        val = np.sum(votes)/np.sum(new_weights)
        return val
      else:
        val = np.sum(votes)/len(votes)
      return val

    def predict(self, features, labels):
      del labels[:]
      # answer = self.knn(features, 15, weighted=True)
      answer = self.HEOM(features, 27)
      # answer = self.knn_reg(features,15, weighted=True)
      labels.append(answer) 
