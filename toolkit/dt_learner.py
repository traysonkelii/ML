from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import log
from math import ceil
import numpy as np
from scipy import stats
import copy 
class DecisionTreeLearner(SupervisedLearner):

    def __init__(self):
      self.path = self.Node()
      self.node_num = 1
      pass

    depth = 0
    bssf = 0
    class Node:
      def __init__(self):
        self.children = []
        self.interest = -1
        self.value = -1
        self.stop = -1
        self.data = -1
        self.pruned = 0

    def concat_data(self, features, labels):
      n_features = np.array(features)
      n_labels = np.array(labels)
      data = np.hstack((n_features, n_labels))
      return data

    def shuffle(self, n_data):
      np.random.shuffle(n_data)
      return n_data

    def get_info(self, data, const=[]):
      total = 0
      for i in data:
        total += -i*log(i,2)
      return total
    
    def split(self, total):
      attributes = total[:,:-1]
      all_info = []
      all_children = []
      interest = []
      for i in range(attributes.shape[1]):
        interest.append(i)
        child = {}
        att = np.unique(attributes[:,i])
        info = 0
        for j in att:
          branch = np.where(j == attributes[:,i])
          const = len(branch[0])/attributes.shape[0]
          new_rows = []

          for k in branch[0]:
            new_row = total[k,:]
            new_rows.append(new_row)

          new_rows = np.array(new_rows)
          child_vals = np.delete(new_rows, i, axis=1)
          vals = np.unique(new_rows[:,-1])
          log_vals = []
          if np.isinf(j):
              child[2] = child_vals
          else:
            child[int(j)] = child_vals

          for k in vals:
            split = np.where(k == new_rows[:,-1])
            log_vals.append(len(split[0])/new_rows.shape[0])

          sub_info = 0

          for k in log_vals:
            sub_info += -k*log(k,2)

          sub_info *= const
          info += sub_info

        all_info.append(info)
        all_children.append(child)

      interest = np.array(interest)
      return all_info, all_children, interest
    
    def build_tree(self, node):
      self.depth += 1
      info, children, interest = self.split(node.data)
      selected = np.argmin(info)
      children = children[selected]
      targets = node.data[:,-1]
      most = stats.mode(targets, axis=None)
      node.interest = interest[selected]
      node.value = most[0]

      if  info[selected] == 0:
        for i in children:
          node_val = 0
          if children[i].shape[1] > 1:
            node_val = children[i][:,-1][0]
          else:
            node_val = children[i][0][0]
          leaf_node = self.Node()
          self.node_num += 1
          leaf_node.value = node_val
          leaf_node.interest = i
          leaf_node.stop = 1
          node.children.append(leaf_node)
        pass

      else:
        for i in children:
          new_node = self.Node()
          self.node_num += 1
          new_node.data = children[i]
          node.children.append(new_node)
          self.build_tree(new_node)
      pass

    def prune(self, data, tree):
      if(tree.stop == 1):
        pass
      elif(tree.pruned == 0):
        tree.pruned = 1
        my_copy = copy.deepcopy(tree)
        self.guess(data, my_copy)
        for children in tree.children:
          self.prune(data, children)
        pass
      else:
        pass
      
    def guess(self, data, tree):
      features = data[:,:-1]
      total = data.shape[0]
      right = 0
      for i in range(features.shape[0]):
        answer = self.find(tree, features[i,:])
        if answer == int(data[i][-1]):
          right += 1
      accuracy = right/total
      if accuracy > self.bssf:
        self.path = tree
        self.bssf = accuracy

    def train(self, features, labels):
      n_data = self.concat_data(features.data, labels.data)
      n_data = self.shuffle(n_data)
      vs_num = int(n_data.shape[0] * .2)
      vs = n_data[0:vs_num,:]
      ts = n_data[vs_num:,:]
  
      self.path.data = ts
      self.build_tree(self.path)
      self.depth = ceil(log(self.depth,2))
      print(self.node_num)
      print(self.depth)
      # self.prune(vs, self.path)
      
      return
    

    def find(self,node, features):
        if node.stop == 1:
          return node.value
        elif len(features) == 1:
          try:
            row = np.where(features[0] == node.data[:,0])[0][0]
          except:
            return int(node.value)
          
          return node.data[row][-1]
        else:
          if (np.isinf(features[int(node.interest)])):
            child_index = 2
          else:
            child_index = int(features[int(node.interest)])
          features = np.delete(features,node.interest)
          if child_index >= len(node.children):
            child_index = len(node.children)-1
          node = node.children[child_index]
          return self.find(node, features)

    def predict(self, features, labels):
      del labels[:]
      """
      :type features: [float]
      :type labels: [float]
      """
      features = np.array(features)
      answer = self.find(self.path, features)    
      labels.append(answer) 
