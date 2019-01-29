from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np

class PerceptronLearner(SupervisedLearner):
    """
    Basic perceptron learning algorithm
    Trayson Keli'i
    """

    labels = []
    weights = []

    def __init__(self):
        pass

    def clean_data(self, features, labels):
        n_features = np.array(features)
        ones = np.ones((n_features.shape[0],1))
        n_labels = np.array(labels)
        training = np.hstack((n_features,ones))
        data = np.hstack((training, n_labels))
        return data

    def get_output(self, data, weights):
        value = np.dot(data,weights)
        if value[0] > 0:
            return 1
        else:
            return 0

    def learning(self, inputs, rate, current_weights, target, output):
        for i in range(len(inputs)):
            value = inputs[i] * (target - output) * rate
            inputs[i] = value
        for i in range(len(current_weights)):
            current_weights[i][0] = current_weights[i][0] + inputs[i]
        return current_weights

    def shuffle(self, n_data):
        np.random.shuffle(n_data)
        return n_data

    def nominal(self, features, labels):

    def continuous(self, features, labels):

    def train(self, features, labels):

        mylist = np.array(labels.data)
        choices = set(mylist.flat)
        print(choices)
        # print(labels.print())
        # print(labels.rows)
        n_data = self.clean_data(features.data, labels.data)
        n_weights = np.zeros((n_data.shape[1]-1,1))
        learning_rate = .1
        count = 0
        old_acc = 100
        epoch = 0

        while count < 5:
            n_data = self.shuffle(n_data)

            n_features = n_data[:,:-1]
            n_labels = n_data[:,n_data.shape[1]-1]
            wrong = 0

            for i in range(len(n_features)):
                output = self.get_output(n_features[i], n_weights)
                if output != n_labels[i]:
                    n_weights = self.learning(n_features[i], learning_rate,n_weights, n_labels[i], output)
                    wrong += 1
            
            new_acc = ((len(n_features) - wrong) / len(n_features)) * 100
            # print(new_acc)
            if (abs(old_acc - new_acc)) < 1:
                count += 1
            else:
                old_acc = new_acc
                count = 0

            epoch += 1 
        # print('Epoch: ',epoch)
        self.weights = n_weights

        

    def predict(self, features, labels):
        del labels[:]
        f = np.array(features)
        f = np.append(f,[1])
        value = np.dot(f,self.weights)
        value = np.around(value, decimals=2)
        if value > 0:
            labels += [1]
        else:
            labels += [0]
        



