from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import e
import numpy as np

class BackpropLearner(SupervisedLearner):

    learning_rate = .1
    node_rate = 2 # hidden nodes * node rates = # of hidden nodes
    momentum = 0
    hidden_weights = []
    input_weights = []
    output_node_num = 0
    input_node_num = 0
    hidden_node_num = 0
    count = 5
    thresh = .7

    def __init__(self):
        pass

    def concat_data(self, features, labels):
        n_features = np.array(features)
        ones = np.ones((n_features.shape[0],1))
        n_labels = np.array(labels)
        training = np.hstack((n_features,ones))
        data = np.hstack((training, n_labels))
        return data

    def shuffle(self, n_data):
        np.random.shuffle(n_data)
        return n_data
    
   

    def train(self, features, labels):
        self.output_node_num = output_node_num = len(np.unique(np.array(labels.data)))
        # self.output_node_num = output_node_num = 1
        self.input_node_num = input_node_num = np.array(features.data).shape[1]
        self.hidden_node_num = hidden_node_num = int(self.node_rate * input_node_num)

        print("\n\noutput node: ",output_node_num)
        print("input node: ", input_node_num)
        print("hidden nodes: ", hidden_node_num)

        self.input_weights = np.random.uniform(0, .5, (input_node_num + 1, hidden_node_num))
        self.hidden_weights = np.random.uniform(0, .5, ( hidden_node_num + 1, output_node_num))

        # print("\n\nhidden weights: \n",self.hidden_weights)
        # print("input weights: \n",self.input_weights)

        data = self.concat_data(features.data, labels.data)
        validation_num = int(data.shape[0] * .15)
        count = 0
        epoch = 0
        while (count < self.count):
            epoch += 1
            data = self.shuffle(data)    
            validation_set = data[0:validation_num,:]
            training_set = data[validation_num:,:]
            for temp in range(training_set.shape[0]):

                current_set = training_set[temp]
                input_output = current_set[:-1]
                hidden_output = np.ones((hidden_node_num+1,))
                output = np.zeros((output_node_num,))
                   
                correct = int(current_set[-1])
                
                target = np.zeros((output_node_num,))
                target[correct] = 1
                
                for i in range(self.input_weights.shape[1]):
                    net = np.dot(input_output,self.input_weights[:,i])
                    out = 1/(1+(e**-net))
                    hidden_output[i] = out
                
                for i in range(self.hidden_weights.shape[1]):
                    net = np.dot(hidden_output, self.hidden_weights[:,i])
                    out = 1/(1+(e**-net))
                    output[i] = out
            
                output_sig = []

                for i in range(output.shape[0]):

                    sig = (target[i] - output[i]) * output[i] * (1 - output[i])
                    output_sig.append(sig)
                output_sig = np.array(output_sig)
                update_hidden = np.zeros(self.hidden_weights.shape)

                for i in range(hidden_output.shape[0]):
                    for j in range(output_sig.shape[0]):
                        change = self.learning_rate * output_sig[j] * hidden_output[i]
                        update_hidden[i][j] = change

                hidden_sig = []
        
                for i in range(hidden_output.shape[0]-1):
                    sig_sum = 0
                    for j in range(output_sig.shape[0]):
                        sig_sum += (self.hidden_weights[i][j] * output_sig[j])
                    sig_out = (hidden_output[i] * (1 - hidden_output[i])) * sig_sum
                    hidden_sig.append(sig_out)
                hidden_sig = np.array(hidden_sig)
                
                update_input = np.zeros(self.input_weights.shape)
                
                for i in range(input_output.shape[0]):
                    for j in range(hidden_sig.shape[0]):
                        change = self.learning_rate * hidden_sig[j] * input_output[i]
                        update_input[i][j] = change
                
                self.hidden_weights += update_hidden
                self.input_weights += update_input          


            correct_answer = 0
            total = 0

            for cur in range(validation_set.shape[0]):
                current_set = validation_set[cur]
                input_output = current_set[:-1]
                hidden_output = np.ones((hidden_node_num+1,))
                output = np.zeros((output_node_num,))

                correct = int(current_set[len(current_set) - 1])
                target = np.zeros((output_node_num,))
                target[correct] = 1

                for i in range(self.input_weights.shape[1]):
                    net = np.dot(input_output,self.input_weights[:,i])
                    out = 1/(1+(e**-net))
                    hidden_output[i] = out

                for i in range(self.hidden_weights.shape[1]):
                    net = np.dot(hidden_output, self.hidden_weights[:,i])
                    out = 1/(1+(e**-net))
                    output[i] = out

                total += 1
                if (np.argmax(output) == correct):
                    correct_answer += 1
            
            percent = correct_answer / total
            if (percent > self.thresh):
                count +=1
            else:
                count = 0

        # print("\n\n\n")
        print(epoch)
        return

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        # print(self.input_weights)
        inputs = np.array(features)
        inputs = np.append(inputs,[1])
        # print(inputs)
        del labels[:]
        # correct = int(current_set[len(current_set) - 1])
        # features.append(1)
        # target = np.zeros((self.output_node_num,))

        hidden_output = np.ones((self.hidden_node_num+1,))
        for i in range(self.input_weights.shape[1]):
            net = np.dot(inputs,self.input_weights[:,i])
            out = 1/(1+(e**-net))
            hidden_output[i] = out

        output = np.zeros((self.output_node_num,))
        for i in range(self.hidden_weights.shape[1]):
            net = np.dot(hidden_output, self.hidden_weights[:,i])
            out = 1/(1+(e**-net))
            output[i] = out
        # print(features)
        # print(output)
        answer = np.argmax(output)
        # print(answer)
        labels.append(answer) 
        # print(labels)
