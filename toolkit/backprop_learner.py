from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import e
import numpy as np

class BackpropLearner(SupervisedLearner):

    learning_rate = .4
    node_rate = 1 # input nodes * node rate = # of hidden nodes
    momentum = .9
    hidden_weights = []
    input_weights = []
    output_node_num = 0
    input_node_num = 0
    hidden_node_num = 0
    count = 20
    thresh = .7
    bssf_hidden = []
    bssf_input =[]

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
        self.input_node_num = input_node_num = np.array(features.data).shape[1]
        self.hidden_node_num = hidden_node_num = int(self.node_rate * input_node_num)
        

        print("\n\ninput node: ", input_node_num)
        print("hidden nodes: ", hidden_node_num)
        print("output node: ",output_node_num)

        self.input_weights = np.random.uniform(0, .5, (input_node_num + 1, hidden_node_num))
        self.hidden_weights = np.random.uniform(0, .5, ( hidden_node_num + 1, output_node_num))
        previous_hidden = np.zeros(self.hidden_weights.shape)
        previous_input = np.zeros(self.input_weights.shape)

        data = self.concat_data(features.data, labels.data)
        validation_num = int(data.shape[0] * .20)
        count = 0
        epoch = 0
        mse_val = 100000000
        while (count < self.count):
            epoch += 1
            data = self.shuffle(data)    
            validation_set = data[0:validation_num,:]
            training_set = data[validation_num:,:]
            training_mse = []
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

                out = np.zeros((output_node_num,))
                out_index = np.argmax(output)
                out[out_index] = 1
                training_mse.append((output - target)**2)

                output_sig = []

                for i in range(output.shape[0]):

                    sig = (target[i] - output[i]) * output[i] * (1 - output[i])
                    output_sig.append(sig)
                output_sig = np.array(output_sig)
                change_hidden = np.zeros(self.hidden_weights.shape)

                for i in range(hidden_output.shape[0]):
                    for j in range(output_sig.shape[0]):
                        change = self.learning_rate * output_sig[j] * hidden_output[i]
                        change_hidden[i][j] = change

                hidden_sig = []
        
                for i in range(hidden_output.shape[0]-1):
                    sig_sum = 0
                    for j in range(output_sig.shape[0]):
                        sig_sum += (self.hidden_weights[i][j] * output_sig[j])
                    sig_out = (hidden_output[i] * (1 - hidden_output[i])) * sig_sum
                    hidden_sig.append(sig_out)
                hidden_sig = np.array(hidden_sig)
                
                change_input = np.zeros(self.input_weights.shape)
                
                for i in range(input_output.shape[0]):
                    for j in range(hidden_sig.shape[0]):
                        change = self.learning_rate * hidden_sig[j] * input_output[i]
                        change_input[i][j] = change
                

                previous_hidden *= self.momentum
                previous_input *= self.momentum

                self.hidden_weights += change_hidden + previous_hidden
                self.input_weights += change_input + previous_input

                previous_hidden = change_hidden
                previous_input = change_input          

            train_mse_val = np.sum(training_mse) / training_set.shape[0]
            
            correct_answer = 0
            total = 0
            mse_sum = []
            for cur in range(validation_set.shape[0]):
                current_set = validation_set[cur]
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

                total += 1
                out = np.zeros((output_node_num,))
                out_index = np.argmax(output)
                out[out_index] = 1
                mse_sum.append((output - target)**2)
                if (np.argmax(output) == correct):
                    correct_answer += 1

            mse_sum = np.sum(np.array(mse_sum))
            mse = mse_sum / validation_set.shape[0]

            if mse < mse_val:
                mse_val = mse
                self.bssf_hidden = np.copy(self.hidden_weights)
                self.bssf_input = np.copy(self.input_weights)
                count = 0
            else:
                count += 1
            percent = correct_answer / total

        print("epoch:", epoch)
        print("VS",mse)
        print("TRAIN",train_mse_val)
        print("LR", self.learning_rate)
        print("momentum", self.momentum)
        print("hidden nodes", self.bssf_hidden)
        print("input nodes", self.bssf_input)
        return

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        inputs = np.array(features)
        inputs = np.append(inputs,[1])
        del labels[:]
       

        hidden_output = np.ones((self.hidden_node_num+1,))
        for i in range(self.bssf_input.shape[1]):
            net = np.dot(inputs,self.bssf_input[:,i])
            out = 1/(1+(e**-net))
            hidden_output[i] = out

        output = np.zeros((self.output_node_num,))
        for i in range(self.bssf_hidden.shape[1]):
            net = np.dot(hidden_output, self.bssf_hidden[:,i])
            out = 1/(1+(e**-net))
            output[i] = out
       
        answer = np.argmax(output)
        labels.append(answer) 
