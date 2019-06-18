import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

from learner.generic import Learner


class ExponentialParam:
    def __init__(self, alpha, beta, n_0):
        self.alpha = alpha
        self.beta = beta
        self.n_0 = n_0


class Exponential(Learner):

    version = 0.0
    bounds = ('alpha', 0.000001, 1.0), ('beta', 0.000001, 1.0),\
             ('n_0', 0.0000001, 1)

    def __init__(self, param, tk, verbose=True):
        self.verbose = verbose
        self.tk = tk

        if param is None:
            pass
        elif type(param) == dict:
            self.pr = ExponentialParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = ExponentialParam(*param)
        else:
            raise Exception(
                f"Type {type(param)} is not handled for parameters")

        self.hist = np.zeros(self.tk.t_max, dtype=int)
        self.success = np.zeros(self.tk.t_max, dtype=bool)
        self.t = 0
        self.last_reply = None
        self.last_forgetting_rate = self.pr.n_0
        self.p_random = 1/self.tk.n_possible_replies

        super().__init__()

    def decide(self, question, possible_replies, time=None):

        p_r = self.p_recall(question)
        r = np.random.random()

        if p_r > r:
            reply = question
        else:
            reply = np.random.choice(possible_replies)

        if self.verbose:
            print(f't={self.t}: question {question}, reply {reply}')

        self.last_reply = reply

        return reply

    def p_recall(self, question, time=None):
        """
        Models from Tabibian et al. (2019). PNAS 116 (10) 3988-3993;
        https://doi.org/10.1073/pnas.1815156116

        Simplified version: http://learning.mpi-sws.org/memorize/

        m(t) = exp(-n(t) * (t - t_{last review})

        alpha, beta and n_0 are parameters we learn from the data
        """

        occurrences = (self.hist == question).nonzero()[0]  # returns indexes
        if len(occurrences):
            t_last_review = self.t - occurrences[-1]
        else:
            t_last_review = 0

        """
        Check whether the last recall of that question index was successful
        or not to decide on the n(t) function
        """
        success = self.success[self.hist == question]
        if len(success):
            if self.success[self.hist == question][-1]:
                forgetting_rate = (1 - self.pr.alpha) \
                                  * self.last_forgetting_rate
            else:
                forgetting_rate = (1 + self.pr.beta) \
                                  * self.last_forgetting_rate
        else:
            forgetting_rate = self.pr.n_0

        self.last_forgetting_rate = forgetting_rate

        p_r = math.exp(-forgetting_rate * (self.t - t_last_review))

        return p_r

    def learn(self, question, time=None):
        self.hist[self.t] = question
        self.success[self.t] = self.last_reply == question
        self.t += 1

    def _p_choice(self, question, reply, possible_replies=None, time=None):

        p_retrieve = self.p_recall(question)
        p_correct = self.p_random + p_retrieve*(1 - self.p_random)

        success = question == reply

        if success:
            return p_correct

        else:
            p_failure = (1-p_correct) / (self.tk.n_possible_replies - 1)
            return p_failure



# ############################################### Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages



# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
