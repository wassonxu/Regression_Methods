import numpy as np
import math

import MLCourse.utilities as utils

# -------------
# - Baselines -
# -------------

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.5, # l2 regularizer
            'features': [1,2,3,4,5],
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]
        # ? why diveded by numsamples
        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

# ---------
# - TODO: -
# ---------

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        #self.params = utils.update_dictionary_items({'regwgt': 0.5}, parameters)
        self.params = utils.update_dictionary_items({
            'regwgt': 0.5, # l2 regularizer
            'features': [1,2,3,4,5],
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]
        # ? why diveded by numsamples
        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest
    
class LassoRegression(Regressor):
    """
    least-squares loss + l1 regularizer
    involves an absolute value, not differenciable
    Substituting in the three cases for  x  from earlier, 
    we find that  x∗=a−λ  if  a>λ ,  a+λ  if  a<−λ , and  0  if  a∈[−λ,λ] . 
    This is known as the soft-thresholding function  S(a,λ) .
    """
    def __init__(self,parameters = {}, learning_rate = 0.001, iterations = 1000, lamda = 0.0005, tolerance = 10e-4):
        self.params = utils.update_dictionary_items({
            'learning_rate': 0.001, 
            'lamda': 0.0005,
        }, parameters)
    
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lamda = lamda
        self.tolerance = tolerance

#        
#    def soft_threshold(self, lamda, weight):
#        if weight > lamda:
#            return weight - self.learning_rate * lamda
#        elif weight < -lamda:
#            return weight + self.learning_rate * lamda
#        else:
#            return 0
    def soft_threshold(self, lamda, weights):
        new_weights = weights
        for i in range(len(weights)):
            if weights[i] > lamda:
                new_weights[i] = weights[i] - self.learning_rate * lamda
            elif weights[i] < -lamda:
                new_weights[i] = weights[i] + self.learning_rate * lamda
            else:
                new_weights[i] = 0
        return new_weights
        
    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        error = 10000000000000
        self.weights = np.zeros(Xtrain.shape[1])
        numsamples = Xtrain.shape[0]
        XX = Xtrain.T.dot(Xtrain) / numsamples
        Xy = Xtrain.T.dot(ytrain) / numsamples
        #import pdb;pdb.set_trace()
        
        for iteration in range(self.iterations):
            iteration += 1
            temp = np.dot(Xtrain,self.weights)-ytrain
            cost = (temp).T.dot(temp) / (2 * numsamples)
            if abs(cost - error) > self.tolerance:
                error = cost
                weight = self.weights - np.dot(self.learning_rate * XX, self.weights) + self.learning_rate * Xy
#                for i in range(len(weight)):
#                    self.weights[i] = self.soft_threshold(self.lamda, weight[i])
                self.weights = self.soft_threshold(self.lamda, weight)
            
#            ols_term = -2 * np.sum(Xtrain * (ytrain - (Xtrain * self.weights))) / numsamples
#            soft_term = self.soft_threshold(self.lamda, self.weights) / numsamples
#            grad = ols_term + soft_term
#            self.weights = self.weights - self.learning_rate * grad


    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class SGD(Regressor):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, parameters = {},step_size = 0.01, epochs = 1000):
        
        self.params = utils.update_dictionary_items({
            'step_size': 0.01, 
            'epochs': 1000,
        }, parameters)

        self.step_size = step_size
        self.epochs = epochs
        
    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples

        self.weights = np.random.rand(Xtrain.shape[1])
        numsamples = Xtrain.shape[0]
        
        for i in range(self.epochs):
            for t in range(numsamples):
                # for one sample
                xt = Xtrain[t,:]
                yt = ytrain[t]
                grad = (xt.T.dot(self.weights) - yt) * xt
                self.weights = self.weights - self.step_size * grad


    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest
    
class BGD(Regressor):
    """
    Batch Gradient Descent
    """
    def __init__(self, parameters = {}, iterations = 1000000, tolerance = 10e-4):
        
        self.params = utils.update_dictionary_items({
            'iterations': 1000000, 
            'tolerance': 0.001,
        }, parameters)
    
        self.iterations = iterations
        self.tolerance = tolerance

    def cost(self, Xtrain, weights, ytrain, numsamples):
        
        temp = np.dot(Xtrain,self.weights)-ytrain # Xw-y
        cost = (temp).T.dot(temp) / (2 * numsamples)
        return cost
    
    def line_search(self, weights, cost, grad, Xtrain, ytrain, numsamples):
        step_size = 1.0 #max initial
        coff = 0.7
        b=0
        for i in range(1000):
            b = b+1 
            new_weight = weights - step_size * grad
            obj = self.cost(Xtrain, new_weight, ytrain, numsamples)
            if obj < cost - 0.0001:
                break
            step_size = coff * step_size
            if b == 999:
                step_size = 0
        #import pdb;pdb.set_trace()    
        return step_size
        
    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        error = 10000000000000000000
        self.weights = np.random.rand(Xtrain.shape[1])
        numsamples = Xtrain.shape[0]
        
        for i in range(self.iterations):
            i += 1
            temp = np.dot(Xtrain,self.weights)-ytrain
            cost = self.cost(Xtrain, self.weights, ytrain, numsamples)
            if abs(cost - error) > self.tolerance:
                error = cost
                grad = Xtrain.T.dot(temp) / numsamples
                step_size = self.line_search(self.weights, cost, grad, Xtrain, ytrain, numsamples)
                self.weights = self.weights - step_size * grad


    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest