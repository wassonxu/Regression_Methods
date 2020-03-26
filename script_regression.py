import csv
import random
import math
import numpy as np

import regressionalgorithms as algs

import MLCourse.dataloader as dtl
import MLCourse.plotfcns as plotfcns

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction, ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction, ytest), ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction, ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions, ytest) / np.sqrt(ytest.shape[0])


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 5

    regressionalgs = {
        #'Random': algs.Regressor,
        #'Mean': algs.MeanPredictor,
#        'FSLinearRegression': algs.FSLinearRegression,
#        'RidgeLinearRegression': algs.RidgeLinearRegression,
        # 'KernelLinearRegression': algs.KernelLinearRegression,
#         'LassoRegression': algs.LassoRegression,
        # 'LinearRegression': algs.LinearRegression,
        # 'MPLinearRegression': algs.MPLinearRegression,
#        'SGD': algs.SGD,
        'BGD': algs.BGD,
    }
    numalgs = len(regressionalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    features_number = 385
    all_features = []
    for i in range(features_number):
        all_features.append(i)
    parameters = {
        'FSLinearRegression': [
            { 'features': [1, 2, 3, 4, 5] },
            { 'features': [1, 3, 5, 7, 9] },
#            { 'features': list(np.sort(np.random.choice(385,size = 10,replace = False))) },
#            { 'features': list(np.sort(np.random.choice(385,size = 20,replace = False))) },
#            { 'features': list(np.sort(np.random.choice(385,size = 50,replace = False))) },
#            { 'features': list(np.sort(np.random.choice(385,size = 80,replace = False))) },
#            { 'features': list(np.sort(np.random.choice(385,size = 100,replace = False))) },
#            { 'features': list(np.sort(np.random.choice(385,size = 120,replace = False))) },
            { 'features': all_features},
        ],
        'RidgeLinearRegression': [
            #{ 'regwgt': 0.00, 'features': all_features },# same as question a issue
            { 'regwgt': 0.01, 'features': all_features },
            { 'regwgt': 0.05, 'features': all_features },
        ],
    }

    errors = {}
    for learnername in regressionalgs:
        # get the parameters to try for this learner
        # if none specified, then default to an array of 1 parameter setting: None
        params = parameters.get(learnername, [ None ])
        errors[learnername] = np.zeros((len(params), numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0], r))

        for learnername, Learner in regressionalgs.items():
            params = parameters.get(learnername, [ None ])
            for p in range(len(params)):
                learner = Learner(params[p])
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p, r] = error


    for learnername in regressionalgs:
        params = parameters.get(learnername, [ None ])
        besterror = np.mean(errors[learnername][0, :]) # inial
        best_std_error = np.std(errors[learnername][0, :])/np.sqrt(numruns)
        bestparams = 0
        for p in range(len(params)):
            aveerror = np.mean(errors[learnername][p, :])
            standard_error = np.std(errors[learnername][p, :])/math.sqrt(numruns)
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p
            if standard_error < best_std_error:
                best_std_error = standard_error
                best_params_stderror = p

        # Extract best parameters
        best = params[bestparams]
        #print ('Best parameters for ' + learnername + ': ' + str(len(best['features'])))
        print ('Best parameters for ' + learnername + ': ' + str(best))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns)))
        # Extract best parameters
#        best = params[bestparams]
#        print ('Best parameters for ' + learnername + ': ' + str(len(best['features'])))
#        print ('Standard error for ' + learnername + ': ' + str(best_std_error))