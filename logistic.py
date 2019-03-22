from __future__ import division, print_function

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################
from typing import List
#HN: Feature function to help with binary_train;

def add_col_ones(X):
    """
    Append a column of 1 to the left of the input matrix; 
    """
    return np.hstack((np.ones((X.shape[0],1)),X))

def normalize_features(X):
    """
    Return an array with element subtracted the max in that column; 
    """
    return np.subtract(X,np.amax(X,axis=0))

def gradient_descent(features: List[List[float]], labels, weights: List[float],
                    step_size=0.5, max_iteration=1000) -> List[float]:
    """
    The gradient descent function should take into account the feature vectors
    and only update by the average of each vector;
    """
#     labels = np.asarray(labels)
#     assert features.shape[0] == labels.shape[0]
    #Append a column of 1 to the features vector;
    features = add_col_ones(features)
    #Initialize weight_delta vector to all 0;
    for n in range(max_iteration):
        #Get the total delta array after each loop through batch of trainign set 
        for i in range(len(labels)):
#             print("features:", features[i])
#             print("weights", weights)
            delta = np.multiply(sigmoid(sum(np.multiply(weights,features[i]))) - labels[i], features[i])
            weights = np.subtract(weights, np.multiply(step_size,delta))
    return weights


def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression

    Find the optimal parameters w and b for inputs X and y.
    Use the average of the gradients for all training examples to
    update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
    base_weights = np.concatenate((np.asarray([b]), w))
#     print(base_weights)
    trained_weights  = gradient_descent(X, y, base_weights, step_size, max_iterations)
    w = trained_weights[1:]
    b = trained_weights[0]

    assert w.shape == (D,)
    return w,b

def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = np.zeros(N)
    
    X = add_col_ones(X)
    weights = np.concatenate((np.asarray([b]),w))#add bias before weights  
#     print("Weights:", weights)
    for i in range(N):
        y_pred = sigmoid(sum(np.multiply(weights,X[i])))
#         print("Predicted Probability:", y_pred)
        if y_pred >= 0.5:
            preds[i] = 1
    assert preds.shape == (N,) 
    return preds 
    
def softmax(features):
    num_rol, num_col = features.shape
    pred = np.asarray([np.true_divide([np.exp(i) for i in x],sum(np.exp(i) for i in x)).tolist() for x in features])
    return pred.reshape((num_rol, num_col))
    
def multinomial_gradient_descent(X,y,w,step_size = 0.5,max_iterations = 1000):
    """
    Train the gradient descent using the softmax function instead;
    """
    
    return None

def multinomial_train(X, y, C, 
                     w0=None, 
                     b0=None, 
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: maximum number for iterations to perform

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement a multinomial logistic regression for multiclass 
    classification. Keep in mind, that for this task you may need a 
    special (one-hot) representation of classification labels, where 
    each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i belongs to. 
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0


    """
    1. Transform the labels into binary-coded set;
    2. Use LR to train for each binary category; 
    3. Output the set of weights for each category;
    """
    num_cat = C
    X = add_col_ones(X)
    new_y = OVR_transform_y(y,num_cat)
    weights = np.hstack((b.reshape(w.shape[0],1),w))
    
 
    #temp step: find the delta with a certain weight; 
    for i in range(max_iterations): 
#         print("True y: {}; y_hot_coded: {}".format(y[i], new_y[i]))
        y_hot_pred = softmax(np.matmul(X, weights.T))
#         print(y_hot_pred)
#         print(y_pred)  
        delta = np.subtract(new_y,y_hot_pred)
#         print(delta)
#         print(delta.shape)
        #BATCH-GRADIENT DESCENT: take the average of tall the deltas for each feature;
        for cat in range(C):#iterate through each category
            update = np.multiply(np.average(np.multiply(X.T, delta[:,cat]),axis = 1),step_size)
            weights[cat,:] = np.add(weights[cat,:], update)
#             print("cat: {}, weight: {}".format(cat, weights[cat,:]))
       
    w = weights[:,1:]
    b = weights[:,0]
    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 

    """
    1. Append 1 to front of X; 
    2. argmax of output to predict; 
    """   
    X = add_col_ones(X)
    weights = np.hstack((b.reshape(w.shape[0],1),w))
    
    preds = np.argmax(softmax(np.matmul(X, weights.T)), axis=1)
#     print(preds.shape)
    assert preds.shape == (N,)
    return preds


def OVR_transform_y(y, C):
    """
        C: number of classes; 
        Tranform the array of labels into a set of K 
        vectors that correspond to binary flag for each 
        k_j category. 
        Ex: 2 becomes [0 0 1];
    """
    N = len(y)
    new_y = np.zeros(N*C).reshape(N,C)
    for i in range(N):
#         print( new_y[i, int(y[i])])
        new_y[i, int(y[i])] = 1
    return new_y


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array, 
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: maximum number of iterations for gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using binary classifier and 
    one-versus-rest strategy. Recall, that the OVR classifier is 
    trained by training C different classifiers. 
    """
    N, D = X.shape
    
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    """
    1. Transform the labels into binary-coded set;
    2. Use LR to train for each binary category; 
    3. Output the set of weights for each category;
    """
    num_cat = C
    new_y = OVR_transform_y(y,num_cat)
    
    #Note: for numpy slicing: axis 1 is horizontally, axis 0 is by row;
    w, b = [],[]
    for c in range(C):
        weight, bias = binary_train(X,new_y[...,c])
        #Append the weight to the array
        w = np.concatenate((w,weight))
        b = np.concatenate((b,[bias]))
#         print("c:{},weight:{},bias:{}".format(c,weight,bias))
    w = np.asarray(w).reshape(C,D)
    b = np.asarray(b).reshape(C)   
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model
    
    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and predictions from binary
    classifier. 
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 
    
    """
    1. Transform the features (add 1 col) and append the weight matrix together;
    2. Get the probability by applying sigmoid on all X' and w'.T (transpose)
    3. This gives a N*C matrix that contains the probs. of each k for each obs. 
    4. Pick the max one
    """
    X = add_col_ones(X)
    w = np.hstack((b.reshape(w.shape[0],1),w))
#     print("X",X)
#     print("w",w)
    res = sigmoid(np.matmul(X,w.T)) 
    preds = np.asarray([np.argmax(r) for r in res]) 
    
    assert preds.shape == (N,)
    return preds


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, \
                            data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
        
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    
    w, b = binary_train(X_train, binarized_y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(binarized_y_train, train_preds),
             accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
                            toy_data_multiclass_5_classes, \
                            data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 
                        'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
            sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
        