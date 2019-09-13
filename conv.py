import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.core.framework import types_pb2
import numpy as np
import math

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """ 
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
       7 
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32,shape= (None,n_H0,n_W0,n_C0))
    Y = tf.placeholder(tf.float32,shape= (None, n_y))

    return X, Y


def initialize_parameters(): 
    """
    Initializes weight parameters (filters/kernels) to build a neural network with tensorflow. 
                        W1 = [3, 3, 1, 10]
                        W2 = [2, 2, 10, 20]
                    
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1) # so that your "random" numbers match ours    
    
    W1 = tf.get_variable("W1", [6, 6, 1, 10],    initializer = tf.contrib.layers.xavier_initializer(seed = 0)) 
    W2 = tf.get_variable("W2", [2, 2, 10, 20],  initializer = tf.contrib.layers.xavier_initializer(seed = 0)) 
  
    parameters = {"W1": W1,"W2": W2} 
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
   
    # CONV2D: filter W1, stride of 3, padding 'SAME', first conv layer with max pooling 
    Z1 = tf.nn.conv2d(X, W1, strides = [1,3,3,1], padding = 'SAME', name="Z1")
    
    # RELU
    A1 = tf.nn.relu(Z1)
   
    # MAXPOOL: window 2x2, sride 2, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,3,3,1], strides = [1,3,3,1], padding = 'SAME')
    #print("P1", P1)
    
    # CONV2D: filter W2, stride of 1, padding 'SAME', first conv layer with max pooling 
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME', name="Z2")
    
    # RELU
    A2 = tf.nn.relu(Z2)
    
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    conv5 = tf.layers.flatten(P2)

    n_classes = 2 # prediction either correct or incoreect
    
    
    Z3 = tf.contrib.layers.fully_connected(conv5, n_classes, activation_fn= None)
    
    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation, of shape (number of examples, 2)
    Y -- "true" labels vector placeholder, same shape as Z5
    
    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    
    return cost


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


