import numpy as np
import math

def UTIL_FeatureNormalization (X,debug=False):
    # Inputs
    # X is a pandad dataframe 
    # Outputs
    # Normalizaded data 
    
    if debug:
        print('UTIL_FeatureNormalization, Input Data',X.head())
        print('UTIL_FeatureNormalization, describe ()',X.describe())
    X_norm = (X - X.mean()) / X.std ()
    if debug:
        print('UTIL_FeatureNormalization,Output Data',X_norm.head())
    return X_norm , X.mean (), X.std ()

def UTIL_getXMatrix (x):
    """
    Obtener matriz X para posterior pocesamiento, se añade una columna de '1'
    X es la matriz con las muestras
    """
    m = x.shape[0] # Número de muestras
    return np.insert (x,0,1, axis=1)

def UTIL_initVTheta (n_variables,type='random'):
    """
    Inicializacion de theta, 
    n_variables es el número de variables que se usarán en la regresión lineal
    """
    assert type in ['random','zeros'], "Posiles valores de type ['random','zeros']"
    if type == 'random':
        return np.random.randn (1 , n_variables + 1)
    else:
        return np.zeros((1,n_variables +1))

def UTIL_getXMatrix (x):
    """
    Obtener matriz X para posterior pocesamiento, se añade una columna de '1'
    X es la matriz con las muestras
    """
    m = x.shape[0] # Número de muestras
    return np.insert (x,0,1, axis=1)

    
def UTIL_random_mini_batches (X, Y, mini_batch_size = 64, seed = 505):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches