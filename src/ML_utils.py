import numpy as np
import math


def CategoricalCrossEntropy (Y,A_last):
    assert A_last.shape [1] > 2 , "Revise la función para calcular cost, Categorical es usada para > 3 salidas y activation Softmax"
    assert A_last.shape == Y.shape , "Revisar shapes A_last, Y "
    m = Y.shape[0]
    J_sum = np.sum(np.multiply(Y.T, np.log(A_last.T)))
    J = -(1./m) * J_sum
    return J

def BinaryCrossEntropy (Y,A_last):
    assert A_last.shape [1] == 1 , "Revise la función para calcular cost, BinaryCrossEntropy es usada para 1 salidas y activation Sigmoid"
    assert A_last.shape == Y.shape , "Revisar shapes A_last, Y "
    m = Y.shape[0]
    J = np.dot(Y.T, np.log(A_last)) + np.dot((1-Y).T, np.log(1-A_last))
    J *= - ( 1 / m )
    return J

    
def UTIL_formatY(Y,num_labels=10):
    # para casos de multiclasificación, nos pone a 1 o 0 dependiendo del número de labels (suponemos 0,1,..., númoro de labels
    # yo = 0 , y1 = 0, .... yn=4 ....ym=9 ....
    # el resuldado será yo = [1,0,0,...] , yn=[0,0,0,0,1,0,...]
    
    result = np.zeros((Y.shape[0],num_labels))
    for idx in range(Y.shape[0]):
        result[idx,Y[idx,0]] = 1
    return result

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

def UTIL_initVTheta (n_variables,type='random',kargs=None):
    """
    Inicializacion de theta, 
    n_variables es el número de variables que se usarán en la regresión lineal
    """
    assert type in ['random','zeros','forzed'], "Posiles valores de type ['random','zeros','forzed']"
    if type == 'random':
        return np.random.randn (1 , n_variables + 1)
    elif type=='zeros':
        return np.zeros((1,n_variables +1))
    else:
        return np.array(kargs['theta_values']).reshape (1,n_variables+1)
    
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
#    shuffled_Y = Y[:, permutation].reshape((1,m))
    shuffled_Y = Y[:, permutation]

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



def softmax (Z):
    y = np.exp (Z - np.max(Z,axis=1,keepdims=True))
    softm  = y / np.sum (y,axis=1,keepdims=True)
    return softm
    
def sigmoid (Z):
    return 1 / (1 + np.exp (-Z))

def NN_CostFunction (A_last,Y):
    # cross-entropy loss.
    assert A_last.shape == Y.shape , "Revisar shapes A_last, Y "
    m = A_last.shape [0]
    K = A_last.shape [1]  # Identificar si es multiclass ( > 1)
    epsilon = 0   # Incluido para evitar divisiones por 0
    J = 0
    for i in range (K):
        Yk = Y[: , i:i+1]
        A_lastk = A_last [:,i:i+1]
        J += np.dot(Yk.T, np.log(A_lastk+epsilon)) + np.dot((1-Yk).T, np.log(1-A_lastk+epsilon))
    J *= - ( 1 / m )
    J = np.squeeze (J)

    assert J.shape == ()
    
    L_sum = np.sum(np.multiply(Y, np.log(A_last)))
    m = Y.shape[0]
    L = -(1./m) * L_sum
    print ('Cost otra forma: ', L )
    return J

def formatY(Y,num_labels=10):
    result = np.zeros((Y.shape[0],num_labels))
    for idx in range(Y.shape[0]):
        result[idx,Y[idx,0]] = 1
    return result

def norm(arr):
    value = np.sqrt(np.dot(arr,arr.T))
    return value
def forward_prop (X,Theta):
    ai = X 
    cache = {'a1':X}
    n_layers = len (Theta) + 1
    for i in range(2,n_layers + 1):
        zi = np.dot (ai , Theta [i-2].T )
        ai = sigmoid (zi)
        cache ['z'+str(i)] = zi
        if i != n_layers :
            ai = np.insert (ai,0,1, axis=1)
        cache ['a'+str(i)] = ai

    return ai , cache

def NNcostFunction ( A_last, Y , Thetas, l2_lambda=0.0 ):
    cost = NN_CostFunction (A_last, Y )
    if l2_lambda != 0.0:
        # Calculamos regularization
        m = Y.shape [0]  # Número de muestras
        l2_cost = (l2_lambda / ( 2 * m ) ) * reduce (lambda ws, w: ws + np.sum(np.square(w)),Thetas,0)
        cost += l2_cost
        print ('costFunction, regularization value {} , with lambda {}'.format(l2_cost,l2_lambda))
    return cost

def backward_prop (X,Y,Thetas,l2_lambda=0.0):
    n_layers = len (Thetas) + 1
    m = X.shape [0]
    prediction,cache = forward_prop (X,Thetas)
    prediction = np.array (prediction)
    print ('Prediction shape',prediction.shape)
    print ('Y shape' , Y.shape)
    delta = {}
    deltai = prediction - Y
    delta ['delta'+str(n_layers) ] = deltai 
    for i in reversed (range(2,n_layers)):
        print ('Calculating delta layer ', i)
        ai = cache['a'+str(i)]
        ai = np.delete (ai,0,axis=1)  # Elimino los 1's que no se tienen en cuenta en el cálculo de deltas
        gprima_z = ai * ( 1 - ai )
        print ('gprima_z shape',gprima_z.shape)
        print ('deltai shape',deltai.shape)

        step1 = np.dot(deltai,Thetas[i-1][:,1:])
        deltaiprev = step1 * gprima_z
        delta ['delta'+str(i) ] = deltaiprev 
        print ('deltaiPREV shape',deltaiprev.shape)
        deltai = deltaiprev
        
    grads = {}
    for i in reversed(range (1, n_layers ) ):
        print ('Calculating grad2 ' , i)
        grad = np.dot (delta['delta'+str(i+1)].T , cache['a'+str(i)] ) / m
        grad [:,1:] = grad [:,1:] + (l2_lambda * Thetas[i-1][:,1:] / m )
        grads ['grad'+str(i)] = grad
        print ('Grad {} shapes delta{} {} a{} {} , theta{} {}, grad shape {} ' .format( i,(i+1),delta['delta'+str(i+1)].shape,i, cache['a'+str(i)].shape,(i-1),Thetas[i-1][:,1:].shape,grad.shape))
        
    return delta , grads


def sigmoid_A(arr, theta):
    z = np.dot(arr, theta)
    return 1.0 / (1 + np.exp(-z))

def reshapeParams(nn_params, input_layer_size=400, hidden_layer_size=25, num_labels=10):
    print ("the type of nn_params in reshapeParams is:%s" % type(nn_params))
    theta1 = np.array(nn_params[:(input_layer_size+1) * hidden_layer_size]).reshape((hidden_layer_size,input_layer_size + 1))
    theta2 = np.array(nn_params[-num_labels * (hidden_layer_size+1):]).reshape((num_labels, hidden_layer_size+1))
    return (theta1, theta2)
def sigmoidGradient(arr, theta):
    sig = sigmoid_A(arr, theta)
    return sig * ( 1 - sig)
def formatY(Y,num_labels=10):
    result = np.zeros((Y.shape[0],num_labels))
    for idx in range(Y.shape[0]):
        result[idx,Y[idx,0]] = 1
    return result

def backpropagation(nn_params,  X, Y, lamda=0.0,input_layer_size=400, hidden_layer_size=25,
                       num_labels=10):
    theta1, theta2 = reshapeParams(nn_params, input_layer_size, hidden_layer_size, num_labels)
    a2 = sigmoid_A(X, theta1.T) # m * hidden_layer_size
    a2 = np.insert(a2,0, 1, axis=1) # m * (hidden_layer_size + 1)
    a3 = sigmoid_A(a2, theta2.T) # m * num_labels

    # format Y from m * 1 to a m*num_labels array
    fY = formatY(Y,num_labels)

    delta3 = a3 - fY   # m * num_labels
    delta2 = np.dot(delta3, theta2[:,1:]) * sigmoidGradient(X, theta1.T)   # m * (hidden_layer_size)
    siggrad = sigmoidGradient(X, theta1.T) 
    print ('Sigmoid gradient - shapes X {} theta {} sigrad {}'.format(X.shape,theta1.shape,siggrad.shape))
    
    grad2 = np.dot(delta3.T, a2) / X.shape[0] # num_labels * (hidden_layer_size+1)
    print ('Grad2 shapes delta3 {} a2 {}, grad2 {} ' .format( delta3.shape,a2.shape,grad2.shape))
    grad2[:,1:] = grad2[:,1:] + (lamda * theta2[:,1:]/X.shape[0]) 
    
    grad1 = np.dot(delta2.T, X) / X.shape[0] # (hidden_layer_size) * (input_layer_size+1)
    grad1[:,1:] = grad1[:,1:] + (lamda * theta1[:,1:]/X.shape[0])

    return np.append(grad1.flatten(),grad2.flatten()) , {'delta3':delta3, 'delta2':delta2 ,'grad2':grad2,'grad1':grad1,'gradienteinicial':siggrad}
