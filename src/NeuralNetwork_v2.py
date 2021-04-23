import numpy as np
from ML_utils import UTIL_random_mini_batches
from tqdm import tqdm
from pandas import DataFrame
import math

class CrossEntropy :
    def f (self):
        raise NotImplementedError
    def df (self):
        raise NotImplementedError
        
class CategoricalCrossEntropy(CrossEntropy):
    def __init__ (self):
        pass
    def  f (self, Y, A_last):
        Y = Y.T
        A_last = A_last.T
        assert A_last.shape [1] > 2 , "Revise la función para calcular cost, Categorical es usada para > 3 salidas y activation Softmax"
        assert A_last.shape == Y.shape , "Revisar shapes A_last, Y "
        m = Y.shape[0]
        J_sum = np.sum(np.multiply(Y.T, np.log(A_last.T)))
        J = -(1./m) * J_sum
        return J
    def df (self,Y,A_last,epsilon=1e-20):
        return - np.divide(Y,np.clip(A_last,epsilon,1.0))

class BinaryCrossEntropy(CrossEntropy):
    def __init__ (self):
        pass
    def  f (self, Y, A_last):
        Y = Y.T
        A_last = A_last.T
        assert A_last.shape [1] == 1 , "Revise la función para calcular cost, BinaryCrossEntropy es usada para 1 salidas y activation Sigmoid"
        assert A_last.shape == Y.shape , "Revisar shapes A_last, Y "
        m = Y.shape[0]
        J = np.dot(Y.T, np.log(A_last)) + np.dot((1-Y).T, np.log(1-A_last))
        J *= - ( 1 / m )
        return J
    def df (self, Y,A_last):
        return - (np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last)) 


class Function :
    def f (self):
        raise NotImplementedError
        
    def df (self):
        raise NotImplementedError


class SoftmaxF (Function):
    def __init__(self):
        self.cache = {}
        
    def f (self,x):
        y = np.exp (x - np.max(x,axis=0,keepdims=True))
        softm  = y / np.sum (y,axis=0,keepdims=True)
        self.cache ['A'] = softm
        return softm
   
    def df (self,dA):
        
        A = self.cache ['A']
        Y = dA * (-A)
        dZ = A - Y
        return dZ
    
class LinearF (Function):
    def __init__(self):
        pass
    def f (self,x):
        return x
    
    def df (self,x):
        return 1
 
    
    
class SigmoidF (Function):
    def __init__(self):
        pass
    def f (self,x):
        return 1 / (1+np.exp(-x))
    
    def df (self,x):
        s = self.f(x)
        return s * (1-s)
    
class ReluF (Function):
    def __init__(self):
        pass
    def f (self,x):
        return np.maximum(0,x)
    
    def df (self,x):
        x [x<=0 ] = 0
        x [ x>0 ] = 1
        return x

class LReluF (Function):
    def __init__(self,epsilon=0.01):
        self.epsilon = epsilon
    def f (self,x):
        return np.maximum(self.epsilon*x,x)
    
    def df (self,x):
        x [x<=0 ] = self.epsilon
        x [ x>0 ] = 1
        return x

class TanhF (Function):
    def __init__(self):
        pass
    
    def f (self,x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def df (self,x):
        s = self.f(x)
        return (1 - s**2)


class Linear :
    def __init__ (self,n_nodes,n_previous):
        self.cache = {}
        self.n_nodes = n_nodes
        self.n_previous = n_previous
        
    def forward(self,A, W, b):
        assert A.shape [0] == self.n_previous , "A debe ser (n_previous, muestras)"
        assert W.shape  == (self.n_nodes,self.n_previous) , "W debe ser (n_nodes, n_previous)"
        assert b.shape  == (self.n_nodes,1) , "b debe ser (n_nodes, 1)"

        Z = np.dot(W, A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        self.cache ['A'] = A
        self.cache ['W'] = W
        self.cache ['b'] = b

        return Z

    def backward (self,dZ):
        A_prev= self.cache ['A']
        W = self.cache ['W']
        b = self.cache ['b']
        m = A_prev.shape[1]

        dW = (1./m) * np.dot(dZ, A_prev.T)
        db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
    
        return dA_prev, dW, db

class Activation_V2:    
    def __init__ (self,function=SigmoidF):
        self.function = function ()
        self.is_softmax = isinstance(self.function,SoftmaxF)
        self.cache = {}
    def forward (self,Z):
        A = self.function.f (Z)
        assert(A.shape == Z.shape)
        self.cache ['Z'] = Z
        return A
    
    def backward (self,dA):
        Z = self.cache ['Z']
        if self.is_softmax:
            dactivation = self.function.df (dA)
            dZ = dactivation
        else:
            dactivation = self.function.df (Z)
            dZ = dA * dactivation
        assert (dZ.shape == Z.shape)
        return dZ    
    

   
    
class Layer:
    def __init__(self,n_nodes,n_inputs,activationF=SigmoidF,keep_prob=1.0):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.W = None
        self.b = None
        self.Vdw = np.zeros((self.n_nodes, self.n_inputs))  # Variable para momentum 
        self.Vdb =  np.zeros((self.n_nodes, 1))             
        self.Sdw = np.zeros((self.n_nodes, self.n_inputs))  # Variable para RMSprop 
        self.Sdb =  np.zeros((self.n_nodes, 1))             
        self.keep_prob = keep_prob                 # Un valor menor que 1 indica dropout regularization
        self.cache = {}
        
        self.linear = Linear (n_nodes,n_inputs)
        self.activation = Activation_V2 (function=activationF)
        self.__initWeigths ()
        
        
    def getParameters (self):
        return self.W, self.b
    def setParameters (self,W,b):
        self.W = W 
        self.b = b
    
    def getGradients (self):
        return self.cache['dW'], self.cache['db']
    def getParametersShape(self):
        return self.W.shape, self.b.shape
    
    def __initWeigths (self):
        
        if isinstance(self.activation.function,TanhF ):
            self.W = np.random.randn(self.n_nodes, self.n_inputs) * np.sqrt( 1 / self.n_inputs)
        elif isinstance(self.activation.function,ReluF) or isinstance(self.activation.function,LReluF) :
            self.W = np.random.randn(self.n_nodes, self.n_inputs) * np.sqrt( 2 / self.n_inputs)
        else:
            self.W = np.random.randn(self.n_nodes, self.n_inputs) / np.sqrt(self.n_inputs)    
            
        self.b = np.zeros((self.n_nodes, 1))
        
        assert (self.W.shape == (self.n_nodes, self.n_inputs))
        assert (self.b.shape == (self.n_nodes, 1))
            
    def forward (self, A_prev,training=False):
        Z = self.linear.forward (A_prev,self.W,self.b)
        A = self.activation.forward (Z)
        if (self.keep_prob < 1.0)  and training :
            D = np.random.rand(A.shape[0], A.shape[1])     
            D = (D < self.keep_prob)
            A = np.multiply(A, D)                          
            A = A / self.keep_prob
            self.cache['D'] = D
        return A
    
    def backward (self, dA , training ):
        if (self.keep_prob < 1) and training:
            dA = np.multiply(dA, self.cache['D'])
            dA = dA/self.keep_prob
            
        dZ = self.activation.backward (dA)
        dA_prev, dW, db = self.linear.backward(dZ)
        self.cache ['dA_prev'] = dA_prev
        self.cache ['dW'] = dW
        self.cache ['db'] = db
        self.cache ['paramdW'] =  np.copy(self.cache ['dW'])
        self.cache ['paramdb'] =  np.copy(self.cache ['db'])
        return dA_prev, dW,db
    
    def optimize (self,learning_rate):
        self.W , self.b = self.optimazationAlgorithm.optimize ( learning_rate,self.W,self.b,self.cache ['dW'],self.cache ['db'] )
        
    def computeAdam (self,iteration,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.computeMomentum (beta1)
        self.computeRMSprop (beta2)
        Vdw_corr = self.Vdw / (1 - np.power(beta1,iteration))
        Vdb_corr = self.Vdb / (1 - np.power(beta1,iteration))
        Sdw_corr = self.Sdw / (1 - np.power(beta2,iteration))
        Sdb_corr = self.Sdb / (1 - np.power(beta2,iteration))       
        self.cache ['paramdW'] =  Vdw_corr / (np.sqrt (Sdw_corr) + epsilon)
        self.cache ['paramdb'] =  Vdb_corr / (np.sqrt (Sdb_corr) + epsilon)
        
    def computeRMSprop (self,iteration,beta=0.9,epsilon=1e-8):
        correction = 1 - np.power (beta,iteration)          
        self.Sdw = beta * self.Sdw + ((1-beta) * np.square (self.cache ['dW'] ))
        self.Sdb = beta * self.Sdb + ((1-beta) * np.square (self.cache ['db'] ))
        Sdw_corrected  = self.Sdw / correction
        Sdb_corrected  = self.Sdb / correction
        self.cache ['paramdW'] = self.cache ['dW'] / (np.sqrt (Sdw_corrected) + epsilon)
        self.cache ['paramdb'] = self.cache ['db'] / (np.sqrt (Sdb_corrected) + epsilon)
        
    def computeMomentum (self,beta=0.9):
        self.Vdw = beta * self.Vdw + (1-beta) * self.cache ['dW']
        self.Vdb = beta * self.Vdb + (1-beta) * self.cache ['db']
        self.cache ['paramdW'] = self.Vdw
        self.cache ['paramdb'] = self.Vdb
     
    def updateParams (self,learning_rate=0.01):
        #self.W = self.W - learning_rate * self.cache ['dW'] 
        #self.b = self.b - learning_rate * self.cache ['db'] 
        self.W = self.W - learning_rate * self.cache ['paramdW']  # si gd: paramdW=dW,si momentum: paramdW=Vdw, si RMSprop=paramdW=dW/np.sqrt(Sdw)
        self.b = self.b - learning_rate * self.cache ['paramdb']  # si gd: paramdb=db,si momentum: paramdb=Vdb, si RMSprop=paramdb=db/np.sqrt(Sdb)
        
class NNClassifier_V2:
    def __init__(self,layers,costFunction=BinaryCrossEntropy,normalization=False,kargs={'steps':100,'learning_rate':0.01},print_cost=True):
        self.layers = layers
        self.costF = costFunction()
        self.kargs = kargs
        self.normalization = normalization
        self.norm_mean = None
        self.norm_std = None
        self.print_cost = print_cost
        
    def _normalizeX (self, X):
        # X is a matrix (n features, m samples)
        dfX = DataFrame (X.T)  #dfX es una matriz (m samples, x features)
        try:
            if self.norm_mean == None:
                self.norm_mean = dfX.mean()
                self.norm_std  = dfX.std ()
        except:
            pass   # No hago nada, se supone que existe mean y std
        
        X_norm = (dfX - self.norm_mean) / self.norm_std
        return DataFrame(X_norm).to_numpy ().T  # devuelvo matriz (x features, m samples)
    

    def predict (self,X):
        if self.normalization:
            X = self._normalizeX (X)
        return self.__forward (X)
        
    def __forward (self,X,training=False):
        Aprev = X
        for layer in self.layers:
            A = layer.forward (Aprev,training)
            Aprev = A
        return A

    def __batchGD (self, X,Y, l2_lambda=0.0,training=True):
        costs = []
        prev_cost = 0
        not_convergency_counter = 0
        lr = self.kargs ['learning_rate']
        optimization = 'GD'
        try:
            optimization = self.kargs ['optimization']
            beta1 = self.kargs ['beta1']
            beta2 = self.kargs ['beta2']
        except:
            pass
        assert optimization in ['GD','momentum','Adam','RMSprop'] , "Available values are : 'GD','momentum','Adam','RMSprop'"
        
        for i in tqdm(range (self.kargs ['steps']),desc='LR = ' + str(lr)):
            prediction = self.__forward (X,training)
            assert prediction.shape == Y.shape
            cost = self.costFunction (Y,prediction,l2_lambda)
            assert not math.isnan(cost) , "Cost es NaN , revise los parámetros"
            if cost >= prev_cost:
                not_convergency_counter += 1
                prev_cost = cost
            else:
                not_convergency_counter = 0
                prev_cost = cost

            assert not_convergency_counter <= 10 , 'No parece que GD esté convergiendo revisa alpha ' + str(lr) + ',cost ' + str(cost) + 'iteration ' + str(i)

            costs.append (cost)
            self.__backward (prediction,Y,training)
            if self.print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                
            if optimization == 'momentum':
                self.__computeMomentum (beta=beta1)
            elif optimization == 'RMSprop':
                self.__computeRMSprop (i+1,beta=beta2)
            elif optimization == 'Adam':
                self.__computeAdam (i+1,beta1=beta1,beta2=beta2)
            else:
                pass
            self.__updateParamsFromCache (lr)
        return costs 

     
    def __miniBatchGD (self, X,Y, l2_lambda=0.0,training=True):
        costs = []
        prev_cost = 0
        not_convergency_counter = 0
        lr = self.kargs ['learning_rate']
       
        optimization = 'GD'
        try:
            optimization = self.kargs ['optimization']
            beta1 = self.kargs ['beta1']
            beta2 = self.kargs ['beta2']
        except:
            pass
        assert optimization in ['GD','momentum','Adam','RMSprop'] , "Available values are : 'GD','momentum','Adam','RMSprop'"

        for i in tqdm(range (self.kargs ['steps']),desc='LR = ' + str(lr)):
            minibatches = UTIL_random_mini_batches (X ,Y, self.kargs['mini_batch_size'])
            for (miniBatchX,miniBatchY) in minibatches:
                prediction = self.__forward (miniBatchX,training)
                assert prediction.shape == miniBatchY.shape
                cost = self.costFunction (miniBatchY,prediction,l2_lambda)
                assert not math.isnan(cost) , "Cost es NaN , revise los parámetros"
                    
                if cost >= prev_cost:
                    not_convergency_counter += 1
                    prev_cost = cost
                else:
                    not_convergency_counter = 0
                    prev_cost = cost

                assert not_convergency_counter <= 10 , 'No parece que GD esté convergiendo revisa alpha ' + str(lr) + ',cost ' + str(cost) + 'iteration ' + str(i)

                costs.append (cost)
                self.__backward (prediction,miniBatchY,training)

                if optimization == 'momentum':
                    self.__computeMomentum (beta=beta1)
                elif optimization == 'RMSprop':
                    self.__computeRMSprop (i+1,beta=beta2)
                elif optimization == 'Adam':
                    self.__computeAdam (i+1,beta1=beta1,beta2=beta2)
                else:
                    pass
                self.__updateParamsFromCache (lr)
        return costs  
    
    def __backward (self, A_last,Y , training=True):
        Y = Y.reshape(A_last.shape)
        # derivative of cost with respect to AL
#        dA_last = - (np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))    Esto es para BinaryCrossEntropy
        dA_last = self.costF.df ( Y, A_last)
        for layer in reversed(self.layers):
            dA_prev, dW, db = layer.backward (dA_last,training)
            dA_last = dA_prev
        
    def __updateParamsFromCache (self,learning_rate=0.01):
        for layer in self.layers:
            layer.updateParams (learning_rate=learning_rate)
            
    def __computeAdam(self,iteration,beta1=0.9,beta2=0.999):
        for layer in self.layers:
            layer.computeAdam (iteration,beta1,beta2)
    def __computeRMSprop(self,iteration,beta=0.0):
        for layer in self.layers:
            layer.computeRMSprop (iteration,beta=beta)
    def __computeMomentum (self,beta=0.0):
        for layer in self.layers:
            layer.computeMomentum (beta)

    def optimize ( self,X,Y,l2_lambda=0.0 ,method='SGD'):            
        if self.normalization:
            X = self._normalizeX (X)
            
        if method == "batchGD":
            return self.__batchGD (X,Y,l2_lambda,training=True)
        elif method == "miniBatchGD":
            return self.__miniBatchGD (X,Y,l2_lambda,training=True)
        else:
            raise NotImplementedError ("Valid methods are :'batchGD','miniBatchGD'")
        
    def costFunction ( self,Y , A_last, l2_lambda=0.0 ):
        cost = self.costF.f (Y,A_last)
        return cost
