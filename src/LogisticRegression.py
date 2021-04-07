import numpy as np
import sys
sys.path.append ('../src') 
from utils import UTIL_FeatureNormalization,UTIL_initVTheta,UTIL_getXMatrix,UTIL_random_mini_batches
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from scipy.optimize import minimize

def LogR_predictProbability (x,vtheta):
    assert vtheta.shape[0] == 1
    Z = np.dot (x,vtheta.T)
    return 1 / (1 + np.exp (-Z))

def LogR_CostFunction (theta,x,y,lambd):
    theta = np.array (theta).reshape (1,len(theta))
    y = np.array (y).reshape (len(y),1)
    prediction = LogR_predictProbability (x,theta)
    m = prediction.shape [0]
    assert prediction.shape[1] == 1
    assert y.shape[1] == 1
    assert prediction.shape[0] == y.shape[0]
    epsilon = 1e-10   # Incluido para evitar divisiones por 0
    regularization = (lambd/(2*m)) * np.sum (theta[0][1:]**2)
    J = - ( 1 / m) * ( np.dot(y.T, np.log(prediction+epsilon)) + np.dot((1-y).T, np.log(1-prediction+epsilon))) + regularization
    J = np.squeeze (J)
    assert J.shape == ()

    return J
def LogR_CalculateGrads (theta,x,y,lambd):
    theta = np.array (theta).reshape (1,len(theta))
    y = np.array (y).reshape (len(y),1)
    prediction = LogR_predictProbability (x,theta)
    m = prediction.shape [0]
    regularization = (lambd / m ) * theta
    regularization [0] [0]= 0    #Theta 0 no tiene regularization
    grads =  (1/m) * np.dot ((prediction-y).T,x) + regularization
    return grads
    
class LogisticRegression:
    def __init__ (self,optimizer="GD" ,normalization=True, minibatch=False,lr_lambda=0.0,kargs={'lambda':0.0,'mini_batch_size':2**10}):
        assert optimizer in ["GD","Optimize"],"Possible optimizers are 'GD,Optimize'"
        self.optimizer  = optimizer
        self.lr_lambda  = lr_lambda
        self.normalization = normalization
        self.X_mean = None
        self.X_std = None
        self.minibatch = minibatch
        self.minibatches = None
        self.X = None
        self.Y = None
        self.n = None
        self.m = None
        self.vtheta = None
        self.kargs = kargs
        self.costs = []
    
    def _initializeData (self,X,Y ):
        assert X.shape [0] == Y.shape[0] , "Revisar dimensiones X e Y"
        self.m = X.shape [0]   # Número de muestras
        self.n = X.shape [1]   # Número de features (sin incluir 1's)
        if self.normalization:
            self.X ,self.X_mean,self.X_std = UTIL_FeatureNormalization (pd.DataFrame(X),debug=False)
            self.X = self.X.values # numpy array (antes era un DataFrame de pandas)
        else:
            self.X = np.array (X)
            
        # Añado columna de 1's
        self.X = UTIL_getXMatrix (self.X)

        self.Y = np.copy (Y)
        
    def train (self,X,Y):
        self._initializeData (X,Y)
        self.vtheta = UTIL_initVTheta (self.n, type=self.kargs['theta_init'])
        if self.optimizer == 'Optimize':
            self._calculateOptimize ()
            assert self.vtheta.shape [1] == self.n + 1
            assert self.vtheta.shape [0] == 1
        else:
            #self.vtheta = UTIL_initVTheta (self.n, type=self.kargs['theta_init'])
            assert self.X.shape[1] == self.vtheta.shape[1] , "Dimensiones X y vtheta no coinciden"
            assert self.vtheta.shape [0] == 1
            if self.minibatch:
                self._calculateMiniBatchGradientDescent ()
            else:
                self._calculateBatchGradientDescent ()
                
        if self.minibatch :  # Vuelvo a valores originales
            self._initializeData (X,Y)
            
        
    def _calculateOptimizev2 (self):
        vtheta = fmin_tnc(func= LogR_CostFunction, x0=self.vtheta.T, fprime=LogR_CalculateGrads,args=(self.X, self.Y.flatten(),self.kargs['lambda']))
        self.vtheta = np.array(vtheta [0]).reshape (1,len(vtheta[0]))

    def _calculateOptimize (self):
        vtheta = minimize(fun= LogR_CostFunction, x0=self.vtheta.T, method='TNC', jac=LogR_CalculateGrads,args=(self.X, self.Y.flatten(),self.kargs['lambda']))
        self.vtheta =  np.array(vtheta.x).reshape (1,len(vtheta.x))
        
        
    def predict (self, X ):
        if self.normalization:
            # Normalizamos en base a la normalización del training e incluimos 1's
            df = pd.DataFrame (X)
            self.X = (( df - self.X_mean ) / self.X_std ).values
        
        else:
            self.X = np.array (X)
            
            
        # Añado columna de 1's
        self.X = UTIL_getXMatrix (self.X)
        return self._predict ()
    
    def _predict (self):
        """
        Predicción logarítmica,
        self.X es la matriz con las muestras .Dimensión  (m,n) donde m es el número de muestras y n el número de features
        self.vtheta es el vector de Thetas [tHeta0,theta1,theta2 ...]. Dimensión (1,n), siendo n el número de features
        """
        assert self.vtheta.shape[0] == 1 , "Theta debe ser  de dimensión (1,n)"
        Z = np.dot (self.X,self.vtheta.T)
        return 1 / (1 + np.exp (-Z))

    def _calculateMiniBatchGradientDescent (self):
        self.costs = []
        lr = self.kargs ['learning_rate']
            
        for i in tqdm(range (self.kargs ['steps']),desc='LR = ' + str(lr)):
            self.minibatches = UTIL_random_mini_batches (self.X.T ,self.Y.T, self.kargs['mini_batch_size'])
            cost = 0
            for (X,Y) in self.minibatches:
                self.X = X.T
                self.Y = Y.T
                self.m = self.X.shape [0]
                prediction = self._predict ()
                assert prediction.shape == self.Y.shape
                cost += self._costFunction (prediction)
                vgrads = self._calculateGrads (prediction)
                self._updateThetas (vgrads,lr)
            self.costs.append (cost)

    
    def _calculateBatchGradientDescent (self):
        self.costs = []
        lr = self.kargs ['learning_rate']
            
        for i in tqdm(range (self.kargs ['steps']),desc='LR = ' + str(lr)):
            prediction = self._predict ()
            assert prediction.shape == self.Y.shape
            cost = self._costFunction (prediction)
            self.costs.append (cost)
            vgrads = self._calculateGrads (prediction)
            self._updateThetas (vgrads,lr)
      
    def _costFunction (self,prediction):
        assert prediction.shape[1] == 1
        assert self.Y.shape[1] == 1
        assert prediction.shape[0] == self.Y.shape[0]

        epsilon = 1e-10   # Incluido para evitar divisiones por 0
        regularization = (self.kargs['lambda'] /(2*self.m)) * np.sum (self.vtheta[0][1:]**2)
        #print ('CostFunction, regularization',regularization)
        #print ('CostFunction, prediction',prediction[0:5])
        J = - ( 1 / self.m) * ( np.dot(self.Y.T, np.log(prediction+epsilon)) + np.dot((1-self.Y).T, np.log(1-prediction+epsilon))) + regularization
        J = np.squeeze (J)
        assert J.shape == ()
        return J

    


    def _calculateGrads (self,prediction):

        assert prediction.shape == self.Y.shape
        error = prediction - self.Y
        error = np.array (error).reshape(1, self.m)
        regularization = (self.kargs['lambda'] / self.m ) * self.vtheta
        regularization [0] [0]= 0    #Theta 0 no tiene regularization
        grads1 =  (1/self.m) * np.dot ((prediction-self.Y).T,self.X) + regularization
        grads2 =  ( 1 / self.m ) * error.dot (self.X) + regularization
        return grads1
    

    def _updateThetas (self,vgrads,learning_rate=1e-2):
        self.vtheta = self.vtheta -  learning_rate * vgrads
