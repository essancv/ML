import numpy as np
import sys
sys.path.append ('../src') 
from utils import UTIL_FeatureNormalization,UTIL_initVTheta,UTIL_getXMatrix,UTIL_random_mini_batches
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

class LinearRegression:
    def __init__ (self,optimizer="GD" ,normalization=True, minibatch=False,lr_lambda=0.0,kargs={'lambda':0.0,'mini_batch_size':2**10}):
        assert optimizer in ["GD","Normal"],"Possible optimizers are 'GD,Normal'"
        self.optimizer  = optimizer
        self.lr_lambda  = lr_lambda
        self.normalization = normalization
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
        """
        assert X.shape [0] == Y.shape[0] , "Revisar dimensiones X e Y"
        self.m = X.shape [0]   # Número de muestras
        self.n = X.shape [1]   # Número de features (sin incluir 1's)
        if self.normalization:
            self.X = UTIL_FeatureNormalization (pd.DataFrame(X),debug=False).values
        else:
            self.X = np.array (X)
            
        # Añado columna de 1's
        self.X = UTIL_getXMatrix (self.X)

        self.Y = np.copy (Y)
        """
        if self.optimizer == 'Normal':
            self._calculateNormalEquation ()
            assert self.vtheta.shape [1] == self.n + 1
            assert self.vtheta.shape [0] == 1
        else:
            self.vtheta = UTIL_initVTheta (self.n, type=self.kargs['theta_init'])
            assert self.X.shape[1] == self.vtheta.shape[1] , "Dimensiones X y vtheta no coinciden"
            assert self.vtheta.shape [0] == 1
            if self.minibatch:
                self._calculateMiniBatchGradientDescent ()
            else:
                self._calculateBatchGradientDescent ()
                
        if self.minibatch :  # Vuelvo a valores originales
            self._initializeData (X,Y)
            
        prediction = self._predict ()

        print ('Mean Squared Error: %.2f' % mean_squared_error(self.Y,prediction))
        print ('Coefficient of determination: %.2f' % r2_score(self.Y,prediction))
        
    def _calculateNormalEquation (self):
            step1 = np.dot(self.X.T,self.X)
            step2 = np.linalg.inv(step1)
            step3 = np.dot (step2,self.X.T)
            self.vtheta = np.dot(step3,self.Y).reshape (1,self.X.shape[1])

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
        Predicción lineal,
        X es la matriz con las muestras .Dimensión  (m,n) donde m es el número de muestras y n el número de features
        theta es el vector de Thetas [tHeta0,theta1,theta2 ...]. Dimensión (1,n), siendo n el número de features
        """
        #print ('_predict , X ', self.X [0:5])
        #print ('_predict , vtheta ', self.vtheta)
        
        assert self.vtheta.shape[0] == 1, "Revise dimensión theta (1,n)"
        
        return np.dot (self.X,self.vtheta.T)


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
        m = prediction.shape [0]
        assert prediction.shape[1] == 1
        assert self.Y.shape[1] == 1
        assert prediction.shape[0] == self.Y.shape[0]
        regularization = (self.kargs['lambda'] / (2*m) ) * np.sum (self.vtheta[0][1:]**2)
        #print ('_costFunction,regularization' , regularization)
        #print ('_costFunction,prediction' , prediction[0:5])
        #print ('_costFunction,Y' , self.Y[0:5])
        J = ( 1/ (2 * m) *  np.sum (np.power ((prediction-self.Y),2)) )+ regularization 
        #print ('_costFunction, costs',J)
        return J

    


    def _calculateGrads (self,prediction):
        assert prediction.shape == self.Y.shape
        regularization = (self.kargs['lambda'] / self.m ) * self.vtheta
        regularization [0] [0] = 0    #Theta 0 no tiene regularization
        grads =  (1/self.m) * np.dot ((prediction-self.Y).T,self.X) + regularization
        return grads

    

    def _updateThetas (self,vgrads,learning_rate=1e-2):
        self.vtheta = self.vtheta -  learning_rate * vgrads