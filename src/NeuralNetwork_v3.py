import numpy as np
from ML_utils import UTIL_random_mini_batches
from tqdm import tqdm
from pandas import DataFrame
import math
from GD_algorithms import GradientDescent
from costs import  BinaryCrossEntropy, CategoricalCrossEntropy

class NNClassifier_V3:
    def __init__(self,layers,costFunction=BinaryCrossEntropy,normalization=False,optimizationF=GradientDescent,kargs={'steps':100,'learning_rate':0.01},print_cost=True):
        self.layers = layers
        self.costF = costFunction()
        self.kargs = kargs
        self.normalization = normalization
        self.optimizationF = optimizationF (layers,kargs=kargs)
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

    def __WDCost (self, costshistory,limit=10):
        try:
            coststoverify = costshistory[-limit:]
            if len (coststoverify) == limit:
                consecutiveincrs = [coststoverify[i:i+1] <= coststoverify[i+1:i+2] for i in range (limit-1)]
                assert not all (consecutiveincrs), 'No parece que GD esté convergiendo revise sus parámetros'        
        except:
           raise     
        
    def __batchGD (self, X,Y, l2_lambda=0.0,training=True):
        costs = []
        prev_cost = 0
        not_convergency_counter = 0
        
        for i in tqdm(range (self.kargs ['steps']),desc='LR = ' + str(self.optimizationF.getLR())):
            cost = self.__oneStepTraining (X,Y,i,l2_lambda=l2_lambda,training=training)
            costs.append (cost)
            self.__WDCost (costs)
        return costs 

     
    def __miniBatchGD (self, X,Y, l2_lambda=0.0,training=True):
        costs = []
        prev_cost = 0
        not_convergency_counter = 0

        for i in tqdm(range (self.kargs ['steps']),desc='LR = ' + str(self.optimizationF.getLR())):
            minibatches = UTIL_random_mini_batches (X ,Y, self.kargs['mini_batch_size'])
            for (miniBatchX,miniBatchY) in minibatches:
                cost = self.__oneStepTraining (miniBatchX,miniBatchY,i,l2_lambda=l2_lambda,training=training)
                costs.append (cost)
                self.__WDCost (costs)
        return costs  

    def __oneStepTraining (self,X,Y,iteration,l2_lambda=0.0,training=True):
        prediction = self.__forward (X,training)
        assert prediction.shape == Y.shape
        cost = self.costFunction (Y,prediction,l2_lambda)
        assert not math.isnan(cost) , "Cost es NaN , revise los parámetros"
        self.__backward (prediction,Y,training)
        self.optimizationF.optimize (iteration+1)
        return cost
            
    def __backward (self, A_last,Y , training=True):
        Y = Y.reshape(A_last.shape)
        # derivative of cost with respect to AL
        dA_last = self.costF.df ( Y, A_last)
        for layer in reversed(self.layers):
            dA_prev, dW, db = layer.backward (dA_last,training)
            dA_last = dA_prev
        
    def train ( self,X,Y,l2_lambda=0.0,method='BatchGD'):            
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
