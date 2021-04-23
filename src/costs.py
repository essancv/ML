import numpy as np

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
