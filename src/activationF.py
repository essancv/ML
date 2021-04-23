import numpy as np

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