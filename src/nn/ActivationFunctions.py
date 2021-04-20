import numpy as np

class  Activation:
    def __init__ (self):
        self.cache = {}
        
    def forward (self, X):
        raise NotImplementedError
    def backward (self, X , cached_y=None):
        raise NotImplementedError
        

class Sigmoid (Activation):
    def __init__ (self):
        super().__init__()
        
    def forward (self,Z):
        A = 1 / ( 1 + np.exp (-Z))
        self.cache = Z
        return A
                 
    def backward (self, dA):
        # dZ = dA * g'(Z) donde g'(Z) = s * (1-s)
        Z = self.cache
        s =  1 /  ( 1 + np.exp (-Z) )
        dZ = dA * s * ( 1 - s )
        
        assert (dZ.shape == Z.shape)
        return dZ