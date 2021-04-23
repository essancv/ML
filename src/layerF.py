import numpy as np
from activationF import SigmoidF,SoftmaxF,TanhF,ReluF,LReluF

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

class Activation:    
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
        self.activation = Activation (function=activationF)
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