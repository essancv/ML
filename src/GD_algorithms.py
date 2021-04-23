import numpy as np

class OptimizationAlgorithm:
    def __init__ (self, layers , kargs={'learning_rate':1e-4}):
        self.layers = layers
        self.lr = kargs ['learning_rate']
    def getLR (self):
        return self.lr
    def optimize (self,iteration):
        raise NotImplementedError
        
class LRDecay (OptimizationAlgorithm):
    def __init__(self,layers,kargs={'learning_rate':0.2 , 'decay_rate':1.0}):
        OptimizationAlgorithm.__init__ (self,layers,kargs=kargs)
        self.decay_rate = kargs ['decay_rate']
    def optimize (self,iteration):
        lr = self.lr / (1 + (self.decay_rate * iteration))
        for layer in self.layers:
            W,b   = layer.getParameters ()
            dW,db = layer.getGradients ()
            Wopt = W - lr * dW
            bopt = b - lr * db
            layer.setParameters (Wopt,bopt)

        
class GradientDescent (OptimizationAlgorithm):
    def __init__(self,layers,kargs={'learning_rate':1e-4}):
        OptimizationAlgorithm.__init__ (self,layers,kargs=kargs)
    def optimize (self,iteration):
        for layer in self.layers:
            W,b   = layer.getParameters ()
            dW,db = layer.getGradients ()
            Wopt = W - self.lr * dW
            bopt = b - self.lr * db
            layer.setParameters (Wopt,bopt)
    
class Momentum (OptimizationAlgorithm):
    def __init__(self,layers,kargs={'learning_rate':1e-4,'beta':0.9}):
        OptimizationAlgorithm.__init__ (self,layers,kargs=kargs)
        self.Vdw = {}
        self.Vdb = {}
        for layer in self.layers:
            Wshape, bshape = layer.getParametersShape ()
            self.Vdw [layer] = np.zeros(Wshape)
            self.Vdb [layer]=  np.zeros(bshape)     
        self.beta = kargs ['beta']
        
    def optimize (self,iteration):
        for layer in self.layers:
            W,b   = layer.getParameters ()
            dW,db = layer.getGradients ()
            self.Vdw [layer] = self.beta * self.Vdw [layer] + (1-self.beta) * dW
            self.Vdb [layer] = self.beta * self.Vdb [layer] + (1-self.beta) * db
            Wopt = W - self.lr * self.Vdw [layer]
            bopt = b - self.lr * self.Vdb [layer]
            layer.setParameters (Wopt,bopt)

class RMSprop (OptimizationAlgorithm):
    def __init__(self,layers,kargs={'learning_rate':1e-4,'beta':0.9}):
        OptimizationAlgorithm.__init__ (self,layers,kargs=kargs)
        self.Sdw = {}
        self.Sdb = {}
        for layer in self.layers:
            Wshape, bshape = layer.getParametersShape ()
            self.Sdw [layer] =  np.zeros(Wshape) 
            self.Sdb [layer] =  np.zeros(bshape)         
        self.beta = kargs ['beta']
        
    def optimize (self,iteration,epsilon=1e-8):
        for layer in self.layers:
            W,b   = layer.getParameters ()
            dW,db = layer.getGradients ()
            correction = 1 - np.power (self.beta,iteration)          
            self.Sdw [layer] = self.beta * self.Sdw [layer] + ((1-self.beta) * np.square (dW ))
            self.Sdb [layer] = self.beta * self.Sdb[layer] + ((1-self.beta) * np.square (db ))
            Sdw_corrected  = self.Sdw [layer] / correction
            Sdb_corrected  = self.Sdb [layer] / correction
            Wopt = W - self.lr * dW / (np.sqrt (Sdw_corrected) + epsilon)
            bopt = b - self.lr * db / (np.sqrt (Sdb_corrected) + epsilon)
            layer.setParameters (Wopt,bopt)

class Adam (OptimizationAlgorithm):
    def __init__(self,layers,kargs={'learning_rate':1e-4,'beta1':0.9,'beta2':0.999}):
        OptimizationAlgorithm.__init__ (self,layers,kargs=kargs)
        self.Vdw = {}
        self.Vdb = {}
        self.Sdw = {}
        self.Sdb = {}
        for layer in self.layers:
            Wshape, bshape = layer.getParametersShape ()
            self.Vdw [layer] = np.zeros(Wshape) 
            self.Vdb [layer] = np.zeros(bshape)             
            self.Sdw [layer] = np.zeros(Wshape)  
            self.Sdb [layer] = np.zeros(bshape)             
        self.beta1 = kargs ['beta1']
        self.beta2 = kargs ['beta2']
        
    def optimize (self,iteration,epsilon=1e-8):
        for layer in self.layers:
            W,b   = layer.getParameters ()
            dW,db = layer.getGradients ()
            correction1 = 1 - np.power (self.beta1,iteration)          
            correction2 = 1 - np.power (self.beta2,iteration)          
            W,b   = layer.getParameters ()
            dW,db = layer.getGradients ()
            self.Vdw [layer]= self.beta1 * self.Vdw [layer] + (1-self.beta1) * dW
            self.Vdb [layer]= self.beta1 * self.Vdb [layer] + (1-self.beta1) * db
            self.Sdw [layer]= self.beta2 * self.Sdw [layer] + ((1-self.beta2) * np.square (dW ))
            self.Sdb [layer]= self.beta2 * self.Sdb [layer] + ((1-self.beta2) * np.square (db ))
            Vdw_corrected  = self.Vdw [layer]/ correction1
            Vdb_corrected  = self.Vdb [layer]/ correction1
            Sdw_corrected  = self.Sdw [layer]/ correction2
            Sdb_corrected  = self.Sdb [layer]/ correction2
            Wopt = W - self.lr * Vdw_corrected / (np.sqrt (Sdw_corrected) + epsilon)
            bopt = b - self.lr * Vdb_corrected / (np.sqrt (Sdb_corrected) + epsilon)
            layer.setParameters (Wopt,bopt)