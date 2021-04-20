import numpy as np
import sys

class LayerError(Exception):
    def __init__(self, message):
        self.message = message

class Layer:
    def __init__ (self,nodes_in_layer,activation, name , output_layer=False,debug=False ,kargs=None):
        self.nodes_in_layer = nodes_in_layer
        self.name = name
        self.activation = activation
        self.debug = debug
        self.input_dimension = None
        self.kargs=kargs
        self.output_layer = output_layer
        self.W = None
        self.b = None
        
    def init (self, input_dimension):
        raise NotImplementedError
    
    def get_name (self):
        return self.name
    
    def forward (self, A_prev):
        raise NotImplementedError
        
    def backward (self, dA):
        raise NotImplementedError
        
    def update_params (self, dW, db):
        raise NotImplementedError

    def get_params (self):
        return self.W , self.b
    
    def get_dimension (self):
        raise NotImplementedError
 

class FullyConnected (Layer):
    def __init__ (self, nodes_in_layer,activation, name ,output_layer=False, debug=False,kargs=None):
        super().__init__(nodes_in_layer,activation, name , output_layer=output_layer,debug=debug,kargs=kargs )
        self.cache = {}
        
    def init (self,input_dimension):
        if self.debug:
            print ('FC - {} - init , initialization with input dimension {} ' .format (self.name,input_dimension ))
        self.input_dimension = input_dimension

        try:
            if self.debug:
                print ('FC - {} - init , initialization forzada ' )
            self.W = self.kargs['W']
            self.b = self.kargs['b'].reshape(1,self.nodes_in_layer)
            assert self.W.shape == (self.nodes_in_layer , input_dimension),"Inizializacion forzada inválida,W"
            assert self.b.shape == (1 , self.nodes_in_layer),"Inizializacion forzada inválida,b"
        except:
            if self.debug:
                print ('FC - {} - init , Excepción, init random  ')
                print("Unexpected error:", sys.exc_info()[0])
            self.W = np.random.randn ( self.nodes_in_layer , input_dimension ) / np.sqrt (input_dimension)
            self.b = np.zeros ((1,self.nodes_in_layer))
    
    def forward (self, A_prev , training=False):
        if self.debug:
            print ('FC - {} - forward , starting forward prop' .format (self.name))
 
        assert A_prev.shape[1] == self.W.shape[1]  # Número de features de entrada
        
        Z = np.dot (A_prev, self.W.T ) + self.b
        A = self.activation.forward (Z)
        
        if self.debug:
            print ('FC - {} - forward , Shapes: A_prev {}, Z {} , A {} ' .format (self.name, A_prev.shape,Z.shape,A.shape ))
        
        if training:
            self.cache.update ({'A_prev':A_prev, 'Z':Z,'A':A})
        return A

    def backward (self, dA , Y):
        if self.debug:
            print ('FC - {} - backward , starting backward prop' .format (self.name))

        A_prev, Z , A = (self.cache[key] for key in ('A_prev', 'Z','A'))
        m = dA.shape[0]
        
        if self.output_layer:
             dZ = A - Y
        else:          
            # Derivada de la activación
            dZ = self.activation.backward (dA)
            
        # Derivada de la función lineal
        dW = ( 1 / m ) * np.dot ( dZ.T , A_prev )
        db = ( 1 / m ) * np.sum ( dZ , axis=0, keepdims = True )
        dA_prev = np.dot ( dZ , self.W)
        
        assert dA_prev.shape == A_prev.shape
        assert dW.shape == self.W.shape
        assert db.shape == self.b.shape
        return dA_prev, dW, db
    
    def get_dimension (self):
        if self.input_dimension == None:
            raise LayerError ('Layer not initilized, call init method before get_dimension ')
            
        return self.input_dimension,self.nodes_in_layer
                  
