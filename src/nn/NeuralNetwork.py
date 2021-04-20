import numpy as np
from functools import reduce

class NeuralNetwork:
    
    def __init__( self , input_dim, layers,costF,gradientF,l2_lambda=0.0 , debug=False):
        self.layers = layers
        self.l2_lambda = l2_lambda
        self.costF = costF
        self.gradientF = gradientF
        self.debug = debug
        self.W_grads = {}
        self.b_grads = {}
        
        # Inicializo los layers con las dimensiones de entrada y salida
        
        self.layers [0].init (input_dim)
        for (previous_layer,current_layer) in zip (self.layers, self.layers[1:]):
            previous_layer_input_size, previous_layer_output_size  = previous_layer.get_dimension ()
            current_layer.init ( previous_layer_output_size )
        

    def forward_prop (self, X , training=False):
        a = X
        for layer in self.layers:
            a = layer.forward (a , training )
        
        return a
    
    def backward_prop (self, A_last , Y):
        dA = self.gradientF (A_last, Y )
        m = A_last.shape [0]
        layerid = len (self.layers)
        grads = {}
        for layer in reversed (self.layers):
            dA_prev,dW , db = layer.backward (dA,Y)
            self.W_grads [layer] = dW
            self.b_grads [layer] = db 
            
            if self.l2_lambda != 0.0:
                # Añadimos el término de regularization
                self.W_grads [layer] += ( self.l2_lambda / m ) * layer.get_params ()[0]

            grads ['grad'+str(layerid)] = {'W':self.W_grads [layer] , 'b':self.b_grads [layer]  }
            layerid -= 1

            dA = dA_prev
            
        return grads
                                               
    def costFunction (self, A_last, Y ):
        cost = self.costF (A_last, Y )
        if self.l2_lambda != 0.0:
            # Calculamos regularization
            m = Y.shape [0]  # Número de muestras
            weigths = [layer.get_params()[0] for layer in self.layers]
            l2_cost = (self.l2_lambda / ( 2 * m ) ) * reduce (lambda ws, w: ws + np.sum(np.square(w)),weigths,0)
            cost += l2_cost
            if self.debug :
                print ('NN - costFunction, regularization value {} , with lambda {}'.format(l2_cost,self.l2_lambda))
        return cost
                                                              