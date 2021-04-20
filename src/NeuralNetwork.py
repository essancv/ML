from functools import reduce
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from ML_utils import sigmoid, softmax, NN_CostFunction, formatY,norm, CategoricalCrossEntropy,UTIL_random_mini_batches

epsilon = 1e-20


costs = []


class NNClassifier:
    def __init__( self , optimization='Optimize', costFunction=CategoricalCrossEntropy, bias=True, nn_config={'n_a1':400,'n_a2':25,'n_a3':10 },activ={'activation_a2':sigmoid,'activation_a2':softmax},debug=False,kargs=None):
        self.nn_config = nn_config  # {'n_a1, n_a2, n_a3 ....} número de nodos en cada layer
        self.n_layers = len (nn_config)    # Layers 1, 2 , 3
        self.thetas = None                 # Thetas 1 para layer 2 , Theta 2 para layer 3 ....
        self.debug = debug
        self.optimization = optimization
        self.costF = costFunction
        self.activ = activ
        self.bias = bias
        self.kargs=kargs
        self._initThetas ()
        
    def _initThetas (self):
        """
        episilon = np.sqrt(6)/np.sqrt(Lin + Lout)
        Lin = the number of input layer unit
        Lout = the number of the adjacent layer unit
        """
        self.thetas = {}
        episilon = 0.12
        for i in range(1,self.n_layers):
            input_layer_size = self.nn_config['n_a' + str(i)]
            number_of_nodes = self.nn_config['n_a' + str(i+1)]
            if self.bias:
                self.thetas ['Theta'+str(i)] = np.random.rand(number_of_nodes,input_layer_size + 1) * 2.0 * episilon - episilon
            else:
                self.thetas ['Theta'+str(i)] = np.random.rand(number_of_nodes,input_layer_size) * 2.0 * episilon - episilon
#            self.thetas ['Theta'+str(i)] = np.random.rand(number_of_nodes,input_layer_size + 1) 
 
    def Cost_wrapper (self,thetas,X,Y,l2_lambda=0.0):
        self._setThetaFromVector (thetas)
        J =  self.costFunction ( X , Y , l2_lambda)
        global costs
        costs.append (J)
        if self.debug:
            print ('NNClassifier - Cost_wrapper, cost : ', J)
        return J
    
    
    def Grads_wrapper (self,thetas,X,Y,l2_lambda):
        self._setThetaFromVector (thetas)
        delta,grads = self.backward_prop (X,Y,l2_lambda)
        return self._getGradVector (grads)

    def _insertBias (self,X):
        if self.bias:
            if X.shape [1] == (self.nn_config ['n_a1'] + 1):
                return X
            else:
                return np.insert (X,0,1,axis=1)
        else:
            return X
        
    def forward_prop (self,X):
        X = self._insertBias (np.copy(X))
        ai = X 
        cache = {'a1':X}
        
        for i in range(2,self.n_layers + 1):
            zi = np.dot (ai , self.thetas['Theta'+str(i-1)].T )
#            ai = sigmoid (zi)
            ai = self.activ ['activation_a'+str(i)](zi)
            cache ['z'+str(i)] = zi
            if ( i != self.n_layers ) and self.bias:
                ai = np.insert (ai,0,1, axis=1)
            cache ['a'+str(i)] = ai

        return ai , cache

    def backward_prop (self,X,Y,l2_lambda=0.0):
        X = self._insertBias (np.copy(X))
        m = X.shape [0]
        prediction,cache = self.forward_prop (X)
        prediction = np.array (prediction)
        delta = {}
        deltai = prediction - Y
        delta ['delta'+str(self.n_layers) ] = deltai 
        for i in reversed (range(2,self.n_layers)):
            ai = cache['a'+str(i)]
            if self.bias:
                ai = np.delete (ai,0,axis=1)  # Elimino los 1's que no se tienen en cuenta en el cálculo de deltas
            gprima_z = ai * ( 1 - ai )
            if self.bias:
                step1 = np.dot(deltai,self.thetas['Theta' + str(i)][:,1:])
            else:
                step1 = np.dot(deltai,self.thetas['Theta' + str(i)][:,:])
            deltaiprev = step1 * gprima_z
            delta ['delta'+str(i) ] = deltaiprev 
            deltai = deltaiprev

        grads = {}
        for i in reversed(range (1, self.n_layers ) ):
            if self.debug:
                print ('Calculating grad2 ' , i)
            grad = np.dot (delta['delta'+str(i+1)].T , cache['a'+str(i)] ) / m
            if self.bias:
                grad [:,1:] = grad [:,1:] + (l2_lambda * self.thetas['Theta' + str(i)][:,1:] / m )
            else:
                grad [:,:] = grad [:,:] + (l2_lambda * self.thetas['Theta' + str(i)][:,:] / m )
            grads ['grad'+str(i)] = grad
            if self.debug:
                print ('Grad {} shapes delta{} {} a{} {} ,theta {}{}, grad shape {} ' .format( i,(i+1),delta['delta'+str(i+1)].shape, i,cache['a'+str(i)].shape,i,self.thetas['Theta' + str(i)][:,1:].shape ,grad.shape)) 
        return delta , grads
                                    
    def costFunction ( self, X , Y , l2_lambda=0.0 ):
        X = self._insertBias (np.copy(X))
        A_last,cache = self.forward_prop (X)
#        cost = NN_CostFunction (A_last, Y )
        cost = self.costF (Y,A_last)
        if l2_lambda != 0.0:
            # Calculamos regularization
            m = Y.shape [0]  # Número de muestras
            Thetas = [self.thetas['Theta'+str(i)][:,1:] for i in range(1,self.n_layers) ]  # No selecciono el termino theta de bias
            l2_cost = (l2_lambda / ( 2 * m ) ) * reduce (lambda ws, w: ws + np.sum(np.square(w)),Thetas,0)
            cost += l2_cost
            if self.debug:
                print ('costFunction, regularization value {} , with lambda {}'.format(l2_cost,l2_lambda))

        return cost
    
    def _stochasticGD (self, X,Y, l2_lambda=0.0):
        costs = []
        prev_cost = 0
        not_convergency_counter = 0
        lr = self.kargs ['learning_rate']
        for i in tqdm(range (self.kargs ['steps']),desc='LR = ' + str(lr)):
            prediction,cache = self.forward_prop (X)
            assert prediction.shape == Y.shape
            cost = self.costFunction (X,Y,l2_lambda)
            if cost >= prev_cost:
                not_convergency_counter += 1
                prev_cost = cost
            else:
                not_convergency_counter = 0
                prev_cost = cost

            assert not_convergency_counter <= 10 , 'No parece que SGD esté convergiendo revisa alpha ' + str(lr) + ',cost ' + str(cost) + 'iteration ' + str(i)

            costs.append (cost)
            delta,grads= self.backward_prop (X,Y,l2_lambda)

            self._updateThetas (grads,lr)
        return costs 

    def _miniBatchGD (self, X,Y, l2_lambda=0.0):
        costs = []
        prev_cost = 0
        not_convergency_counter = 0
        lr = self.kargs ['learning_rate']
        for i in tqdm(range (self.kargs ['steps']),desc='LR = ' + str(lr)):
            minibatches = UTIL_random_mini_batches (X.T ,Y.T, self.kargs['mini_batch_size'])
            for (miniBatchX,miniBatchY) in minibatches:
                prediction,cache = self.forward_prop (miniBatchX.T)
                assert prediction.shape == miniBatchY.T.shape
                cost = self.costFunction (miniBatchX.T,miniBatchY.T,l2_lambda)
                if cost >= prev_cost:
                    not_convergency_counter += 1
                    prev_cost = cost
                else:
                    not_convergency_counter = 0
                    prev_cost = cost

                assert not_convergency_counter <= 10 , 'No parece que SGD esté convergiendo revisa alpha ' + str(lr) + ',cost ' + str(cost) + 'iteration ' + str(i)

                costs.append (cost)
                delta,grads= self.backward_prop (miniBatchX.T,miniBatchY.T,l2_lambda)

                self._updateThetas (grads,lr)
        return costs     

    def optimize ( self,X,Y,l2_lambda=0.0 ):
        X = self._insertBias (np.copy(X))
        if self.optimization == "Optimize":
            self._optimize (X,Y,l2_lambda=0.0)
            global costs
            return costs
        elif self.optimization == "SGD":
            return self._stochasticGD (X,Y,l2_lambda)
        elif self.optimization == "miniBatchGD":
            return self._miniBatchGD (X,Y,l2_lambda)
        else:
            raise NotImplementedError ("Valid methods are :'SGD','Optimize','miniBatchGD'")
            
    def _updateThetas (self,grads,lr=0.001):
        for i in range(1,len(self.thetas)+1):
            theta = "Theta"+str(i)
            grad = "grad"+str(i)
            self.thetas[theta] = self.thetas[theta] -  lr * grads[grad]
        pass
    
    def _setThetaFromVector (self,vtheta):
        index = 0
        for i in range(1,len(self.thetas)+1):
            theta = "Theta"+str(i)
            shape = self.thetas[theta].shape[0] * self.thetas[theta].shape[1]
            self.thetas [theta] = vtheta[index:index+shape].reshape (self.thetas[theta].shape[0],self.thetas[theta].shape[1])
            index += shape

    def _getGradVector (self,graddict):
        concatgrad = np.array([[]])
        for i in range(1,len(self.thetas)+1):
            grad = "grad"+str(i)
            shape = graddict[grad].shape[0] * graddict[grad].shape[1]
            concatgrad = np.concatenate ((concatgrad,graddict[grad].reshape(1,shape)),axis=None)
        return concatgrad

    def _getThetaVector (self):
        concattheta = np.array([[]])
        for i in range(1,len(self.thetas)+1):
            theta = "Theta"+str(i)
            shape = self.thetas[theta].shape[0] * self.thetas[theta].shape[1]
            concattheta = np.concatenate ((concattheta,self.thetas[theta].reshape(1,shape)),axis=None)
        return concattheta
    
    def _optimize (self,X,Y,l2_lambda=0.0):
        vtheta = self._getThetaVector ()
        vtheta = minimize(fun=self.Cost_wrapper , x0=vtheta , method=self.kargs['algorithm'], jac=self.Grads_wrapper,args=(X, Y,l2_lambda),options = {'maxiter':self.kargs['maxiter'],'disp':True})
        self._setThetaFromVector (vtheta.x)
        if self.debug :
            print ('NueralNetwork - _optimize, success {} , message {}'.format(vtheta.success,vtheta.message))

    def _computeNumericalGradient (self,X,Y,l2_lambda=0.0):
        epsilon = 1e-4
        ngrads = {}
        for i in range(1,len(self.thetas)+1):
            n_nodes,n_outputs = self.thetas['Theta'+str(i)].shape
            ngrad = np.zeros(self.thetas['Theta'+str(i)].shape)
            for node in range (n_nodes):
                for output in range(n_outputs):
                    theta_bck =  self.thetas['Theta'+str(i)] [node][output]
                    #theta_plus  
                    self.thetas['Theta'+str(i)] [node][output] = theta_bck + epsilon
                    cost_theta_plus = self.costFunction(X,Y,l2_lambda) 
                    #theta_minus  
                    self.thetas['Theta'+str(i)] [node][output] = theta_bck - epsilon
                    cost_theta_minus = self.costFunction(X,Y,l2_lambda) 
                    ngrad [node][output] = (cost_theta_plus - cost_theta_minus ) / (2*epsilon)
#                    print ( 'Cost theta plus {}, minus {} ngrad {}'.format(cost_theta_plus,cost_theta_minus,ngrad[node][output]) )
                    # vuelvo theta su valor inicial
                    self.thetas['Theta'+str(i)] [node][output] = theta_bck
                ngrads ['grad'+str(i)] = ngrad
                
        return ngrads

    def checkGradient (self,m,l2_lambda=0.0):
        X = np.random.rand(m,self.nn_config['n_a1'])
        if self.bias:
            X = np.insert (X,0,1,axis=1)
        Y = (np.arange(m) % self.nn_config['n_a'+str(len(self.nn_config))]).reshape(m,1)
        fY = formatY (Y,num_labels=self.nn_config['n_a'+str(len(self.nn_config))])
        ngrads = self._computeNumericalGradient(X,fY,l2_lambda)
        delta,grads = self.backward_prop ( X, fY, l2_lambda )
        concatgrads = np.array([[]])
        concatnumgrad = np.array([[]])
        for i in range(1,len(self.thetas)+1):
            gradiente = "grad"+str(i)
            shape1 = grads[gradiente].shape[0] * grads[gradiente].shape[1]
            shape2 = ngrads[gradiente].shape[0] * ngrads[gradiente].shape[1]
            concatgrads = np.concatenate ((concatgrads,grads[gradiente].reshape(1,shape1)),axis=None)
            concatnumgrad = np.concatenate ((concatnumgrad,ngrads[gradiente]),axis=None)
        print ("Grad differences %.15f" % (norm (concatnumgrad.flatten() - concatgrads.flatten()) / norm(concatnumgrad.flatten() + concatgrads.flatten())))