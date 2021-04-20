import numpy as np

epsilon = 1e-20
def NN_CostFunction (A_last,Y):
    assert A_last.shape == Y.shape , "Revisar shapes A_last, Y "
    m = A_last.shape [0]
    K = A_last.shape [1]  # Identificar si es multiclass ( > 1)
    epsilon = 1e-20   # Incluido para evitar divisiones por 0
    J = 0
    for i in range (K):
        Yk = Y[: , i:i+1]
        A_lastk = A_last [:,i:i+1]
        J += np.dot(Yk.T, np.log(A_lastk+epsilon)) + np.dot((1-Yk).T, np.log(1-A_lastk+epsilon))
        #print (Yk[0:5])
        #print (A_lastk [0:5])
    J *= - ( 1 / m )
    J = np.squeeze (J)

    assert J.shape == ()

    return J

def NN_Gradient (A_last,Y):
    assert A_last.shape == Y.shape , "Revisar shapes A_last, Y "
    return -np.divide(Y,np.clip(A_last,epsilon,1.0))
