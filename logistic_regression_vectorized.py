import numpy as np
import math

def sigmoid(z):
    return 1/(1+np.exp(-z))


def compute_cost(X, y, w, b, lambda_=0):
    m = X.shape[0]
    z = np.dot(X, w) + b # (m,)
    prediction = sigmoid(z) # (m,)
    losses = (-y * np.log(prediction)) - ((1 - y) * np.log(1 - prediction))  #(m,)
    cost = np.mean(losses) # scalar
    cost += (np.sum(w**2) * lambda_ / (2*m)) # scalar

    return cost


def compute_gradients(X, y, w, b, lambda_=0):
    m = X.shape[0]
    z = np.dot(X, w) + b # (m,)
    prediction = sigmoid(z) # (m,)
    error = (prediction - y) # (m,)
    dj_dw = np.dot(error, X) / m # (n,)
    dj_dw += (w * lambda_ / m) # (n,)

    dj_db = np.mean(error) # scalar
    
    return dj_dw, dj_db


def run_gradient_descent(self, X, y, w, b, alpha=0.01, num_iter=10, lambda_=0, verbose=False):
    """
    Ejecuta Batch Gradient Descent para entrenar el modelo de regresión lineal.
        
    Args:
        - X (ndarray (m,n)): dataset. 
        - y (ndarray (m,)): etiqueta o salida verdadera.
        - w (ndarray (n,)): pesos o weights.
        - b (float): sesgo o bias.
        - alpha (float): learning rate.
        - num_iter (int): número de iteraciones o epochs del gradient descent.
        - verbose (bool): True para imprimir el costo mientras se ejecuta el algoritmo.
        Nota: 'm' es el número de ejemplos y 'n' es el número de features.    
        
    Returns:
        - w (ndarray (n,)): pesos o weights finales.
        - b (int): sesgo o bias final.
        - history_cost (ndarray (num_iter,)): historial del costo obtenido al evaluar el modelo.
        - history_params (ndarray (num_iter,n+1)): historial de parámetros del modelo calculados 
                                                   por el algoritmo.
    """

    history_cost = np.zeros((num_iter,))
    history_params = np.zeros((num_iter, w.shape[0] + 1))
        
    for i in range(num_iter):
        # gradients
        dj_dw, dj_db = self.__compute_gradients(X, y, w, b)
            
        # model params update
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # compute cost
        cost = self.compute_cost(X, y, w, b)
        
        # append cost and params
        history_cost[i] = cost
        history_params[i] = np.concatenate((w, np.array([b])))
            
        # print
        if verbose:
            if i% math.ceil(num_iter/10) == 0:
                print(f"Costo hasta la iteración {i}: {cost}")
        
    self.__weights = w
    self.__bias = b
    return w, b, history_cost, history_params    

