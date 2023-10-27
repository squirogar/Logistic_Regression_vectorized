import numpy as np
import math

class LogisticRegressionVectorized():
    """
    Clase que crea un modelo de logistic regression. Este modelo se entrena mediante 
    Batch Gradient Descent. Todas las operaciones de cálculo son realizadas con
    vectorización. Este modelo soporta regularización L2. Además, se proporciona un 
    método para la normalización de features que se utilizarán para el modelo.
    """
    def __init__(self):
        self.__weights = None
        self.__bias = None
        self.__mu = None
        self.__sigma = None

    def get_weights(self):
        """
        Retorna un ndarray (n,) con los pesos del modelo de logistic regression. 
        "n" es el número de features. 
        Si no se ejecuta previamente el gradient descent, retornará None.
        """
        return self.__weights


    def get_bias(self):
        """
        Retorna el sesgo del modelo de logistic regression. 
        Si no se ejecuta previamente el gradient descent, retornará None.
        """        
        return self.__bias

    def get_mean(self):
        """
        Retorna un ndarray (n,) con las medias de las features a utilizar en el
        modelo.
        Si no se ejecuta previamente el normalización de features, retornará None.
        """
        return self.__mu

    def get_std(self):
        """
        Retorna un ndarray (n,) con las desviaciones estándar de las features a 
        utilizar en el modelo.
        Si no se ejecuta previamente la normalización de features, retornará None.
        """        
        return self.__sigma


    def __sigmoid(self, z):
        """
        Aplica la logistic function retornando un valor float entre 0 y 1 dada una 
        entrada.
        
        Args:
            - z (ndarray | float): número o ndarray a aplicarle la logistic function.

        """
        return 1/(1+np.exp(-z))


    def compute_cost(self, X, y, w, b, lambda_=0):
        """
        Mide el desempeño del modelo de regresión logística calculando la logistic loss.

        Args:
            - X (ndarray (m,n)): dataset. 
            - y (ndarray (m,)): etiqueta o salida verdadera.
            - w (ndarray (n,)): pesos o weights.
            - b (float): sesgo o bias.
            - lambda_ (float): término de regularización. Es un número entre 0 e infinito.
            Nota: 'm' es el número de ejemplos y 'n' es el número de features.
            
        Returns:
            - cost (float): costo o error cuadrático medio.
        """        
        epsilon = 1e-7 # necesario para que no ocurra un error debido a ln(0)

        m = X.shape[0]
        z = np.dot(X, w) + b # (m,)
        prediction = self.__sigmoid(z) # (m,)
        losses = (-y * np.log(prediction + epsilon)) - ((1 - y) * np.log(1 - prediction + epsilon))  #(m,)
        cost = np.mean(losses) # scalar
        cost += (np.sum(w**2) * lambda_ / (2*m)) # scalar

        return cost


    def __compute_gradients(self, X, y, w, b, lambda_=0):
        """
        Calcula las derivadas parciales para todos los weights y bias del modelo de
        regresión logística.
        
        Args:
            - X (ndarray (m,n)): dataset. 
            - y (ndarray (m,)): etiqueta o salida verdadera.
            - w (ndarray (n,)): pesos o weights.
            - b (float): sesgo o bias.
            - lambda_ (float): término de regularización. Es un número entre 0 e infinito.
            Nota: 'm' es el número de ejemplos y 'n' es el número de features.
        
        Returns:
            - dj_dw (ndarray (n,)): numpy array de derivadas parciales de la función
                                    de costo con respecto a los weigths.
            - dj_db (float): derivada parcial de la función de costo con respecto al
                             bias.
        """
        m = X.shape[0]
        z = np.dot(X, w) + b # (m,)
        prediction = self.__sigmoid(z) # (m,)
        error = (prediction - y) # (m,)
        dj_dw = np.dot(error, X) / m # (n,)
        dj_dw += (w * lambda_ / m) # (n,)

        dj_db = np.mean(error) # scalar
        
        return dj_dw, dj_db


    def run_gradient_descent(self, X, y, w, b, alpha=0.01, num_iter=10, lambda_=0, verbose=False):
        """
        Ejecuta Batch Gradient Descent para entrenar el modelo de regresión logística.
            
        Args:
            - X (ndarray (m,n)): dataset. 
            - y (ndarray (m,)): etiqueta o salida verdadera.
            - w (ndarray (n,)): pesos o weights.
            - b (float): sesgo o bias.
            - alpha (float): learning rate.
            - num_iter (int): número de iteraciones o epochs del gradient descent.
            - lambda_ (float): término de regularización l2. Es un número entre 0 e infinito.
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
            dj_dw, dj_db = self.__compute_gradients(X, y, w, b, lambda_)
                
            # model params update
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            
            # compute cost
            cost = self.compute_cost(X, y, w, b, lambda_)
            
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


    def z_scaling_features(self, X):
        """
        Aplica la normalización z-score sobre la data proporcionada.

        Args:
            - X (ndarray (m,n)): dataset.

        Returns:
            - X_norm (ndarray (m,n)): dataset normalizada.
        """

        self.__mu = np.mean(X, axis=0) # (n,)
        self.__sigma = np.std(X, axis=0) # (n, )

        X_norm = (X - self.__mu) / self.__sigma
        return X_norm