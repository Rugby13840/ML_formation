import matplotlib.pyplot as plt 
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss

X,y= make_circles(n_samples=100,noise=0.1,factor=0.3,random_state=0)#On génère les données 
y=y.reshape((y.shape[0],1)) #on tranforme y en un array de la bonne taille

X=X.T#on transpose les données pour que ça soit cohérent avec le modèle
Y=y.reshape(1,y.shape[0])

#plt.scatter(X[0,:],X[1,:], c=y, cmap="summer")
#plt.show()


def initialisation(n0,n1,n2): #initialisation des différents W et b wij reprsente le poids de la connexion entre l'entrée j et l'arrivé i
    
    W1 = np.random.randn(n1, n0)
    b1 = np.zeros((n1, 1))
    W2 = np.random.randn(n2, n1)
    b2 = np.zeros((n2, 1))

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres

def forward_propagation(X,parametres):#on parcours le réseau de neuronnes
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    
    activations = {
        'A1' : A1,
        'A2' : A2
    }

    return activations



def back_propagation(X, y,activations,parametres):
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims = True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }
    
    return gradients

def update(gradients,parameters,learning_rate):  #on met à jour b et w selon le taux d'appprentissage
    
    W1=parameters['W1']
    b1=parameters['b1']
    W2=parameters['W2']
    b2=parameters['b2']

    dW1=gradients['dW1']
    db1=gradients['db1']
    dW2=gradients['dW2']
    db2=gradients['db2']
    
    
    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2

    parametres ={ #on crée un dictionnaire avec les résultats
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2
    }
    
    
    return parametres

def predict(X,parametres):
    activations=forward_propagation(X,parametres)
    A2=activations['A2']
    return A2 >= 0.5



def neural_network(X, y, n1=32, learning_rate = 0.1, n_iter = 10000):

    # initialisation parametres
    n0 = X.shape[0]
    n2 = y.shape[0]
    parametres = initialisation(n0, n1, n2)

    train_loss = []
    train_acc = []

    # gradient descent
    for i in range(n_iter):
        activations = forward_propagation(X, parametres)
        A2 = activations['A2']

        # Plot courbe d'apprentissage
        train_loss.append(log_loss(y.flatten(), A2.flatten()))
        y_pred = predict(X, parametres)
        train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))
        

        # mise a jour
        gradients = back_propagation(X, y, activations,parametres)
        parametres = update(gradients, parametres, learning_rate)
        

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()

    return parametres

X,y= make_circles(n_samples=100,noise=0.1,factor=0.3,random_state=0)#On génère les données 
y=y.reshape((1,y.shape[0])) #on tranforme y en un array de la bonne taille

X=X.T#on transpose les données pour que ça soit cohérent avec le modèle


parametres = neural_network(X,y,n1=16)