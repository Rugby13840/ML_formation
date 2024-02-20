import matplotlib.pyplot as plt 
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss

X,y= make_circles(n_samples=100,noise=0.1,factor=0.3,random_state=0)#On génère les données 
y=y.reshape((y.shape[0],1)) #on tranforme y en un array de la bonne taille

X=X.T#on transpose les données pour que ça soit cohérent avec le modèle
y=y.reshape((1,y.shape[0]))

#plt.scatter(X[0,:],X[1,:], c=y, cmap="summer")
#plt.show()


def initialisation(dimensions): #initialisation des différents W et b wij reprsente le poids de la connexion entre l'entrée j et l'arrivé i
    C=len(dimensions)
    
    parametres = {}
    
    for c in range (1,C):

        parametres['W'+str(c)]=np.random.randn(dimensions[c], dimensions[c-1])
        parametres['b'+str(c)]=np.zeros((dimensions[c], 1))

    return parametres

def forward_propagation(X,parametres):#on parcours le réseau de neuronnes
    
    activations = {'A0' : X}
    
    C = len(parametres) // 2

    for c in range(1,C+1):

        Z = parametres['W'+str(c)].dot(activations['A'+str(c-1)]) + parametres['b'+str(c)]
        activations['A'+str(c)]= 1 / (1 + np.exp(-Z))
        
    return activations



def back_propagation(y,activations,parametres):

    C = len(parametres) // 2

    gradients = {}
    m = y.shape[1]

    dZ=activations['A'+str(C)]-y
    
    for c in reversed(range(1,C+1)):
        gradients['dW'+str(c)]=1 / m * np.dot(dZ,activations['A'+str(c - 1)].T)
        gradients['db'+str(c)]=1 / m * np.sum(dZ, axis=1, keepdims = True)
        if c > 1 :
            dZ = np.dot(parametres['W'+str(c)].T, dZ) * activations['A'+str(c-1)] * (1 - activations['A'+str(c-1)])
   
    return gradients

def update(gradients,parametres,learning_rate):  #on met à jour b et w selon le taux d'appprentissage
    
    C = len(parametres) // 2

    for c in  range(1,C+1):
        parametres['W'+str(c)]=parametres['W'+str(c)]-learning_rate*gradients['dW'+str(c)]
        parametres['b'+str(c)]=parametres['b'+str(c)]-learning_rate*gradients['db'+str(c)]                                                                      

    return parametres

def predict(X,parametres):
    activations=forward_propagation(X,parametres)
    C = len(parametres) // 2
    return activations['A'+str(C)] >= 0.5



def neural_network(X, y, hidden_layers=(32,32,32), learning_rate = 0.1, n_iter = 10000):#n1 le nombre de neuronnes de la première couche, n0 le nombre de paramètre en entrée, n2 le nombre de neuronnes dans la deuxième couche

    # initialisation parametres
    dimensions=list(hidden_layers)
    dimensions.insert(0,X.shape[0])
    dimensions.append(y.shape[0])
    parametres = initialisation(dimensions)

    train_loss = []
    train_acc = []

    # gradient descent
    for i in range(n_iter):
        activations = forward_propagation(X, parametres)
        gradients = back_propagation(y, activations,parametres)
        parametres = update(gradients, parametres, learning_rate)
        
        if i%10 == 0 :
            C=len(parametres)//2
            train_loss.append(log_loss(y, activations['A'+str(C)]))
            y_pred = predict(X, parametres)
            train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))
        

        

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()
    print(train_acc.pop())
    return parametres

parametres=initialisation([2,32,32,1])

activations=forward_propagation(X,parametres)

grad = back_propagation(y,activations,parametres)

neural_network(X, y, hidden_layers=(32,32,32), learning_rate = 0.1, n_iter = 10000)