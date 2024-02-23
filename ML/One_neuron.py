import matplotlib.pyplot as plt 
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

X,y= make_blobs(n_samples=100,n_features=3,centers=2,random_state=0)#On génère les données 
y=y.reshape((len(y),1)) #on tranforme y en un array de la bonne taille
#X,y= make_circles(n_samples=100,noise=0.1,factor=0.3,random_state=0)#On génère les données 
#y=y.reshape((y.shape[0],1)) #on tranforme y en un array de la bonne taille




def init(X): #initialisation de W et b 
    W=np.random.randn(X.shape[1],1)
    b=np.random.randn(1)
    return W,b

def model(X,W,b):#modèle du perceptron
    Z= X.dot(W) + b 
    A = 1 / (1 + np.exp(-Z)) #fonction d'activation
    return A

def log_loss(A,y): #fonction coût
    M= y * np.log(A) + (1-y) * np.log(1-A)
    return -(1/len(y))*np.sum(M)

def gradient(X,A,y):
    dw=(1/len(y))*np.dot(X.T,A-y)
    db=(1/len(y))*np.sum(A-y)
    return dw,db 

def update(dw,db,W,b,learning_rate):  #on met à jour b et w selon le taux d'appprentissage
    W=W-learning_rate*dw
    b=b-learning_rate*db
    return W,b

def predict(X,W,b):
    A=model(X,W,b)
    return A >= 0.5

def artificial_neuron(X,y,learning_rate=0.1,n_iter=100000): #on crée notre neuronne artificiel et on entraine notre modèle sur les données 
    (W,b)=init(X) #on itialise W et b 
    Loss=[]
    for i in range (n_iter): #Algo de descente de gradient 
        A=model(X,W,b)
        Loss.append(log_loss(A,y))
        dw,db=gradient(X,A,y)
        W,b=update(dw,db,W,b,learning_rate)
    
    y_pred = predict(X,W,b)
    print(accuracy_score(y,y_pred))
    
    plt.plot(Loss) #on affiche l'évolution de Loss
    plt.show()
    return(W,b)

W,b = artificial_neuron(X,y) #On détermine W,b de notre modèle et à partir de nos données 


new_plant=np.array([[2,2]])


x0=np.linspace(-1,4,100)
Y0=(-W[0]*x0 - b )/W[1]
plt.plot(x0,Y0,c='b')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="summer")
plt.scatter(new_plant[0][0],new_plant[0][1], c='r')

print(predict(new_plant,W,b))

plt.show()