import math
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X,y= make_blobs(n_samples=100,n_features=2,centers=2,random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="summer")
plt.show()