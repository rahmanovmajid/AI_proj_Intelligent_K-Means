import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import scipy.stats

data = np.zeros((100,2))
centroids = np.array([[4,8,0.5,1],[6,10,6/8,10/8],[6,7,6/8,7/8],[8,8,1,1]])

def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*(-0.5*(x-mu)**2/(sigma**2))
np.random.seed(0)
for i in range(100):
    
    k = random.randint(0,3)
    data[i][0] = centroids[k][0] + random.gauss(centroids[k][0],centroids[k][2])
    data[i][1] = centroids[k][1] + random.gauss(centroids[k][1],centroids[k][3])
np.save('my_dataset', data)
plt.figure(figsize=(7,7))
plt.scatter(data[:,0],data[:,1])
plt.show()