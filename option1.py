import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import scipy.stats

data = np.load('my_dataset.npy')
for i in data[:,1]:
    print(i,end=',')
plt.scatter(data[:,0],data[:,1])
def manhattan_distance(vect1,vect2):
    subt = np.abs(vect1-vect2)
    return np.sum(subt)
def k_means_p(data,K):
    index = np.random.randint(0,len(data))
    dat_in = data[index]
    centroids = [[dat_in[0],dat_in[1]]]

    num_k = K

    sums = []
    dat_tab = np.zeros((num_k-1,len(data)))
    vals = deepcopy(data)
    for i in range(num_k-1):
        for j in range(len(data)):
            dat_tab[i][j] = manhattan_distance(centroids[i],data[j])**2
        current = dat_tab[:i+1]
        mins = []
        for l in range(len(data)):
            mins.append(np.min(current[:,l]))
        centroids.append(list(data[np.argmax(mins/np.sum(mins))]))
        mins = []
    centroids = np.array(centroids)
    # Following matrix will help to store the previous (update) version of the centroids
    centroids_old = np.zeros(centroids.shape)
    # Following vector will contain the appropriate cluster of each set
    targets = np.zeros(len(data))
    distances = np.zeros(num_k)

    # Importance of the difference is to see the error between the updated version
    differences = np.zeros(num_k)
    differences = np.array([manhattan_distance(centroids[i], centroids_old[i]) for i in range(num_k)])

    # The loop will not sop is none of the error elements is zero ( convergence)
    while differences.all() != 0:
        # Fins the nearest cluster for each dataset
        for i in range(len(data)):
            distances = np.array([manhattan_distance(data[i], centroids[j]) for j in range(num_k)])
            answer = np.argmin(distances)
            targets[i] = answer
        # Storing the centroids
        centroids_old = deepcopy(centroids)
        for i in range(num_k):
            belongings=[]
            #collect all the sets with the target (cluster) i
            for j in range(len(data)):
                if targets[j]==i:
                    belongings.append(data[j])
            # Update the centorids
            centroids[i] = np.mean(belongings, axis=0)
        differences = np.array([manhattan_distance(centroids[i], centroids_old[i]) for i in range(num_k)])
    return centroids,targets
def partial_plot(data,K,centroids,targets):
    plt.figure(figsize=(7,7))
    colors = ['green', 'navy', 'black','yellow','orange']
    for i in range(K):
        belongings=[]
        for j in range(len(data)):
            if targets[j]==i:
                belongings.append(data[j])
        for l in range(len(belongings)):
            belongings[l]=[belongings[l][0],belongings[l][1]]
            belongings=np.array(belongings)
    #         plt.scatter(belongings[:, 0], belongings[:,1], c=colors[i], alpha=0.5)

    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='P',c='red', s=200)
    # plt.xlabel('x axis')
    # plt.ylabel('y axis')
    # plt.show()
def evaluate_model(data):
    x = []
    y = []
    A = []
    for k in range(3,10):
        epo_centers = {}
        for i in range(32):
            epo_centers[str(i)] = k_means_p(data,k)[0]
        l_let = []
        for i in epo_centers:
            for j in epo_centers[i]:
                l_let.append(j)
        l_let = np.array(l_let)
        p_data1 = np.unique(l_let[:,0],return_counts=True)[1]           #
        entropy1 = scipy.stats.entropy(p_data1)

        p_data2 = np.unique(l_let[:,1],return_counts=True)[1]           #
        entropy2 = scipy.stats.entropy(p_data2)
        entropy = entropy1 + entropy2
        x.append(k)
        y.append(entropy)
    plt.plot(x,y)
    plt.xlabel('k')
    plt.ylabel('entropy')
    plt.show()

# partial_plot(data,3,centroids,targets)
centroids,targets = k_means_p(data,4)
partial_plot(data,4,centroids,targets)
evaluate_model(data)
# plt.scatter(data[:,0],data[:,1])
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='D',c='red', s=200)
# plt.show()