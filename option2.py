import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import scipy.stats

data = np.load('my_dataset.npy')
# plt.scatter(data[:,0],data[:,1])
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
            plt.scatter(belongings[:, 0], belongings[:,1], c=colors[i], alpha=0.5)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='P',c='red', s=200)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()

def coupling(data,targets):
    couples = {}
    densities = []
    for i in set(targets):
        couples[str(int(i))] = []
    for j in range(len(data)):
        couples[str(int(targets[j]))].append(list(data[j]))
    return couples

def scattering(data,couples,targets,centroids):
    entropies = []
    scats = []
    for ch in couples:
        
        k = np.array(centroids[int(float(ch))])
        x_y = np.array(couples[ch])
        scats.append(np.sum(np.sqrt((x_y - k)**2)))
        entropy_x = scipy.stats.entropy(np.array(couples[ch])[:,0])
        entropy_y = scipy.stats.entropy(np.array(couples[ch])[:,1])
        total_entropy = (entropy_x) + (entropy_y)
        entropies.append(total_entropy)
    return entropies
def density(data,couples,targets):
    densities = []
    for ch in couples:
        base =  np.array(couples[ch])[:,0].max()- np.array(couples[ch])[:,0].min()
        height = np.array(couples[ch])[:,1].max()- np.array(couples[ch])[:,1].min()
        Area = base * height
        if(len(couples[ch])==1):
            Area = np.array(couples[ch])[:,0].max() * np.array(couples[ch])[:,1].max()
        density = len(couples[ch]) / Area
        densities.append(density)
    return densities
def quality(data,couples,targets,w,centroids):
    dens = np.array(density(data,couples,targets))
    scatt = 1.0 / np.array(scattering(data,couples,targets,centroids))
    qual = w*(0.002*dens - 0.1*scatt)
    return np.sum(qual)
x = []
sumes = []
for k in range(3,11):
    x.append(k)
    quals = []
    for i in range(35):
        centroids,targets = k_means_p(data,k)
        couples = coupling(data,targets)
        w = []
        for i in couples:
            w.append(len(couples[i]))
        w = np.array(w)
        qual = quality(data,couples,targets,w,centroids)
        quals.append(qual)
    sumes.append(np.mean(quals))
plt.xlabel('Number of centroids')
plt.ylabel('Quality')
plt.plot(x,sumes)
plt.show()
