from sklearn.cluster import KMeans
from bohrium_kmeans import *

# import sklearn.cluster.k_means_
import sklearn

import numpy as np
import bohrium as bh
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
import time
import inspect

np.random.seed(0)
bh.random.seed(0)

print(np.random.randint(100))
print(bh.random.randint(100))

data = np.loadtxt("../data/birchgrid.txt")
k=10
random_state = check_random_state(None)
x_squared_norms = row_norms(data, squared=True)



# start = time.time()
# center = sklearn.cluster.k_means_._k_init(data, k, x_squared_norms, random_state)
# print(time.time()-start)
# print(inspect.getsourcefile(sklearn.cluster.k_means_._k_init))
# print(time.time() - start)
centroids = data[:k]

kmeans = bohrium_kmeans(k, gpu=True)
# dist = kmeans.euclidian_distance(centroids, data)


start = time.time()
center2 = kmeans.init_plus_plus(data)

h1, h2 = kmeans.centroids_closest(data, center2)

print("mine", time.time() - start)

# print(center)
# print(center2)
