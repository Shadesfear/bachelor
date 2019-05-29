from sklearn.cluster import KMeans
import time
import numpy as np



points = np.random.randint(5*2*10**6, size=(5 * 10**6, 2))

start = time.time()
kmeans = KMeans(n_clusters=25, n_init=1, random_state=0, algorithm='full', init='random').fit(points)
print(time.time() - start)
