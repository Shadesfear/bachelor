import logging
import bohrium as bh
import numpy as np
from sklearn.cluster import KMeans
import time
import random
from benchpress.benchmarks import util

def timeit(func):
    def let_time(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('Function "{name}" took {time} seconds to complete.'.format(name=func.__name__, time=te-ts))
        return result
    return let_time



class bohrium_kmeans:

    def __init__(self, k, userkernel = True):

        self.userkernel = userkernel
        self.k = k

        if self.userkernel:
            with open('centroids_closest.c', 'r') as content_file:
                self.kernel_centroids_closest = content_file.read()

            with open('move_centroids.c', 'r') as content_file:
                self.kernel_move_centroids = content_file.read()

            with open('shuffle.c', 'r') as content_file:
                self.kernel_shuffle = content_file.read()


    def __str__(self):
        return "Userkernel: {}, Number of clusters: {}".format(self.userkernel, str(self.k))


    def init_random_userkernel(self, points):
        temp = points.copy()

        self.kernel_shuffle = self.kernel_shuffle.replace("int rows", "int rows = " + str(points.shape[0]))
        self.kernel_shuffle = self.kernel_shuffle.replace("int cols", "int cols = " + str(points.shape[1]))

        bh.user_kernel.execute(self.kernel_shuffle, [temp])

        return temp[:self.k]


    def init_random_centroids(self, points):

        centroids = bh.zeros_like(points[:self.k])
        for i in range(self.k):
            randint = bh.random.randint(points.shape[0])
            while (centroids == points[randint]).all(1).any():
                randint = bh.random.randint(self.k)
            centroids[i] = points[randint]

        self.centroids = centroids
        return centroids


    def init_first_k(self, points):
        return points[:self.k]


    def init_plus_plus(self, points):
        """
        Initialize the clusters using KMeans++

        """

        centroids = bh.zeros_like(points[:self.k])
        centroids[0] = points[0]

        distances = self.euclidian_distance(centroids[:1], points, mode = 'squared')

        for i in range(1, self.k):
            pass
        return(centroids)



    def euclidian_distance(self, point1, point2, mode = 'squared'):

        if mode == 'squared':
            distances = ((point1 - point2[:, bh.newaxis])**2).sum(axis=2)

        elif mode == 'normal':
            distances = bh.sqrt(((point1 - point2[:, bh.newaxis])**2).sum(axis=2))
        return(distances)

    @timeit
    def centroids_closest(self, points, centroids, mode):

        distances = self.euclidian_distance(points, centroids, mode)
        min_dist = bh.minimum.reduce(distances, 0)

        if not self.userkernel:
            # diff = points[None, :, :] - centroids[:, None, :]
            # distances = bh.sqrt(((points - centroids[:, bh.newaxis])**2).sum(axis=2))

            distances = distances.copy2numpy()
            ary = bh.array(np.argmin(distances, axis = 0))

            return ary, min_dist

        else:

            result = bh.zeros(points.shape[0], dtype = bh.int)

            distances_transposed = bh.user_kernel.make_behaving(distances.T)

            self.kernel_centroids_closest = self.kernel_centroids_closest.replace("int n_points = 0", "int n_points = " + str(points.shape[0]))
            self.kernel_centroids_closest = self.kernel_centroids_closest.replace("int n_k = 0", "int n_k = " + str(self.k))

            # cmd = bh.user_kernel.get_default_compiler_command() + " -lfftw3 -lfftw3_threads"

            bh.user_kernel.execute(self.kernel_centroids_closest, [distances_transposed, result])

            return result, min_dist


    def run_plot(self, points):
        """
        Run the kmeans, afterwards plot with matplotlib.
        Also converts to numpy arrays for pyplot

        Parameters:
        -----------

        points: points to run the kmeans on

        Returns:
        --------

        Closest: array of labels for each points to centroids
        Centroids: List of centroids
        Iterations: Number of iterations it took to converge

        """
        import matplotlib.pyplot as plt
        closest, centroids, iterations = self.run(points)

        points = points.copy2numpy()
        centroids = centroids.copy2numpy()
        closest = closest.copy2numpy()

        plt.scatter(points[:,0], points[:,1], marker = ".", s=50, c = closest)
        plt.scatter(centroids[:,0], centroids[:,1], marker = "X", s=400, c = 'r')
        plt.show()


    @timeit
    def move_centroids(self, points, closest, centroids, n = 0 ):

        if self.userkernel:
            logging.info("move_centroids: userkernel")

            centroids = bh.user_kernel.make_behaving(centroids)

            bh.user_kernel.execute(self.kernel_move_centroids, [bh.array([self.k]),
                                                                bh.array(closest),
                                                                points,
                                                                bh.array([len(closest)]),
                                                                bh.array([len(points[0])]),
                                                                centroids])
            return centroids

        else:

            logging.info("move_centroids: numpy")
            if not n:
                n = self.k

            mask = (closest == bh.arange(n)[:,None])
            out = mask.dot(points)/ mask.sum(1)[:,None]

            #new_centroid = bh.array([points[closest==k].mean(axis = 0) for k in range(self.k)])

            return out



    def scale_data(self, points):
        """
        Scales data on a pr feature level.

        Parameters
        ----------
        Data points : array
            The data to be scaled

        Returns
        -------
        Scaled Data : array

        """
        std = points.std(axis = 0)
        zero_std = std == 0

        if zero_std.any():
            std[zero_std] = 1.0

        scaled_data = points / std
        return scaled_data


    def run(self, points, epsilon=0.1, mode = 'squared'):

        if self.userkernel:
            centroids = self.init_random_userkernel(points)
        else:
            centroids = self.init_random_centroids(points)

        centroids_old = bh.zeros(centroids.shape)
        iterations, diff = 0, epsilon+1

        avg_dist = []

        while diff > epsilon:

            centroids_old = centroids
            closest, min_dist = self.centroids_closest(points, centroids, mode)

            centroids = self.move_centroids(points, closest, centroids)

            avg_dist.append(bh.mean(min_dist, axis = -1))

            if len(avg_dist) > 1:
                diff = avg_dist[-2] - avg_dist[-1]

                if ((avg_dist[-2] - avg_dist[-1])/avg_dist[-2])*100 < epsilon:

                    print("broke here")
                    return closest, centroids, iterations

            iterations += 1
        return closest, centroids, iterations




if __name__ == "__main__":


    points = bh.loadtxt("../data/dim032.txt")

    kmeans = bohrium_kmeans(100, userkernel=True)

    # kmeans.run(points)

    # points = bh.array([[1,1],
    #                    [2,2],
    #                    [3,3],
    #                    [4,4],
    #                    [5,5],
    #                    [6,6]], dtype=np.float64)

    for i in range(10):
        start = time.time()
        centroids = kmeans.init_random_userkernel(points)
        print("Time: ", time.time() - start)
    print(centroids)


    # print("USER KERNEL FALSE\n")

    # kmeans.run(points)
    # print("\n")
    # print(kmeans)
    # print("\n")

    # kmeans = bohrium_kmeans(100, userkernel=True)

    # print("USERKERNEL TRUE")
    # kmeans.run(points)
    # print(kmeans
