import logging
import bohrium as bh
import numpy as np
from sklearn.cluster import KMeans
import scipy
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

    def __init__(self, k, init = "kmeans++", userkernel = True):

        self.userkernel = userkernel
        self.k = k
        self.init = init
        userkerneldir = "user-kernels/"

        if self.userkernel:
            with open(userkerneldir + 'centroids_closest.c', 'r') as content_file:
                self.kernel_centroids_closest = content_file.read()

            with open(userkerneldir + 'move_centroids.c', 'r') as content_file:
                self.kernel_move_centroids = content_file.read()

            with open(userkerneldir + 'shuffle.c', 'r') as content_file:
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
        Initialize the clusters using KMeans++, where each centroid
        is chosen with randomly with weights

        Parameters:
        -----------
        points: Points to sample centroids from

        Returns:
        --------
        Centroids: K-centroids

        """

        centroids = bh.zeros_like(points[:self.k])
        r = bh.random.randint(0, points.shape[0])
        centroids[0] = points[r]

        for k in range(1, self.k):

            min_distances = self.euclidian_distance(centroids[:k], points, mode = 'squared').min(1)
            prob = min_distances / min_distances.sum()
            cs = bh.cumsum(prob)
            idx = bh.sum(cs < bh.random.rand())
            centroids[k] = points[int(idx)]

        self.init_centroids = centroids
        return centroids


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

            start = time.time()
            result = bh.zeros(points.shape[0], dtype = bh.int)

            distances_transposed = bh.user_kernel.make_behaving(distances.T)

            self.kernel_centroids_closest = self.kernel_centroids_closest.replace("int n_points = 0", "int n_points = " + str(points.shape[0]))
            self.kernel_centroids_closest = self.kernel_centroids_closest.replace("int n_k = 0", "int n_k = " + str(self.k))



            cmd = bh.user_kernel.get_default_compiler_command()
            start = time.time()
            bh.user_kernel.execute(self.kernel_centroids_closest, [distances_transposed, result], compiler_command = cmd)
            print("CC: ",time.time()-start)
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
        if self.init_centroids.size > 0:
            plt.scatter(self.init_centroids[:,0], self.init_centroids[:,1], marker = "X", s=400, c = 'b')
        plt.show()


    @timeit
    def move_centroids(self, points, closest, centroids, n = 0 ):

        if self.userkernel:
            # logging.info("move_centroids: userkernel")

            # centroids = bh.user_kernel.make_behaving(centroids)

            # bh.user_kernel.execute(self.kernel_move_centroids, [bh.array([self.k]),
            #                                                     bh.array(closest),
            #                                                     points,
            #                                                     bh.array([len(closest)]),
            #                                                     bh.array([len(points[0])]),
            #                                                     centroids])
            # return centroids

            if not n:
                n = self.k

            mask = (closest == bh.arange(n)[:,None])
            out = mask.dot(points)/ mask.sum(1)[:,None]
            return out

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

    def get_error(self, closest, points, centroids):
        pass

    @timeit
    def run(self, points, epsilon=0.01, mode = 'squared'):

        if self.userkernel:
            centroids = self.init_random_userkernel(points)

        elif self.init == "kmeans++":
            centroids = self.init_plus_plus(points)

        else:
            centroids = self.init_random_centroids(points)

        centroids_old = bh.zeros(centroids.shape)
        iterations = 0
        diff = epsilon + 1
        avg_dist = []


        while diff > epsilon:

            if iterations > 0:
                old_min_dist = min_dist.copy()
                old_closest = closest.copy()

            # centroids_old = centroids
            closest, min_dist = self.centroids_closest(points, centroids, mode)
            centroids = self.move_centroids(points, closest, centroids)

            if iterations > 0:

                if (old_closest==closest).all():
                    print("broke closes")
                    return closest, centroids, iterations

                if (bh.sum(old_min_dist) - bh.sum(min_dist)) < epsilon:
                    print("broke new")
                    return closest, centroids, iterations

            # avg_dist.append(bh.mean(min_dist, axis = -1))

            # if len(avg_dist) > 1:
            #     diff = avg_dist[-2] - avg_dist[-1]

            #     if ((avg_dist[-2] - avg_dist[-1])/avg_dist[-2])*100 < epsilon:

            #         print("Avg dist broke")
            #         return closest, centroids, iterations

            iterations += 1
        return closest, centroids, iterations




if __name__ == "__main__":

    import matplotlib.pyplot as plt
    points = bh.loadtxt("../data/birchgrid.txt")


    kmeans = bohrium_kmeans(100, userkernel=True)
    kmeans_random = bohrium_kmeans(100, init = "random", userkernel=True)

    # centroids = kmeans.init_plus_plus(points)

    start = time.time()
    closest, centroids, iterations = kmeans.run(points)


    print("++: ", time.time()-start)


    # plt.scatter(points[:,0], points[:,1], marker = ".", s=50, c = closest)
    # plt.scatter(centroids[:,0], centroids[:,1], marker = "X", s=400, c = 'r')
    # plt.show()





    # print("USER KERNEL FALSE\n")

    # kmeans.run(points)
    # print("\n")
    # print(kmeans)
    # print("\n")

    # kmeans = bohrium_kmeans(100, userkernel=True)

    # print("USERKERNEL TRUE")
    # kmeans.run(points)
    # print(kmeans
