import logging
import bohrium as bh
import numpy as np
from sklearn.cluster import KMeans
import scipy
import time
import utils
import random
import os
from benchpress.benchmarks import util

def timeit(func):
    def let_time(*args, **kwargs):
        bh.flush()
        ts = time.time()
        result = func(*args, **kwargs)
        bh.flush()
        te = time.time()
        print('Function "{name}" took {time} seconds to complete.'.format(name=func.__name__, time=te-ts))
        return result
    return let_time

class bohrium_kmeans:

    def __init__(self, k, init = "kmeans++", userkernel = True, max_iter = 300, gpu=False, verbose = False):

        if k <= 0:
            raise ValueError("Invalid number of initializations."
                             " n_init=%d must be bigger than zero." % k)

        self.max_iter = max_iter
        self.verbose = verbose
        self.userkernel = userkernel
        self.k = k
        self.init = init
        self.init_centroids = bh.array([0])
        self.gpu = gpu

        dirname, filename = os.path.split(os.path.abspath(__file__))
        userkerneldir = dirname + "/user-kernels/"

        if self.userkernel:
            self.kernel_centroids_closest = open(userkerneldir + 'centroids_closest.c').read()
            self.kernel_centroids_closest_opencl = open(userkerneldir + 'centroids_closest_opencl.c').read()
            self.kernel_move_centroids = open(userkerneldir + 'move_centroids.c', 'r').read()
            self.kernel_shuffle = open(userkerneldir + 'shuffle.c', 'r').read()

        if self.verbose:
            print("Userkernels has been loaded")

    def __str__(self):
        return "Userkernel: {}, Number of clusters: {}".format(self.userkernel, str(self.k))


    def init_random_userkernel(self, points):
        # os.environ["BH_STACK"] = "openmp"
        temp = points.copy()
        bh.user_kernel.execute(self.kernel_shuffle, [temp], tag="openmp")

        if self.verbose:
            print("Initialized {} number of centroids randomly with userkernel".format(str(self.k)))

        self.init_centroids = temp[:self.k]

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
        self.centroids = points[:self.k]
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

            min_distances = self.euclidian_distance(centroids[:k], points).min(1)
            prob = min_distances / min_distances.sum()
            cs = bh.cumsum(prob)
            idx = bh.sum(cs < bh.random.rand())
            centroids[k] = points[int(idx)]

        self.init_centroids = centroids

        if self.verbose:
            print("Initialized {} centroids according to kmeans++".format(str(self.k)))

        return centroids


    def init_parallel(self, points, l = 2):
        # centroids = bh.zeros_like(points[:self.k])

        r = bh.random.randint(0, points.shape[0])
        centroids = points[None,r]
        psi = self.euclidian_distance(centroids, points).min(1).sum()


        for i in range(int(bh.log(psi))):
            distance = self.euclidian_distance(centroids, points)
            norm_const = bh.sum(bh.min(distance,1))
            prob = l * bh.min(distance, axis = 1) / norm_const
            cs = bh.cumsum(prob)
            sut = bh.random.rand(l)
            idxk = bh.sum(cs[:,None] < sut, 0)

            temp_point = bh.empty_like(points[:len(idxk)])
            for idx, val in enumerate(idxk):
                temp_point[idx] = points[val]


            centroids = bh.vstack((centroids, temp_point))

        a = self.init_random_userkernel(centroids)

        self.init_centroids = a
        return a

    @timeit
    def euclidian_distance(self, points1, points2, square = True):
        """
        Calculates the euclidian distance between two sets of points.
        This uses numpy broadcasting trick if the to sets arent the same size

        Parameters:
        -----------
        points1: nd array
            First set of points
        points2: nd array
            Second set of points
        square: bool, optional
            If to take the square root or not when performing the calculation

        Returns:
        --------
        Distances matrix:
        """

        X = points1 - points2[:, None]
        distances = (X * X).sum(axis=2)
        return distances if square else bh.sqrt(distances)

    @timeit
    def centroids_closest(self, points, centroids):

        distances = self.euclidian_distance(points, centroids)
        distances = bh.array(distances,dtype=bh.float64)

        if self.userkernel:

            result = bh.zeros(points.shape[0], dtype = bh.int64)
            min_dist = bh.zeros(points.shape[0], dtype = bh.float64)
            distances_transposed = bh.user_kernel.make_behaving(distances.T)


            if self.gpu:
                # os.environ["BH_STACK"] = "opencl"
                
                bh.user_kernel.execute(self.kernel_centroids_closest_opencl,
                                       [distances_transposed, min_dist, result],
                                       tag="opencl", param={"global_work_size": [points.shape[0]], "local_work_size": [1]})

            else:
                cmd = bh.user_kernel.get_default_compiler_command()

                bh.user_kernel.execute(self.kernel_centroids_closest,
                                       [distances_transposed, result, min_dist], compiler_command = cmd)


        else:
            distances = distances.copy2numpy()
            result = bh.array(np.argmin(distances, axis = 0))
            min_dist = bh.minimum.reduce(distances, 0)

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

        if not n:
            n = self.k

        #mask = (closest == bh.arange(n)[:,None])
        #out= mask.dot(points)/ mask.sum(1)[:,None]

        #out = bh.array()a
        out = bh.zeros_like(centroids)
        for i in range(self.k):
            out[i] = points[closest==i].mean(0)
            if closest[i] == i:
                temp += points[i]
            

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


    @timeit
    def run(self, points, epsilon=0.00001, square = True):

        if self.k > points.shape[0]:
            raise ValueError("number of points=%d should be >= k=%d" % (
                points.shape[0], self.k))


        if self.userkernel:
            self.kernel_centroids_closest_opencl = self.kernel_centroids_closest_opencl.replace("int n_points = 0", "int n_points = " + str(points.shape[0]))
            self.kernel_centroids_closest_opencl = self.kernel_centroids_closest_opencl.replace("int n_k = 0", "int n_k = " + str(self.k))
            self.kernel_centroids_closest = self.kernel_centroids_closest.replace("int n_points = 0", "int n_points = " + str(points.shape[0]))
            self.kernel_centroids_closest = self.kernel_centroids_closest.replace("int n_k = 0", "int n_k = " + str(self.k))
            self.kernel_shuffle = self.kernel_shuffle.replace("int rows", "int rows = " + str(points.shape[0]))
            self.kernel_shuffle = self.kernel_shuffle.replace("int cols", "int cols = " + str(points.shape[1]))


            if self.init != "kmeans++":
                print("init: random")
                centroids = self.init_random_userkernel(points)



        if self.init == "kmeans++":
            print("init: ++")
            centroids = self.init_plus_plus(points)

        else:
            centroids = self.init_random_centroids(points)

        iterations = 0


        if self.verbose:
            print("Done initializing, starting..")

        for iterations in range(self.max_iter):

            old_centers = centroids.copy()

            closest, min_dist = self.centroids_closest(points, centroids)
            centroids = self.move_centroids(points, closest, centroids)
            inertia = min_dist.sum()

            x = old_centers - centroids
            x = bh.ravel(x)

            #This is the SKlearn way of exiting.
            if (bh.dot(x, x) <= epsilon):
                if self.verbose:
                    print("Converged after {} iterations".format(str(iterations)))

                return closest, centroids, iterations, inertia

        if self.verbose:
            print("Exit after max ({}) iterations".format(str(iterations)))

        return closest, centroids, iterations, inertia

def benchmark():

    times = bench.args.size[0]
    exp = bench.args.size[1]
    gp = bench.args.size[2]



    k = 25

    bh.random.seed(0)

    points = bh.random.randint(2*times*10**exp, size=(times*10**exp, 2), dtype=bh.float64)

    kmeans = bohrium_kmeans(k, userkernel=True, init="random", gpu=True, verbose=True)

    bh.flush()
    bench.start()

    kmeans.run(points)

    bh.flush()

    bench.stop()
    bench.pprint()





if __name__ == "__main__":
    # from sklearn.cluster import KMeans
    # print("here")
    bench = util.Benchmark("kmeans", "k")
    benchmark()
    #points = bh.loadtxt("../data/birchgrid.txt")
    #bh.random.seed(0)
    #points = bh.random.randint(2*10, size=(10, 2), dtype=bh.float64)
    #print(points)
    #kmeans = bohrium_kmeans(100, userkernel=True, init="kmeans++", gpu=False)

    # points = bh.array(points)
    #clos, cent, ite, iner = kmeans.run(points)
    # # print(iner)

    #kmeans.kernel_centroids_closest_opencl = kmeans.kernel_centroids_closest_opencl.replace("int n_points = 0", "int n_points = " + str(points.shape[0]))
    #kmeans.kernel_centroids_closest_opencl = kmeans.kernel_centroids_closest_opencl.replace("int n_k = 0", "int n_k = " + str(kmeans.k))

    #centroids = kmeans.init_random_userkernel(points)

    #closest, min_dist = kmeans.centroids_closest(points, centroids)
    #closest, min_dist = kmeans.centroids_closest(points, centroids)


    # start = time.time()
    # skmeans = KMeans(n_clusters = 100, n_init = 1, verbose =  1, ).fit(points)
    # end = time.time()
