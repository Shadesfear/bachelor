import logging
import bohrium as bh
import numpy as np
from sklearn.cluster import KMeans
import scipy
import time
import utils
import random
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

    def __init__(self, k, init = "kmeans++", userkernel = True, max_iter = 300, gpu=False):

        if k <= 0:
            raise ValueError("Invalid number of initializations."
                             " n_init=%d must be bigger than zero." % k)

        self.max_iter = max_iter
        self.userkernel = userkernel
        self.k = k
        self.init = init
        self.init_centroids = np.array([0])
        self.gpu = gpu

        # userkerneldir = "/home/chris/Documents/bachelor/src/user-kernels/"
        userkerneldir = "/home/cca/bachelor/src/user-kernels/"


        if self.userkernel:

            self.kernel_centroids_closest = open(userkerneldir + 'centroids_closest.c').read()
            self.kernel_centroids_closest_opencl = open(userkerneldir + 'centroids_closest_opencl.c').read()
            self.kernel_move_centroids = open(userkerneldir + 'move_centroids.c', 'r').read()
            self.kernel_shuffle = open(userkerneldir + 'shuffle.c', 'r').read()





    def __str__(self):
        return "Userkernel: {}, Number of clusters: {}".format(self.userkernel, str(self.k))


    def init_random_userkernel(self, points):
        temp = points.copy()

        if type(temp) != 'numpy.float32':
            temp = bh.float64(temp)


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
        """
        Initialize the first k points as centroids
        Parameters
        ----------
        points: Points to sample centroids from

        Returns
        -------
        Centroids: K-centroids
        """

        self.centroids = centroids
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
        return centroids


    def init_parallel(self, points, l = 2):
        # centroids = bh.zeros_like(points[:self.k])

        r = bh.random.randint(0, points.shape[0])
        centroids = points[None,r]
        psi = self.euclidian_distance(centroids, points).min(1).sum()


        for i in range(self.k / l):
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


    def euclidian_distance(self, point1, point2, square = True):
        """
        Calculates the euclidian distance between two sets of points.
        This uses numpy broadcasting trick if the to sets arent the same size

        Parameters:
        -----------
        point1: nd array
            First set of points
        point2: nd array
            Second set of points
        square: bool, optional
            If to take the square root or not when performing the calculation

        Returns:
        --------
        Distances matrix:


        """



        X = point1 - point2[:, None]

        if square:
            distances = (X * X).sum(axis=2)
            # return bh.dot(X,X)

        else:
            distances = bh.sqrt((X * X).sum(axis=2))

        return(distances)


    def centroids_closest(self, points, centroids):
        bh.flush()
        distances = self.euclidian_distance(points, centroids)

        if self.userkernel:

            result = bh.zeros(points.shape[0], dtype = bh.int64)
            min_dist = bh.zeros(points.shape[0], dtype = bh.float64)
            result2 = bh.zeros(points.shape[0], dtype = bh.int64)
            min_dist2 = bh.zeros(points.shape[0], dtype = bh.float64)

            distances_transposed = bh.user_kernel.make_behaving(distances.T)

            if self.gpu:

                bh.user_kernel.execute(self.kernel_centroids_closest_opencl,
                                       [distances_transposed, result, min_dist],
                                       tag="opencl", param={"global_work_size": [points.shape[0], self.k], "local_work_size": [1, 1]})

            else:
                cmd = bh.user_kernel.get_default_compiler_command()
                bh.user_kernel.execute(self.kernel_centroids_closest,
                                       [distances_transposed, result, min_dist], compiler_command = cmd)

                bh.user_kernel.execute(self.kernel_centroids_closest_opencl,
                                       [distances_transposed, result2, min_dist2],
                                       tag="opencl", param={"global_work_size": [points.shape[0], self.k], "local_work_size": [1, 1]})

                print((result == result2).all())

        else:
            bh.flush()
            distances2 = distances.copy2numpy()
            # bh.set_printoptions(precision=2, suppress=True)
            # print(distances)
            result2 = bh.array(np.argmin(distances2, axis = 0))
            # print("Result: ", result)
            bh.flush()
            # start =  time.time()
            start = time.time()# min_dist = bh.minimum.reduce(distances, 0)
            # min_dist = bh.min(distances, 0)
            min_dist = bh.amin(distances, 0)
            bh.flush()
            print("bh: ", time.time() - start)

            start = time.time()
            re = bh.argmin(distances, 0 )
            # min_dist2 = np.amin(distances2, 0)
            print("np: ", time.time() - start)



            # mask = (distances[:,None,0] <= data) & (data <= distances[:,None,1])
            # r,c = bh.nonzero(mask)
            # cut_idx = bh.unique(r, return_index=1)[1]
            # out = bh.minimum.reduceat(data[mask], cut_idx)



            # print(min_dist3)
            # bh.flush()
            # print("time: ", time.time() - start)
            # print(distances.shape)
            # mask = (distances.T == min_dist[:,None])
            # print(mask.T)
            # mask = mask.T
            # print(reorganization.nonzero(mask))
            # out = bh.masked_get(distances, mask)
            # print(out)
            # print(mask.dot(distances)/mask.sum(0))
            # print(min_dist)
            out = bh.nonzero(min_dist[:,None] == distances.T)[1]
            # print(out)
            # out2 = np.nonzero(distances2 == min_dist2)
            # print(out2)
            # out = bh.array(out).T

            # print(out[out[:,1].argsort()][:,0])
            # print(out[bh.argsort(out[:,1])][:,0])
            # print(out[(-out[:,1]).argsort()])
            # print(bh.sort(out.view('f8,f8,f8'), order =['f1'], axis = 0))

            # print(min_dist.shape)
            # bh.flush()
            # start = time.time()
            result =  bh.nonzero(distances.T==min_dist[:,None])[1]
            # bh.flush()
            # print("time2: ", time.time() -start)

            # print(result)
            print(result2)

            # print((result2==result).all())

            # print("idx ", idx[0])
            # print(min_dist)

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
        closest, centroids, iterations, iner = self.run(points)

        points = points.copy2numpy()
        centroids = centroids.copy2numpy()
        closest = closest.copy2numpy()

        plt.scatter(points[:,0], points[:,1], marker = ".", s=50, c = closest)
        plt.scatter(centroids[:,0], centroids[:,1], marker = "X", s=400, c = 'r')
        if self.init_centroids.size > 0:
            plt.scatter(self.init_centroids[:,0], self.init_centroids[:,1], marker = "X", s=400, c = 'b')
        plt.show()



    def move_centroids(self, points, closest, centroids, n = 0 ):

        if not n:
            n = self.k

        mask = (closest == bh.arange(n)[:,None])
        out = mask.dot(points)/ mask.sum(1)[:,None]
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



        if type(points) != 'numpy.float32':
            points = bh.float64(points)


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


        while iterations < self.max_iter:

            if iterations > 0:

                old_centers = centroids.copy()
                old_min_dist = min_dist.copy()
                old_closest = closest.copy()

            closest, min_dist = self.centroids_closest(points, centroids)
            centroids = self.move_centroids(points, closest, centroids)
            inertia = min_dist.sum()

            inertia = inertia.copy2numpy()


            if iterations > 0:

                x = old_centers - centroids
                x = bh.ravel(x)

                #This is the SKlearn way of exiting.
                if (bh.dot(x, x) <= epsilon):
                    print("OLD CENTERS")
                    return closest, centroids, iterations, inertia


                # if (bh.sum(old_min_dist) - bh.sum(min_dist)) < epsilon:
                #     print("Difference less than threshold")
                #     return closest, centroids, iterations, inertia


            iterations += 1
        return closest, centroids, iterations, inertia

def benchmark():
    k = bench.args.size[0]
    points = bh.loadtxt("/home/chris/Documents/bachelor/data/birchgrid.txt")
    # points = np.random.randint(2 * 10**k, size=(10**k, 2))
    # points = bh.array(points)

    # print(10**k / 10)

    bh.flush()
    bench.start()

    kmeans = bohrium_kmeans(k, userkernel=True, init="random")
    kmeans.run(points)
    bh.flush()

    bench.stop()
    bench.pprint()


def gpu_bench():
    k = bench.args.size[0]
    gp = bench.args.size[1]

    np.random.seed(0)
    # points = bh.loadtxt("/home/chris/Documents/bachelor/data/birchgrid.txt")
    points = np.random.randint(2*10**6, size=(10**6, 2))
    points = bh.array(points)

    bh.flush()
    bench.start()

    kmeans = bohrium_kmeans(k, userkernel=True, init="random", gpu=gp)
    kmeans.run(points)
    bh.flush()

    bench.stop()
    bench.pprint()


if __name__ == "__main__":
    # from sklearn.cluster import KMeans
    # from bohrium_api import stack_info, _bh_api
    # print("here")
    # bench = util.Benchmark("kmeans", "k*gpu")
    # gpu_bench()
    points = bh.loadtxt("../data/birchgrid.txt")
    # points = np.random.randint(200000, size=(100000, 2))
    # points = bh.array(points)
    kmeans = bohrium_kmeans(100, userkernel=True, init="random", gpu=True)

    clos, cent, ite, iner = kmeans.run_plot(points)
    # centroids = kmeans.init_plus_plus(points)

    # closest, min_dist = kmeans.centroids_closest(points, centroids)
    # closest, min_dist = kmeans.centroids_closest(points, centroids)


    # start = time.time()
    # skmeans = KMeans(n_clusters = 100, n_init = 1, verbose =  1, ).fit(points)
    # end = time.time()
