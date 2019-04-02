
import bohrium as np
import matplotlib.pyplot as plt
import time

class bohrium_kmeans:

    def __init__(self, k, userkernel = True):
        # import numpy as np
        self.userkernel = userkernel
        self.k = k

    def timeit(func):
        def let_time(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            print('Function "{name}" took {time} seconds to complete.'.format(name=func.__name__, time=te-ts))
            return result
        return let_time


    def init_random_centroids(self, points):

        row_i = np.random.choice(points.shape[0], self.k)

        self.init_clusters = points[row_i]

        return points[row_i]

    def init_plus_plus(self, points):
        """
        Initialize the clusters using KMeans++

        """

        centroids = points[:k]
        centroids[0] = points[0]
        # print(points)
        print(centroids)
        distances = self.euclidian_distance(centroids[:1], points, mode = 'squared')
        print(np.argmax(distances))
        centroids[1]=points[7]
        distances = self.euclidian_distance(np.mean(centroids[:2]), points, mode = 'squared')
        print(np.argmin(distances))


        for i in range(1, k):
            pass
        print(distances)
        # print(points[5])


    def euclidian_distance(self, point1, point2, mode = 'squared'):

        if mode == 'squared':
            distances = ((point1 - point2[:, np.newaxis])**2).sum(axis=2)

        elif mode == 'normal':
            distances = np.sqrt(((point1 - point2[:, np.newaxis])**2).sum(axis=2))
        # diff = point1[None, :, :] - point2[:, None, :]
        # print(diff)
        return(distances)


    def centroids_closest(self, points, centroids, mode):

        # diff = points[None, :, :] - centroids[:, None, :]

        # distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))

        distances = self.euclidian_distance(points, centroids, mode)

        # distances2 = np.sqrt(np.sum(diff*diff, -1))

        min_dist = np.minimum.reduce(distances, 0)
        return np.argmin(distances, axis = 0), min_dist


    def move_centroids(self, points, closest, centroids):

        if self.userkernel:
            with open('move_centroids.c', 'r') as content_file:
                kernel = content_file.read()

            return np.user_kernel.execute(kernel, [self.k, centroids, closest, points, g])

        else:
            return np.array([points[closest==k].mean(axis=0) for k in range(self.k)])


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

    def run(self, points, epsilon=1, mode = 'squared'):

        centroids = self.init_random_centroids(points)
        centroids_old = np.zeros(centroids.shape)
        iterations, diff = 0, epsilon+1

        avg_dist = []

        while diff > epsilon:

            centroids_old = centroids
            closest, min_dist = self.centroids_closest(points, centroids, mode)

            centroids = self.move_centroids(points, closest, centroids)
            avg_dist.append(np.mean(min_dist, axis = -1))

            if len(avg_dist) > 1:
                diff = avg_dist[-2] - avg_dist[-1]

            iterations += 1
        return closest, centroids, iterations


if __name__ == "__main__":

    # my_kmeans = bohrium_kmeans('bohrium')
    # points = np.loadtxt('datasets/birchgrid.txt')

    # print('# of points', len(points))


    # closest, centroids, iteraions = my_kmeans.run(points, 100)


    with open('move_centroids.c', 'r') as content_file:
        kernel = content_file.read()


    closest = np.zeros(4, np.int)
    closest[0] = 0
    closest[1] = 0
    closest[2] = 1
    closest[3] = 1
    # points = np.ones(4, np.double)
    points = np.ones(20, np.double)
    points = points.reshape(4,5)
    points[3] = 5.5
    print(points)
    # points = points.astype(np.double)




    k = np.zeros(1, np.int)

    size_closest = np.zeros(1, np.int)
    size_closest[0] = len(closest)
    k[0]=2

    dim = np.array(len(points[0]))
    print(dim)

    centroids = np.zeros([k[0],dim], np.double)
    print(centroids)

    #print("Centroids before:\n ", centroids)
    # points = np.loadtxt('datasets/dim032.txt')
    my_kmeans = bohrium_kmeans(k[0], userkernel = False)
    # new_points = my_kmeans.scale_data(points)

    # closest, centroids, itera = my_kmeans.run(new_points, 10)

    # res = np.empty_like(a)



    st = time.time()
    hello = my_kmeans.move_centroids(points, closest, centroids)
    end = time.time()
    # print("Centroids after normal:\n ", hello)
    print("normal took ", end-st)

    st = time.time()
    np.user_kernel.execute(kernel, [k, closest, points, size_closest,dim, centroids])
    end = time.time()
    print("Kernel took, " ,end-st)
    print("\nCentroids after kernel:\n", centroids)
    # my_kmeans.init_plus_plus(points[:10], 3)

    # print('{:.2e}'.format(end-start))

    # plt.scatter(centroids[:,0], centroids[:,1], marker = "X", s=400, c = 'r')
    # plt.scatter(my_kmeans.init_clusters[:,0], my_kmeans.init_clusters[:,1], marker = "X", s=100, c = 'k')
