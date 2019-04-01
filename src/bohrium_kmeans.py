
import numpy as np
import matplotlib.pyplot as plt
import time

class bohrium_kmeans:

    def __init__(self, bhornp):
        print('\nStarting __init__')
        print('Done')


    def init_random_centroids(self, points, k):


        row_i = np.random.choice(points.shape[0], k)

        # print('Initialized ', k, ' random points, it took:', '{:.2e}'.format(end-start))
        self.init_clusters = points[row_i]

        return points[row_i]

    def init_plus_plus(self, points, k):
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

    def move_centroids(self, points, closest, centroids, k):

        return np.array([points[closest==k].mean(axis=0) for k in range(k)])

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

    def kmeans_vectorized(self, points, k, epsilon=1, mode = 'squared'):

        centroids = self.init_random_centroids(points, k)
        centroids_old = np.zeros(centroids.shape)
        iterations, diff = 0, epsilon+1

        avg_dist = []

        while diff > epsilon:

            centroids_old = centroids
            closest, min_dist = self.centroids_closest(points, centroids, mode)

            centroids = self.move_centroids(points, closest, centroids, k)
            avg_dist.append(np.mean(min_dist, axis = -1))

            if len(avg_dist) > 1:
                diff = avg_dist[-2] - avg_dist[-1]

            iterations += 1

        return closest, centroids, iterations


if __name__ == "__main__":

    # my_kmeans = bohrium_kmeans('bohrium')
    points = np.loadtxt('datasets/birchgrid.txt')

    # print('# of points', len(points))


    # closest, centroids, iteraions = my_kmeans.kmeans_vectorized(points, 100)


    with open('move_centroids.c', 'r') as content_file:
        kernel = content_file.read()


    closest = np.zeros(4, np.int)
    closest[0] = 0
    closest[1] = 1
    closest[2] = 1
    closest[3] = 1
    # points = np.ones(4, np.double)
    points = np.arange(10)
    points = points.reshape(2,5)
    points = points.astype(float)
    centroids = np.zeros(2, np.double)

    k = np.zeros(1, np.int)
    g = np.zeros(1, np.int)
    g[0] = len(closest)
    k[0]=2


    points = np.loadtxt('datasets/dim032.txt')
    my_kmeans = bohrium_kmeans('bohrium')
    new_points = my_kmeans.scale_data(points)

    # closest, centroids, itera = my_kmeans.kmeans_vectorized(new_points, 10)




    # a = np.ones(100, np.double)
    # b = np.ones(100, np.double)
    # print(a)
    # res = bh.empty_like(a)
    # bh.user_kernel.execute(kernel, [a, b, res])
    # print(closest)
    # print(k)
    # np.user_kernel.execute(kernel, [k, centroids, closest, points, g])
    # print(a)

    # my_kmeans.init_plus_plus(points[:10], 3)

    # print('{:.2e}'.format(end-start))

    # plt.scatter(centroids[:,0], centroids[:,1], marker = "X", s=400, c = 'r')
    # plt.scatter(my_kmeans.init_clusters[:,0], my_kmeans.init_clusters[:,1], marker = "X", s=100, c = 'k')
