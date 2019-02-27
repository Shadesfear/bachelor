

import matplotlib.pyplot as plt
import time
import bohrium as np
import timeit

class bohrium_kmeans:

    def __init__(self, bhornp):
        print('init')


    def init_random_centroids(self, points, k):
        row_i = np.random.choice(points.shape[0], k)
        return points[row_i]


    def centroids_closest(self, points, centroids):

        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis = 0)



    def move_centroids(self, points, closest, centroids, k):



        return np.array([points[closest==k].mean(axis=0) for k in range(k)])

        #
        # May not be possible in Bohrium, but seems to give faster speeds that before
        #

        # sums = np.zeros((closest.max()+1,points.shape[1]),dtype=float)
        # np.add.at(sums,closest,points)
        # return sums/np.bincount(closest).astype(float)[:,None]


    def numpy_kmeans(self, points, k, epsilon=0.000000001):

        centroids = self.init_random_centroids(points, k)
        centroids_old = np.zeros(centroids.shape)

        iterations = 0

        def convergence_loop():
            nonlocal points
            nonlocal centroids

            distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
            closest = np.argmin(distances, axis = 0)
            centroids = np.array([points[closest==k].mean(axis=0) for k in range(k)])

            return np.linalg.norm(centroids - centroids_old) > epsilon

        while np.linalg.norm(centroids - centroids_old) > epsilon:

            centroids_old = centroids

            closest = self.centroids_closest(points, centroids)

            centroids = self.move_centroids(points, closest, centroids, k)

            iterations += 1

        # np.do_while(convergence_loop, None)


        return closest, centroids, iterations


if __name__ == "__main__":

    my_kmeans = bohrium_kmeans('bohrium')

    points = np.vstack(((np.random.randn(150, 2) * 0.75432 + np.array([1, 0])),
                        (np.random.randn(150, 2) * 0.25212 + np.array([-0.5, 0.5])),
                        (np.random.randn(150, 2) * 0.5443 + np.array([-0.5, -0.5]))))

    start = time.time()
    closest, centroids, iteraions = my_kmeans.numpy_kmeans(points, 3)
    end = time.time()

    print('{:.2e}'.format(end-start))

    # plt.scatter(points[:,0],points[:,1], c=closest)

    # plt.scatter(centroids[:,0], centroids[:,1], marker = "X", s=400, c = 'r')

    # plt.show()
