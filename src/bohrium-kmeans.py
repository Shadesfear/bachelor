
import numpy as np
import matplotlib.pyplot as plt
import time

class bohrium_kmeans:

    def __init__(self, bhornp):
        print('\nStarting __init__')
        print('Done')


    def init_random_centroids(self, points, k):

        # Initializes random points, indexing with arrays is not
        # Available in Bohrium. Maybe write a user-kernel. But since it's only
        # Initialized once, is there a reason?
        row_i = np.random.choice(points.shape[0], k)
        # print('Initialized ', k, ' random points, it took:', '{:.2e}'.format(end-start))
        self.init_clusters = points[row_i]

        return points[row_i]

    def init_plus_plus(self, points, k):

        #
        # Init the clusters according to Kmeans++
        #

        centroids = points[:k]
        centroids[0] = points[0]
        # print(points)
        print(centroids)
        distances = self.euclidian_distance(centroids[:1], points)
        print(np.argmax(distances))
        centroids[1]=points[7]
        distances = self.euclidian_distance(np.mean(centroids[:2]), points)
        print(np.argmin(distances))


        for i in range(1, k):
            pass
        print(distances)
        # print(points[5])




    def euclidian_distance(self, point1, point2):


        # diff = point1[None, :, :] - point2[:, None, :]
        # print(diff)
        distances = np.sqrt(((point1 - point2[:, np.newaxis])**2).sum(axis=2))

        return(distances)


    def centroids_closest(self, points, centroids):

        # diff = points[None, :, :] - centroids[:, None, :]

        # distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        distances = self.euclidian_distance(points, centroids)
        # print(distances)
        # distances2 = np.sqrt(np.sum(diff*diff, -1))

        min_dist = np.minimum.reduce(distances, 0)
        return np.argmin(distances, axis = 0), min_dist

    def move_centroids(self, points, closest, centroids, k):

        return np.array([points[closest==k].mean(axis=0) for k in range(k)])

        #
        # May not be possible in Bohrium, but seems to give faster speeds that before
        #

        # sums = np.zeros((closest.max()+1,points.shape[1]),dtype=float)
        # np.add.at(sums,closest,points)
        # return sums/np.bincount(closest).astype(float)[:,None]


    def kmeans_vectorized(self, points, k, epsilon=1):

        centroids = self.init_random_centroids(points, k)
        centroids_old = np.zeros(centroids.shape)
        iterations, diff = 0, epsilon+1

        avg_dist = []

        while diff > epsilon:

            centroids_old = centroids
            closest, min_dist = self.centroids_closest(points, centroids)
            print(closest)
            centroids = self.move_centroids(points, closest, centroids, k)
            avg_dist.append(np.mean(min_dist, axis = -1))

            if len(avg_dist) > 1:
                diff = avg_dist[-2] - avg_dist[-1]

            iterations += 1

        return closest, centroids, iterations


if __name__ == "__main__":

    my_kmeans = bohrium_kmeans('bohrium')
    points = np.loadtxt('datasets/birchgrid.txt')

    print('# of points', len(points))


    # closest, centroids, iteraions = my_kmeans.kmeans_vectorized(points, 100)








    with open('move_centroids.c', 'r') as content_file:
        kernel = content_file.read()


    kernel = r'''

#include <stdint.h>
#include <stdio.h>


void execute(int64_t *k, double *centroids) {

  if (k[0] == 1) {
    centroids[0] = 20;
      } else if (k[0]==2) {
    centroids[0] = 30;
      } else if (k[0]==3) {
    centroids[0] = 40;

  }
}



'''

    a = np.zeros(3, np.double)
    b = np.zeros(4, np.double)
    c = np.zeros(1, np.double)
    k = np.zeros(1, np.int)
    k[0]=1

    # a = np.ones(100, np.double)
    # b = np.ones(100, np.double)
    # print(a)
    # res = bh.empty_like(a)
    # bh.user_kernel.execute(kernel, [a, b, res])

    # np.user_kernel.execute(kernel, [k, c])
    # print(c)
    points = np.array([[0,0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    my_kmeans.init_plus_plus(points[:10], 3)
    # print(iteraions)


    # print('{:.2e}'.format(end-start))

    plt.scatter(points[:10,0], points[:10,1],c='r')
    plt.scatter(points[0,0], points[0,1], c='b')
    plt.scatter(points[7,0], points[7,1], c='b')
    plt.scatter(points[3,0], points[3,1], c='b')
    # plt.scatter(points[:,0],points[:,1], c=closest)
    # plt.scatter(centroids[:,0], centroids[:,1], marker = "X", s=400, c = 'r')
    # plt.scatter(my_kmeans.init_clusters[:,0], my_kmeans.init_clusters[:,1], marker = "X", s=100, c = 'k')

    plt.show()
