
import matplotlib.pyplot as plt
import bohrium as np

class bohrium_kmeans:

    def __init__(self, bhornp):
        print('init')
        # if bhornp == 'numpy':
        #     print('numpy')
        #     import numpy as np

        # elif bhornp == 'bohrium':
        #     print('bohrium')
        #     import bohrium as np


    def init_random_centroids(self, points, k):
        # centroids_copy = points.copy()
        # np.random.shuffle(centroids_copy)
        # return centroids_copy[:k]
        row_i = np.random.choice(points.shape[0], k)
        return points[row_i]

    def centroids_closest(self, points, centroids):
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        # distances = np.linalg.norm(points-centroids[:,np.newaxis], axis = 2)
        # print(centroids[:,None].shape)
        return np.argmin(distances, axis = 0)


    def move_centroids(self, points, closest, centroids):
        return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])


    def numpy_kmeans(self, points, k, epsilon=0.000000001):

        centroids = self.init_random_centroids(points, k)
        centroids_old = np.zeros(centroids.shape)

        iterations = 0

        while np.linalg.norm(centroids - centroids_old) > epsilon:

            centroids_old = centroids

            closest = self.centroids_closest(points, centroids)
            centroids = self.move_centroids(points, closest, centroids)
            iterations += 1

        return closest, centroids, iterations


if __name__ == "__main__":


    # points = np.loadtxt('dataset.txt')
    my_kmeans = bohrium_kmeans('bohrium')

    points = np.vstack(((np.random.randn(149, 2) * 0.75432 + np.array([1, 0])),
                (np.random.randn(50, 2) * 0.25212 + np.array([-0.5, 0.5])),
                (np.random.randn(50, 2) * 0.5443 + np.array([-0.5, -0.5]))))


    closest, centroids, iteraions = my_kmeans.numpy_kmeans(points, 4)
    print(closest)

    plt.scatter(points[:,0],points[:,1], c=closest)

    plt.scatter(centroids[:,0], centroids[:,1], marker = "X", s=400, c = 'r')

    plt.show()
