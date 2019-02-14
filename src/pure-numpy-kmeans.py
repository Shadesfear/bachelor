# Keep number of imports short to show that only numpy was used
#
#

import numpy as np
import matplotlib.pyplot as plt

def pure_numpy_kmeans(points, k, epsilon = 0.00001):

    if not type(points) is np.ndarray:
        points = np.array(points)
    else:

        number_points, number_dim = points.shape

        row_i = np.random.choice(points.shape[0], k) #K random rows
        centroids=points[row_i,:]
        centroids_old, belongs_to = np.zeros(centroids.shape), np.zeros(number_points)


        colors = ['r', 'b', 'g', 'k']
        iterations = 0

        plt.scatter(*zip(*points), marker='o', color = 'r')
        plt.scatter(*zip(*centroids), marker = 'x', s=600, color = 'k')

        diff = np.linalg.norm(centroids_old[0,:] - centroids[0,:])

        while diff > epsilon:

            iterations += 1
            diff = np.linalg.norm(centroids_old[0,:] - centroids[0,:])

            centroids_old = centroids #Update old for the diff

            #################################################
            # FIX ENUMERATE TO NUMPY, probably slow in python
            #################################################

            for p_x, point in enumerate(points):
                distance_array = np.zeros(k)

                for c_x, centroid in enumerate(centroids):

                    dist = np.linalg.norm(point-centroid)
                    distance_array[c_x] = dist

                belongs_to[p_x] = np.argmin(distance_array)

            for index in range(k):

                instance_closest = [i for i in range(belongs_to.size) if belongs_to[i] == index]
                new_centroid = np.mean(points[instance_closest], axis=0)
                plt.scatter(*zip(*points[instance_closest]), marker = 'o', color=colors[index])
                centroids[index,:] = new_centroid

    print(iterations)
    for i in range(k):
        plt.scatter(*zip(centroids[i]), marker = 'x', color = colors[i], s=600)

    plt.show()

    print("done")



if __name__ == "__main__":
    #Get points from text file
    pnt_ary = np.loadtxt('dataset.txt')
    pure_numpy_kmeans(pnt_ary, 3)
