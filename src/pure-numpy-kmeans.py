# Keep number of imports short to show that only numpy was used
#
#
from benchpress.benchmarks import util
import numpy as np
import matplotlib.pyplot as plt


def move_centroids(points, closest, centroids, k):
    mask = (closest == np.arange(k)[:,None])
    out = mask.dot(points)/ mask.sum(1)[:,None]
    return out


def euclidian_distance(points, centroids):
        X = points - centroids[:, None]
        distances = (X * X).sum(axis=2)
        return distances


def centroids_closest(points, centroids):
    distances = euclidian_distance(points, centroids)
    result = np.argmin(distances, axis = 0)
    min_dist = distances.min(1)
    return result, min_dist

def pure_numpy_kmeans(points, k, epsilon = 0.00001):

    if not type(points) is np.ndarray:
        points = np.array(points)
    else:

        number_points, number_dim = points.shape
        row_i = np.random.choice(points.shape[0], k) #K random rows
        centroids=points[row_i,:]
        centroids_old, belongs_to = np.zeros(centroids.shape), np.zeros(number_points)
        iterations = 0



        diff = np.linalg.norm(centroids_old - centroids)

        while iterations < 300:

            iterations += 1
            diff = np.linalg.norm(centroids_old - centroids)

            centroids_old = centroids #Update old for the diff


            closest, min_dist = centroids_closest(points, centroids)
            centroids = move_centroids(points, closest, centroids, k)

            x = centroids_old - centroids
            x = np.ravel(x)
            if (np.dot(x, x) <= epsilon):
                print(iterations)
                return

            # for p_x, point in enumerate(points):
            #     distance_array = np.zeros(k)

            #     for c_x, centroid in enumerate(centroids):

            #         dist = np.linalg.norm(point-centroid)
            #         distance_array[c_x] = dist

            #     belongs_to[p_x] = np.argmin(distance_array)

            # for index in range(k):

            #     instance_closest = [i for i in range(belongs_to.size) if belongs_to[i] == index]
            #     new_centroid = np.mean(points[instance_closest], axis=0)
            #     # plt.scatter(*zip(*points[instance_closest]), marker = 'o', color=colors[index])
            #     centroids[index,:] = new_centroid
        print(iterations)



def benchmark():
    k = bench.args.size[0]
    points = np.loadtxt("/home/chris/Documents/bachelor/data/birchgrid.txt")


    bench.start()
    print("starting")
    pure_numpy_kmeans(points, k)

    bench.stop()
    bench.pprint()


    print("done")



if __name__ == "__main__":

  # points = np.loadtxt("/home/chris/Documents/bachelor/data/birchgrid.txt")
    # pure_numpy_kmeans(points, 100)
    #Get points from text file
    bench = util.Benchmark("kmeans", "k")
    # pnt_ary = np.loadtxt('../../data/birchgrid.txt')
    benchmark()
