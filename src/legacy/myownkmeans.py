import time
import numpy as np
import matplotlib.pyplot as plt
import math
from random import *
import random

from benchpress.benchmarks import util


bench = util.Benchmark("kmeans", "k")


def mykmeans(data, k, epsilon = 0.000001):

    centroids = [data[randint(0, len(data)-1)] for i in range(k)]
    centroids_old = [[0,0]]*k

    belongs_to = [0]*len(data)
    iterations = 0



    diff = sum([eucDistance(centroids[i],centroids_old[i]) for i in range(k)])
    print("diff", diff)

    while diff > epsilon:

        diff = sum([eucDistance(centroids[j],centroids_old[j]) for j in range(k)])
        print("diff", diff)
        iterations += 1 #keep track of how many times we had to iterate
        # print(iterations)

        centroids_old =  centroids



        for p_i, point in enumerate(data):
            dst_ary = [0]*k
            for c_i, centroid in enumerate(centroids):
                #Used my own distance function
                dst = eucDistance(point, centroid)
                dst_ary[c_i] = dst

            #We need to find the index of the shortest distance, in numpy this is easy with argmin.
            belongs_to[p_i] = dst_ary.index(min(dst_ary))

        for index in range(k):

            instance_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            points_to_mean = [data[instance_close[i]] for i in range(len(instance_close))]

            vec = [0]*len(centroids[1])

            for point in points_to_mean:
                vec = [vec[i] + point[i] for i in range(len(point))]


            mean_point = [i/len(points_to_mean) for i in vec if len(points_to_mean)]

            centroids[index] = mean_point
        return centroids, iterations, belongs_to


def eucDistance(p_i, p_j):
    if len(p_i) != len(p_j):
        raise ValueError('Points must have same dimension')
    else:
        result = math.sqrt( sum( [ ( p_j[i] - p_i[i] ) ** 2 for i in range(len(p_i)) ] ) )
        # print(result)
        return result

def benchmarks():
    k = bench.args.size[0]

    points = np.loadtxt("../../data/birchgrid.txt")

    points = np.random.randint(2 * 10**k, size=(10**k, 2))

    bench.start()
    kmeans = mykmeans(points, int(10**k / 100))
    bench.stop()
    bench.pprint()




if __name__ == "__main__":
    print("hello")
    # n_points = 10
    # point_ary = [[randint(0,10), randint(0,10)] for i in range(n_points)]
    # point_ary1 = []

    # points = np.loadtxt('../../data/birchgrid.txt')




    # start = time.time()
    # centroids, iterations, closest = mykmeans(points,1000)
    # end = time.time()

    # plt.scatter(*zip(*points), c=closest)
    # plt.scatter(*zip(*points), marker = "X", s=400, c = 'r')

    # # plt.show()
    # print(iterations)

    # print("Tid: ", end-start)
    benchmarks()
