import random
import numpy as np



def benchmarks():
    k = bench.args.size[0]

    points = np.loadtxt("../../data/birchgrid.txt")


    bench.start()

    row_i = np.random.choice(points.shape[0], k)
    centroids = points[row_i,:]
    bench.stop()
    bench.pprint()



if __name__=="__main__":

    from benchpress.benchmarks import util


    bench = util.Benchmark("kmeans", "k")


    benchmarks()
