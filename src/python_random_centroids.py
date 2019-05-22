import random
import numpy as np



def benchmarks():
    k = bench.args.size[0]

    points = np.loadtxt("../../data/birchgrid.txt")


    bench.start()
    centroids = [points[random.randint(0, len(points)-1)] for i in range(k)]
    bench.stop()
    bench.pprint()



if __name__=="__main__":


    from benchpress.benchmarks import util


    bench = util.Benchmark("kmeans", "k")

    benchmarks()
