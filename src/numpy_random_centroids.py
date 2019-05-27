import random
import numpy as np
import time



def benchmarks():
    k = bench.args.size[0]
    k = eval(k)
    bench.start()

    centroids=points[ np.random.choice(points.shape[0], size=k, replace=False)]
    # centroids = points[row_i]
    bench.stop()
    bench.pprint()



if __name__=="__main__":

    from benchpress.benchmarks import util
    np.random.seed(0)
    points = np.random.randint(2*10**8, size=(10**8, 2))

    bench = util.Benchmark("kmeans", "k", delimiter="å")

    benchmarks()
