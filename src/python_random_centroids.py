import random
import numpy as np



def benchmarks():
    k = bench.args.size[0]

    k = eval(k)
    bench.start()
    centroids = [points[random.randint(0, len(points)-1)] for i in range(k)]
    bench.stop()
    bench.pprint()



if __name__=="__main__":


    from benchpress.benchmarks import util
    np.random.seed(0)
    points = np.random.randint(2*10**8, size=(10**8, 2))


    bench = util.Benchmark("kmeans", "k", delimiter = "å")

    benchmarks()
