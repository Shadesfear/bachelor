import bohrium_kmeans
import bohrium as bh

k = 2000

bh.random.seed(0)
# points = bh.random.randint(2*2*10**5, size=(2*10**5, 2), dtype=bh.float64)
# points = points[:10]


points = bh.loadtxt("../data/birchgrid.txt")
points = points
kmeans = bohrium_kmeans.bohrium_kmeans(k, userkernel=True, init="random", gpu=False, verbose=True)

# kmeans.kernel_centroids_closest = kmeans.kernel_centroids_closest.replace("int n_points = 0", "int n_points = " + str(points.shape[0]))
# kmeans.kernel_centroids_closest = kmeans.kernel_centroids_closest.replace("int n_k = 0", "int n_k = " + str(kmeans.k))
kmeans.kernels_replace(points)

# centroids = kmeans.init_plus_plus(points)
centroids = kmeans.init_random_userkernel(points)
labels, min_dist = kmeans.centroids_closest(points, centroids)

centroids = kmeans.move_centroids(points, labels, centroids)
