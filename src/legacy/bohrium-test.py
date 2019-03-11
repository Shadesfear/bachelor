import bohrium as np

class bohrium_kmeans:
    def __init__(self):
        pass

    def init_random_centroids(self, points, k):
        # row_i = np.random.choice(points.shape[0], k)
        # return points[row_i]
        a =
        return

my_bohrium_kmeans = bohrium_kmeans()

points = np.vstack(((np.random.randn(149, 2) * 0.75432 + np.array([1, 0])),
            (np.random.randn(50, 2) * 0.25212 + np.array([-0.5, 0.5])),
            (np.random.randn(50, 2) * 0.5443 + np.array([-0.5, -0.5]))))

print(my_bohrium_kmeans.init_random_centroids(points, 3))
