import bohrium as np

def partial_argsort(a):
    size = a.max()+1

    idar = np.zeros(int(size))

    idar[a] = np.arange(len(a))
    sortedd = a.sort()

def argmin_0(a):
    # Define a scaling array to scale each col such that each col is
    # offsetted against its previous one
    s = (a.max()+1)*np.arange(a.shape[1])

    # Scale each col, flatten with col-major order. Find global partial-argsort.
    # With the offsetting, those argsort indices would be limited to per-col
    # Subtract each group of ncols elements based on the offsetting.
    m,n = a.shape
    a1D = (a+s).T.ravel()
    return partial_argsort(a1D)[::m]-m*np.arange(n)

# np.random.seed(0)
a = np.random.randint(11,9999,(1000,1000))
idx0 = argmin_0(a)
# idx1 = a.argmin(0)
# r = np.arange(len(idx0))
#print(a[idx0,r] == a[idx1,r]).all()
