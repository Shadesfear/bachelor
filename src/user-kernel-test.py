import bohrium as bh

test_kernel =r"""


#include <stdint.h>

void execute(double *a, double *b, double *c) {
    a[0] = 10;
    for (uint64_t i=0; i<100; ++i) {
        c[i] = a[i]*a[i];
   }
}

"""


print(test_kernel)

a = bh.ones(100, bh.double)
b = bh.ones(100, bh.double)
res = bh.empty_like(a)
bh.user_kernel.execute(test_kernel, [a, b, res])


print(res)
