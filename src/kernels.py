
test_kernel =r"""


#include <stdint.h>

void execute(double *a, double *b, double *c) {

    for (uint64_t i=0; i<100; ++i) {
        c[i] = a[i]*b[i]

}
}

"""
