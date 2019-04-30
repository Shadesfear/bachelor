
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


//int get_index(int x, int y, int width) { return x + (y * width); }

typedef struct idx_min {
  int idx;
  double min;
} idx_min;

struct idx_min argmin(double *array, int end)
{
  double minimum = array[0];
  int index;

  for (int j = 0; j < end; j++)
  {
    if (array[j] < minimum)
    {
      minimum = array[j];
      index = j;
    }
  }
  idx_min idxmin = {index, minimum};

  return idxmin;
}

void execute(double *dist, double * res_min, int64_t *res)
{
  int n_points = 0;
  int n_k = 0;

  #pragma omp parallel for
  for (int i = 0; i < n_points; i++)
  {
    //int row_index = get_index(0, i, n_k);
    int row_index = i * n_k;
    struct idx_min idxmin = argmin(&dist[row_index], n_k);

    res[i] = idxmin.idx;
    res_min[i] = idxmin.min;

  }
}
