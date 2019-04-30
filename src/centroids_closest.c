
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

struct idx_min {
  int index;
  double minimum;
};

int get_index(int x, int y, int width) { return x + (y * width); }

struct idx_min argmin(double *array1, int end)
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
  tuple idx_min = {index, minimum};
  return idx_min;
}

void execute(double *dist, double *res_min, int64_t *res)
{
  int n_points = 0;
  int n_k = 0;


#pragma omp parallel for
  for (int i = 0; i < n_points; i++)
  {
    //int row_index = get_index(0, i, n_k);
    //printf("%d", omp_get_num_threads());
    //printf("Hello world \n")

    int row_index = i * n_k;
    struct idx_min = argmin(&dist[row_index], n_k);
    res[i] = idx_min.index;
    res_min[i] = idx_min.minimum
  }


}
