
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int argmin(double *array, int end)
{
  double minimum = array[0];
  int index;

  for (int j = 1; j < end; j++)
  {
    if (array[j] < minimum)
    {
      minimum = array[j];
      index = j;
    }
  }
  return index;
}


void execute(double *dist, int64_t *res, double *min_dist)
{
  int n_points = 0;
  int n_k = 0;

  #pragma omp parallel for
  for (int i = 0; i < n_points; i++)
  {
    int row_index = i * n_k;
    res[i] = argmin(&dist[row_index], n_k);
    min_dist[i] = dist[res[i] + row_index];

  }
}
