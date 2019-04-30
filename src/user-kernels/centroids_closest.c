
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


//int get_index(int x, int y, int width) { return x + (y * width); }

int argmin(double *array, int end)
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
  return index;
}

void execute(double *dist, int64_t *res)
{
  int n_points = 0;
  int n_k = 0;

  double start = omp_get_wtime();

  #pragma omp parallel for
  for (int i = 0; i < n_points; i++)
  {
    //int row_index = get_index(0, i, n_k);
    int row_index = i * n_k;
    res[i] = argmin(&dist[row_index], n_k);
  }
  printf("%f", omp_get_wtime()-start);
}
