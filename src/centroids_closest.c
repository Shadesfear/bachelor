
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>



int get_index(int x, int y, int width) { return x + (y * width); }

int argmin(double *start, int end)
{
  double minimum = start[0];
  int index;

  for (int i = 1; i < end; ++i)
  {
    if (start[i] < minimum)
    {
      minimum = start[i];
      index = i;
    }
  }
  return index;
}

void execute(double *dist, int64_t *res)
{
  int n_points = 0;
  int n_k = 0;
  int i;

  #pragma omp parallel for
  for (i = 0; i < n_points; i++)
  {
    int row_index = get_index(0, i, n_k);
    int location = argmin(&dist[row_index], n_k);
    res[i] = location;
  }

}
