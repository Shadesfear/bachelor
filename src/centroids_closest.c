
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>



int get_index(int x, int y, int width) { return x + (y * width); }

void execute(double *dist, int64_t *res)
{
  int n_points = 0;
  int n_k = 0;
  int i;

#pragma omp parallel
   {
    double minimum;
    int location = 0;

   #pragma omp for
    for (i = 0; i < n_points; i++)
    {
      int row_index = get_index(0, i, n_k);
      minimum = dist[row_index];
      location = 0;


      for (int j = 1; j < n_k; j++)
      {
	if (dist[row_index+j] < minimum)
	{
	  minimum = dist[row_index+j];
	  location = j;
	}
      }

      res[i] = location;

    }
    }
}
