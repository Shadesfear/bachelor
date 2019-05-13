
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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


void execute(global double *dist, global int64_t *res, global double *min_dist)
{


  int n_points = 0;
  int n_k = 0;

  //for (int i = 0; i < n_points; i++)
  //{
  int i = get_global_id(0);
  int row_index = i * n_k;

  res[i] = argmin(&dist[row_index], n_k);
  min_dist[i] = dist[res[i] + row_index];

//}
}
