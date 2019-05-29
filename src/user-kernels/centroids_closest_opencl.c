#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void execute(__global double *dist, __global int *res, __global double *min_dist)
{
  int n_k = 0;
  int n_points = 0;
  int i = get_global_id(0);

/*  if (i > n_points) {
    return;
    }*/

  int row_index = i * n_k;

  double minimum = dist[row_index];
  int index;

  for (int j = 1; j < n_k; j++)
  {
    if (dist[j + row_index] < minimum)
    {
      minimum = dist[j + row_index];
      index = j + row_index;
    }
  }

  res[i] = index;
  min_dist[i] = dist[res[i] + row_index];
}
