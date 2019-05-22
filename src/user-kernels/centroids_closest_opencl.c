#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/*int argmin(global double *array, int end)
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
  }*/


/*
Function calls does not work on OPENCL on the current version on the Nelson Machine.

 */

kernel void execute(global double *dist, global long *res, global double *min_dist)
{
  int n_k = 0;
  int n_points = 0;
  int i = get_global_id(0);

  if (i > n_points) {
    return;
  }

  int row_index = i * n_k;


  double minimum = dist[row_index];
  int index;

  for (int j = 0; j < n_k; j++)
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
