#pragma OPENCL EXTENSION cl_khr_fp64 : enable

int argmin(global double *array, int end)
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

kernel void execute(global double *dist, global long *res, global double *min_dist)
{
  int n_points = 0;
  int n_k = 0;
  int i = get_global_id(0);

  if (i > n_points) {
    return;
  }

  int row_index = i * n_k;
  res[i] = argmin(&dist[row_index], n_k);
  min_dist[i] = dist[res[i] + row_index];

}
