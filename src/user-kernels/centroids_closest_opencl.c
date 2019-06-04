#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void execute(global double *dist, global double *min_dist, global long *res)
{
  int n_k = 0;
  int n_points = 0;
  int i = get_global_id(0);

/*  if (i > n_points) {
    return;
    }*/

  int row_index = i * n_k;

  double minimum = dist[row_index];
  int index = 0;

  for (int j = 0; j < n_k; j++)
  {

    if (dist[j + row_index] < minimum)
    {
      minimum = dist[j + row_index];
      index = j;
    }
  }


  res[i] = index;
  min_dist[i] = dist[res[i]];
}
