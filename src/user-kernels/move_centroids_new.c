
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void execute(global long *labels, global long *old_labels, global double *points, global double *result)
{

  int size_labels = 0;
  int dim = 0;
  int k = 0;
  int id = get_global_id(0);

  int counter = 0;
  double temp[dimm] = {0};
  

  //iter over labels
  for (int j = 0; j < size_labels; j++)
  {
    if (labels[j] == id)
    {
      int row_index = j * dim;
      for (int p = 0; p < dim; p++)
      {
	temp[p] = temp[p] +  points[row_index + p];
      }
      counter++;
    }
  }


  for (int d = 0; d < dim; d++)
  {
    if (counter > 0)
    {
      temp[d] /= counter;
    }
    int row_index = id * dim;
    result[row_index + d] = temp[d];

  }
}
