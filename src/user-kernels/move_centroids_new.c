
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void execute(int64_t *labels, int64_t old_labels, double *points, double *centroids)
{

  int size_labels = 0;
  int dim = 0;
  int k = 0;
  int id = get_global_id(0);

  int counter = 0;
  double temp[dim[0]];

  //iter over labels
  for (int j = 0; j < size_labels; j++)
  {
    if (labels[j] == id)
    {
      double vect[dim];
      int row_index = j * dim;
      //memory ?

      for (int p = 0; p < dim; p++)
      {
	vect[p] = points[row_index + p];
      }

      for (int u = 0; u < dim; u++)
      {
	temp[u] = temp[u] + vect[u];
      }
    }
  }


  for (int d = 0; d < dim; d++)
  {
    if (counter > 0)
    {
      temp[d] /= counter;
    }
  }

  for (int j = 0; j < dim; j++)
  {
    int row_index = id * dim;

    centroids[row_index + j] = temp[j];
  }


}