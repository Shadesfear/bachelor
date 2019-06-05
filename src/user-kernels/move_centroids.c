#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

void vector_add(double *v1,
	        double *v2,
		double *res,
                int64_t dim)
{

  for (int i = 0; i < dim; i++)
  {
    res[i] = v1[i] + v2[i];
  }
}

int get_index(int x, int y, int width) { return x + (y * width); }


void execute(int64_t *k,
	     int64_t *closest,
	     double *points,
             int64_t *size_closest,
	     int64_t *dim,
	     double *centroids)
{
  // For each centroid
#pragma omp parallel for
  for (int i = 0; i < k[0]; i++)
  {
    int counter = 0;
    double temp[dim[0]];
    memset(temp, 0, dim[0]*sizeof(double));
    // For each label in closest
    #pragma omp parallel for
    for (int j = 0; j < size_closest[0]; j++)
    {
      // If point-j belongs to point k
      if (i == closest[j])
      {
	//Populates vect
	double vect[dim[0]];
	memset(vect, 0, dim[0]*sizeof(double));

	for (int p = 0; p < dim[0]; ++p)
	{
	  //  printf("%f = ", points[get_index(p, j, dim[0])]);

	  vect[p] = points[get_index(p, j, dim[0])];
//	  printf("%f \n", vect[p]);

	}
	vector_add(temp, vect, temp, dim[0]);
	counter += 1;
      }
    }

    /*
      Dividing by counter to get mean
    */
    #pragma omp parallel
    for (int d = 0; d < dim[0]; d++)
    {
      if (counter > 0)
      {
	temp[d] /= counter;
      }
    }
    /*
      Looping over each row in
    */
      for (int j = 0; j < dim[0]; ++j)
      {
	centroids[get_index(j, i, dim[0])] = temp[j];
//	printf("%f\n", centroids[get_index(j, i, dim[0])]);

      }
  }
}
