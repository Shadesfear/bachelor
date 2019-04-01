#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


void vector_add(double *vector1,
	        double *vector2,
		double *result,
                int64_t dimensions) {

  for (int i = 0; i < dimensions; i++) {
    result[i] = vector1[i] + vector2[i];
  }
}

int get_index(int x, int y, int width) { return x + y * width; }







void execute(int64_t *k,
	     double *centroids,
	     int64_t *closest,
	     double *points,
             int64_t *size_closest) {


  //printf("%lu", sizeof(*closest)/sizeof(*closest[0]));


  // For each centroid
  for (int i = 0; i < k[0]; i++) {

    int counter = 0;
    int size_1 = k[0];

    int dim = 10;


    double temp[] = {10,0,0,0,0,0,0,0,0,0};


    // For each point
    for (int j = 0; j < size_closest[0]; j++) {

      // If point-j belongs to point k
      if (i == closest[j]) {
	printf("%f\n", temp[i]);
	vector_add(temp, &points[j], temp, dim);
	printf("%f\n", temp[i]);


	counter += 1;
      }
    }

    // temp /= counter;
    // for (int d = 0; d < dim; d++) {
    //	temp[d] /= counter;
    //}
    //centroids[i] = temp;
  }
}
