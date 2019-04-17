
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>



int get_index(int x, int y, int width) { return x + (y * width); }

void execute(double *array)
{
  srand(time(NULL));
  int rows;
  int cols;

  for (int i = rows-1; i > 0; i--)
  {

    int r = rand() % i;

    double temp[cols];
    memset(temp, 0, cols*sizeof(double));


    int row_index = get_index( 0 , i , cols);
    int random_row = get_index(0, r, cols);

    for (int j = 0; j < cols; j++)
    {
      temp[j] = array[random_row+j];
      array[random_row+j] = array[row_index+j];
      array[row_index+j] = temp[j];
    }
  }
}
