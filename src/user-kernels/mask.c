#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void execute(int64_t *labels, int64_t *mask)
{
  int rows = 0;
  int cols = 0;

  for (int i = 0; i < rows; i++)
  {
    int row_index = cols * i;

    for (int j = 0; j < cols; j++)
    {
      if (labels[j] == i)
      {
	mask[row_index + j] = 1;
      }
    }
  }
}
