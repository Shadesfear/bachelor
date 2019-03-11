#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void execute(double k, double centroids[], double closest[], double points[]) {
  printf("started");

  int j;
  int i;

  for (i=0; i < k; ++i) {
    printf("Got here");


    for (j=0; i < sizeof(closest); ++j)

      if (k==closest[j]) {
	char str[50];
	snprintf(str,50,"%f", points[j]);
	printf("%s", str);

      }
  }
}

void main() {
  double k = 3;
  double closest[] = {0, 1, 2, 1};
  double points[4][2] = {{0, 0}, {1, 1}, {3, 3}, {5, 5}};
  double centroids[4][2] = {{0, 0}, {1, 1}};

  execute(k, centroids, closest, points);

}
