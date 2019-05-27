void execute(int_64t *seed, int_64t *result)
{
  int m = 2147483647;
  int a = 1103515245;
  int c = 1;

  result = (a * seed * c) % m;

}
