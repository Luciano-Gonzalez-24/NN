#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

float Normal_dist(double mu, double sigma) {
  double stored_value = 0;
  double v1, v2, rsq, fac;
  if (stored_value == 0.0) {
    do {
      v1 = 2.0 * drand48() - 1.0;
      v2 = 2.0 * drand48() - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    stored_value = v1 * fac;
    return mu + sigma * v2 * fac;
  } else {
    fac = stored_value;
    stored_value = 0.0;
    return mu + sigma * fac;
  }
}

int main() {
  int n;
  fscanf(stdin, "%d", &n);
  float *results = (float *)calloc(n, sizeof(float));

  double start;
  double end;

  start = omp_get_wtime();

  for (int i = 0; i < n; i++) {
    results[i] = Normal_dist(0, 1);
  }

  double eps = 1e-7;
  int *quantiles = (int *)calloc(14, sizeof(int));
  for (int i = 0; i < n; i++) {
    if (results[i] <= -3.0 + eps)
      quantiles[0]++;
    else if (results[i] <= -2.5 + eps)
      quantiles[1]++;
    else if (results[i] <= -2.0 + eps)
      quantiles[2]++;
    else if (results[i] <= -1.5 + eps)
      quantiles[3]++;
    else if (results[i] <= -1.0 + eps)
      quantiles[4]++;
    else if (results[i] <= -0.5 + eps)
      quantiles[5]++;
    else if (results[i] <= 0.0 + eps)
      quantiles[6]++;
    else if (results[i] <= 0.5 + eps)
      quantiles[7]++;
    else if (results[i] <= 1.0 + eps)
      quantiles[8]++;
    else if (results[i] <= 1.5 + eps)
      quantiles[9]++;
    else if (results[i] <= 2.0 + eps)
      quantiles[10]++;
    else if (results[i] <= 2.5 + eps)
      quantiles[11]++;
    else if (results[i] <= 3.0 + eps)
      quantiles[12]++;
    else
      quantiles[13]++;
  }

  double time_taken = omp_get_wtime() - start; // Time in seconds

  printf("The generation took %f seconds to execute \n", time_taken);

  printf("\nDistribution Percentages:\n");
  for (int j = 0; j < 14; j++) {
    printf("%6.2lf%% ", 100.0 * ((double)quantiles[j] / (double)n));
  }

  end = omp_get_wtime();
  printf("With n = %d\n", n);
  printf("Work took %f seconds\n", end - start);

  return 0;
}