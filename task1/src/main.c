
#include <iso646.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "main.h"

/*
Setting the number of threads:
[ export OMP_NUM_THREADS=8 ]

h := size/(N+1)
*/

static inline double **_alloc_u(size_t N) {
  double **u = calloc(N, sizeof(double *));

  if (u == NULL)
    return NULL;

  for (size_t i = 0; i <= N; i++) {
    u[i] = calloc(N, sizeof(double));

    if (u[i] == NULL) {

      for (size_t j = 0; j < i; j++)
        free(u[i]);

      free(u);
      return NULL;
    }
  }
  return u;
}

static inline void _free_u(size_t N, double **u) {
  for (size_t i = 0; i <= N; i++)
    free(u[i]);
  free(u);
}

static inline pb_parms *init(size_t N, double grid_size, double eps,
                             point l_down, point r_up, func f) {
  pb_parms *pb = calloc(1, sizeof(pb_parms));

  if (pb == NULL)
    return pb;

  pb->N = N + 1;
  pb->eps = eps;
  pb->grid_size = grid_size;
  pb->f = f;

  // TODO: validate!
  pb->l_down = l_down;
  pb->r_up = r_up;

  pb->h = grid_size / pb->N;
  pb->u = _alloc_u(pb->N);

  if (pb->u == NULL)
    return NULL;

  return pb;
}

pb_parms *approximate_values(uint32_t N, double grid_size, double eps,
                             point l_down, point r_up, func f) {

  pb_parms *pb = init(N, grid_size, eps, l_down, r_up, f);
  
  if(pb == NULL) return pb;


  return NULL;
}

void free_results(pb_parms *) {
  // TODO:
}

int main(int argc, char *argv[]) { return 0; }