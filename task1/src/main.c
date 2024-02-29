
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

#define x_ij(i, j, pb) (i * pb->h + pb->shift.x)
#define y_ij(i, j, pb) (i * pb->h + pb->shift.y)

#define b_k0(bk, bl_size) (bk * bl_size + 1)
#define b_kn(k0, bl_size, pb) min(k0 + bl_size, pb->size - 1)

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

static inline void init_matrix(pb_parms *pb, func f, func d2f) {

  for (size_t i = 0; i <= pb->size; i++) {
    for (size_t j = 0; j <= pb->size; j++) {
      if ((i == 0) || (j == 0) || (i == pb->size) || (j == pb->size)) {
        pb->u[i][j] = d2f(x_ij(i, j, pb), y_ij(i, j, pb));
      } else {
        pb->u[i][j] = 0;
      }

      pb->f[i][j] = f(x_ij(i, j, pb), y_ij(i, j, pb));
    }
  }
}

static inline pb_parms *create_pb(size_t N, double grid_size, double eps,
                                  point l_down) {
  pb_parms *pb = calloc(1, sizeof(pb_parms));

  if (pb == NULL)
    return pb;

  pb->size = N + 1;
  pb->grid_size = grid_size;
  pb->eps = eps;
  pb->h = grid_size / pb->size;
  pb->shift = l_down;

  pb->u = _alloc_u(pb->size);
  pb->f = _alloc_u(pb->size);

  if (pb->u == NULL || pb->f == NULL)
    return NULL;

  return pb;
}

static inline double process_block(pb_parms *pb, size_t bx, size_t by,
                                   size_t bl_size) {
  size_t i0 = b_k0(bx, bl_size);
  size_t in = b_kn(i0, bl_size, pb);
  size_t j0 = b_k0(by, bl_size);
  size_t jn = b_kn(j0, bl_size, pb);
  double dm = 0;

  for (size_t i = i0; i < in; i++) {
    for (size_t j = j0; j < jn; j++) {
      double temp = pb->u[i][j];

      pb->u[i][j] = u_ij(i, j, pb);

      double d = fabs(temp - pb->u[i][j]);
      if (dm < d)
        dm = d;
    }
  }

  return dm;
}

pb_parms *approximate_values(uint32_t N, double grid_size, double eps,
                             point l_down, func f, func d2f) {

  pb_parms *pb = create_pb(N, grid_size, eps, l_down);
  double **un = _alloc_u(N);
  size_t iter = 0;

  if (pb == NULL || un == NULL)
    return NULL;
  init_matrix(pb, f, d2f);

  return NULL;
}

void free_pb(pb_parms *) {
  // TODO:
}

int main(int argc, char *argv[]) { return 0; }