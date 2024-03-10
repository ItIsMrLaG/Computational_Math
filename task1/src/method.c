#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "method.h"

void free_matrix(double **u, int64_t N) {
  for (int64_t i = 0; i < N; i++)
    free(u[i]);
  free(u);
}

double **alloc_matrix(int64_t N) {
  double **u = calloc(N, sizeof(double *));

  if (u == NULL)
    return NULL;

  for (int64_t i = 0; i < N; i++) {
    u[i] = calloc(N, sizeof(double));
    if (u[i] == NULL) {
      free_matrix(u, i);
      return NULL;
    }
  }
  return u;
}

void init_matrixes(func_R2 f, func_R2 u, problem *pb) {
  srand(time(NULL));

  int64_t mod = 0;
  int n = 0;

  for (int64_t i = 0; i < pb->size; i++) {
    for (int64_t j = 0; j < pb->size; j++) {
      if ((i == 0) || (j == 0) || (i == pb->size - 1) || (j == pb->size - 1)) {
        pb->u[i][j] = u(x_i(i, pb), y_j(j, pb));
        n += 1;
        mod = (mod * (n - 1) + (int64_t)pb->u[i][j]) / n;
      }
      pb->f[i][j] = f(x_i(i, pb), y_j(j, pb));
    }
  }

  for (int64_t i = 1; i < pb->size - 1; i++) {
    for (int64_t j = 1; j < pb->size - 1; j++) {
      if (mod == 0)
        pb->u[i][j] = 0;
      else
        pb->u[i][j] = mod;
    }
  }

  pb->max_init = mod;
}

double **alloc_copy(double **u, size_t N) {
  double **copy = alloc_matrix(N);
  if (copy == NULL)
    return NULL;

  for (size_t i = 0; i < N; i++)
    memcpy(copy[i], u[i], N * sizeof(double));

  return copy;
}

int init_pb(double eps, int64_t N, int64_t bs, problem *pb) {

  if (N < 2)
    return -1;

  pb->size = N;

  pb->f = alloc_matrix(pb->size);
  if (pb->f == NULL)
    return -1;

  pb->u = alloc_matrix(pb->size);
  if (pb->u == NULL) {
    free_matrix(pb->f, pb->size);
    return -1;
  }

  pb->h = 1.0 / (N - 1);
  pb->eps = eps;
  pb->iters = 0;
  pb->l = 1;
  pb->bs = bs;

  return 0;
}

void free_pb(problem *pb) {
  if (pb->u != NULL)
    free_matrix(pb->u, pb->size);
  if (pb->f != NULL)
    free_matrix(pb->f, pb->size);
  free(pb);
}

double process_block(int64_t bi, int64_t bj, problem *pb) {
  int64_t i_0, j_0, i_n, j_n;

  i_0 = 1 + bi * pb->bs;
  j_0 = 1 + bj * pb->bs;

  i_n = MIN(i_0 + pb->bs, pb->size - 1);
  j_n = MIN(j_0 + pb->bs, pb->size - 1);

  double d, temp, dm = 0;

  for (int64_t i = i_0; i < i_n; i++) {
    for (int64_t j = j_0; j < j_n; j++) {
      temp = pb->u[i][j];
      pb->u[i][j] = u_ij(i, j, pb);

      d = fabs(temp - pb->u[i][j]);

      if (dm < d)
        dm = d;
    }
  }
  return dm;
}

int process_blocks(problem *pb) {
  int64_t N = pb->size - 2;
  int64_t NB = CEIL_DIV_UP(N, pb->bs);

  double dmax = 0;
  double *dm = calloc(NB, sizeof(double));
  if (dm == NULL)
    return -1;

  do {
    dmax = 0;

    for (int64_t nx = 0; nx < NB; nx++) {
      int64_t bi, bj;
      double res;
      dm[nx] = 0;

#pragma omp parallel for shared(nx, pb, dm) private(bi, bj, res)
      for (bi = 0; bi < nx + 1; bi++) {
        bj = nx - bi;

        res = process_block(bi, bj, pb);

        if (res > dm[nx])
          dm[nx] = res;
      }
    }

    for (int64_t nx = NB - 2; nx >= 0; nx--) {
      int64_t bi, bj;
      double res;

#pragma omp parallel for shared(nx, pb, dm) private(bi, bj, res)
      for (bi = NB - nx - 1; bi < NB; bi++) {
        bj = NB + ((NB - 2) - nx) - bi;

        res = process_block(bi, bj, pb);

        if (res > dm[nx])
          dm[nx] = res;
      }
    }

    for (int64_t i = 0; i < NB; i++) {
      if (dm[i] > dmax)
        dmax = dm[i];
    }
    pb->iters++;

  } while (dmax > pb->eps);
  free(dm);

  return 0;
}

problem *approximate(double eps, int64_t sz, int64_t bs, func_R2 f, func_R2 u) {
  problem *pb = calloc(1, sizeof(problem));
  if (pb == NULL)
    return NULL;

  if (init_pb(eps, sz, bs, pb) != 0) {
    free(pb);
    return NULL;
  }

  init_matrixes(f, u, pb);

  if (process_blocks(pb) != 0) {
    free(pb);
    return NULL;
  }
  return pb;
}