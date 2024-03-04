#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main.h"

#define BLOCK_SIZE 64

#define x_i(i, pb) (pb->h * i)
#define y_j(j, pb) (pb->h * j)
#define u_ij(i, j, pb)                                                         \
  (0.25 * (pb->u[i - 1][j] + pb->u[i + 1][j] + pb->u[i][j - 1] +               \
           pb->u[i][j + 1] - pb->h * pb->h * pb->f[i][j]))

#define CEIL_DIV_UP(x, y) ((x + y - 1) / y)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef double (*func_R2)(double, double);

typedef struct problem {
  double **u;
  double **f;
  double h;

  double eps;
  int64_t size;
  int64_t iters;
} problem;

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
  for (int64_t i = 0; i < pb->size; i++) {
    for (int64_t j = 0; j < pb->size; j++) {
      if ((i == 0) || (j == 0) || (i == pb->size - 1) || (j == pb->size - 1))
        pb->u[i][j] = u(x_i(i, pb), y_j(j, pb));
      else
        pb->u[i][j] = 0;

      pb->f[i][j] = f(x_i(i, pb), y_j(j, pb));
    }
  }
}

int init_pb(double eps, int64_t N, problem *pb) {

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

  i_0 = 1 + bi * BLOCK_SIZE;
  j_0 = 1 + bj * BLOCK_SIZE;

  i_n = MIN(i_0 + BLOCK_SIZE, pb->size - 1);
  j_n = MIN(j_0 + BLOCK_SIZE, pb->size - 1);

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
  int64_t NB = CEIL_DIV_UP(N, BLOCK_SIZE);

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

problem *approximate(double eps, int64_t sz, func_R2 f, func_R2 u) {
  problem *pb = calloc(1, sizeof(problem));
  if (pb == NULL)
    return NULL;

  if (init_pb(eps, sz, pb) != 0) {
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

/* =================================================== */

double d_kx3_p_2ky3(double x, double y) { return 6000 * x + 12000 * y; }

double kx3_p_2ky3(double x, double y) {
  return 1000 * pow(x, 3) + 2000 * pow(y, 3);
}

void check(problem *pb, func_R2 u) {
  printf("iters: %ld\n", pb->iters);

  for (size_t i = 1; i < pb->size - 1; i++) {
    for (size_t j = 1; j < pb->size - 1; j++) {
      printf("%7.2f ", fabs(pb->u[i][j] - u(x_i(i, pb), y_j(j, pb))));
    }
    printf("\n");
  }
}

void main(void) {
  omp_set_num_threads(8);
  problem *pb = approximate(0.1, 20, d_kx3_p_2ky3, kx3_p_2ky3);
  check(pb, kx3_p_2ky3);
}
