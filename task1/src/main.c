#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "main.h"

#define BLOCK_SIZE 64

#define x_ij(i, j, pb) (i * pb->h + pb->shift.x)
#define y_ij(i, j, pb) (i * pb->h + pb->shift.y)

#define b_k0(bk, bl_size) (bk * bl_size + 1)
#define b_kn(k0, bl_size, pb) min(k0 + bl_size, pb->size - 1)

// #define DEBUG

static double **my_alloc_u(int64_t N) {
  double **u = calloc(N, sizeof(double *));

  if (u == NULL)
    return NULL;

  for (int64_t i = 0; i < N; i++) {
    u[i] = calloc(N, sizeof(double));

    if (u[i] == NULL) {

      for (int64_t j = 0; j < i; j++)
        free(u[i]);

      free(u);
      return NULL;
    }
  }
  return u;
}

static inline void my_free_u(int64_t N, double **u) {
  for (int64_t i = 0; i < N; i++)
    free(u[i]);
  free(u);
}

static inline void init_matrix(pb_parms *pb, func f, func u) {

  int64_t i, j;

  for (i = 0; i < pb->size; i++) {
    for (j = 0; j < pb->size; j++) {
      if ((i == 0) || (j == 0) || (i == pb->size) || (j == pb->size)) {
        pb->u[i][j] = u(x_ij(i, j, pb), y_ij(i, j, pb));
      } else {
        pb->u[i][j] = 0;
      }

      pb->f[i][j] = f(x_ij(i, j, pb), y_ij(i, j, pb));
    }
  }

#ifdef DEBUG
  for (size_t i = 0; i <= pb->size; i++) {
    printf("\n %lu: ", i);
    for (size_t j = 0; j <= pb->size; j++) {
      printf("|(%lu) %f ", j, pb->u[i][j]);
    }
  }
#endif
}

static inline pb_parms *create_pb(int64_t N, double grid_size, double eps,
                                  point l_down) {
  pb_parms *pb = calloc(1, sizeof(pb_parms));

  if (pb == NULL)
    return pb;

  pb->size = N + 2;
  pb->grid_size = grid_size;
  pb->eps = eps;
  pb->h = grid_size / pb->size;
  pb->shift = l_down;

  pb->u = my_alloc_u(pb->size);
  pb->f = my_alloc_u(pb->size);

  if (pb->u == NULL || pb->f == NULL)
    return NULL;
#ifdef DEBUG
  for (size_t i = 0; i <= pb->size; i++) {
    printf("\n %lu: ", i);
    for (size_t j = 0; j <= pb->size; j++) {
      printf("|(%lu) %f ", j, pb->u[i][j]);
    }
  }
#endif
  return pb;
}

static double process_block(pb_parms *pb, int64_t bx, int64_t by,
                            int64_t bl_size) {
  int64_t i0 = b_k0(bx, bl_size);
  int64_t in = b_kn(i0, bl_size, pb);
  int64_t j0 = b_k0(by, bl_size);
  int64_t jn = b_kn(j0, bl_size, pb);
  double dm = 0;

  for (int64_t i = i0; i < in; i++) {
    for (int64_t j = j0; j < jn; j++) {
      double temp = pb->u[i][j];

      pb->u[i][j] = u_ij(i, j, pb);

      double d = fabs(temp - pb->u[i][j]);
      if (dm < d)
        dm = d;
    }
  }

  return dm;
}

void free_pb(pb_parms *pb) {
  if (pb == NULL) {
    return;
  }

  //	TODO:
  //  my_free_u(pb->size, pb->u);
  //  my_free_u(pb->size, pb->f);
  //  free(pb);
}

int64_t process_blocks(pb_parms *pb) {
  double dmax;
  int64_t iter = 0;
  int64_t bl_size = BLOCK_SIZE;
  int64_t N, NB;
  N = pb->size - 2;
  NB = N / BLOCK_SIZE;

  if (N < BLOCK_SIZE)
    return -MEM_ERR;
  if (NB * BLOCK_SIZE < N)
    NB += 1;

  double *dm = calloc(NB, sizeof(double));
  // double dm[3] = {0};

  double **un = my_alloc_u(N);
  if (dm == NULL || un == NULL)
    return -MEM_ERR;

  do {
    int64_t bj, bi;
    int64_t nx;
    double res;
    iter++;
    dmax = 0;

    for (nx = 0; nx < NB; nx++) {
      dm[nx] = 0;

// #pragma omp parallel for shared(pb, nx, dm) private(bi, bj, res)
      for (bi = 0; bi < nx + 1; bi++) {

        bj = nx - bi;

        res = process_block(pb, bi, bj, bl_size);

        if (dm[bi] < res)
          dm[bi] = res;
      }
    }

    for (nx = NB - 2; nx >= 0; nx--) {

// #pragma omp parallel for shared(pb, nx, dm) private(bi, bj, res)
      for (bi = 0; bi < nx + 1; bi++) {
        bj = 2 * (NB - 1) - nx - bi;
        res = process_block(pb, bi, bj, bl_size);
        if (dm[bi] < res)
          dm[bi] = res;
      }
    }

    for (int64_t k = 0; k < NB; k++) {
      if (dmax < dm[k])
        dmax = dm[k];
    }
    printf("%f\n", dmax);
  } while (dmax > pb->eps);
  free(dm);
  // free(um???)

  return iter;
}

pb_parms *approximate_values(int64_t N, double grid_size, double eps,
                             point l_down, func f, func u) {
  pb_parms *pb = create_pb(N, grid_size, eps, l_down);
  int64_t iter = 0;

  if (pb == NULL)
    return NULL;

  init_matrix(pb, f, u);
  iter = process_blocks(pb);
  if (iter < SUCCESS) {
    free_pb(pb);
    return NULL;
  }
  printf("res: %lu !", iter);
  return pb;
}

// double d_x3_p_y3(double x, double y) { return 6 * x + 6 * y; }

// double x3_p_y3(double x, double y) { return pow(x, 3) + pow(y, 3); }

double d_kx3_p_2ky3(double x, double y) { return 6000 * x + 12000 * y; }

double kx3_p_2ky3(double x, double y) {
  return 1000 * pow(x, 3) + 2000 * pow(y, 3);
}

int main(void) {

  omp_set_num_threads(8);
  func f = d_kx3_p_2ky3; // d_book;q
  func u = kx3_p_2ky3;   // book;
  point p = {0};

  // int64_t sz[] = {100, 200, 300, 500, 1000, 2000, 3000};
  pb_parms *res = approximate_values(1000, 1, 0.1, p, f, u);
  free_pb(res);
  if (res)
    printf("good");
  return 0;
}
