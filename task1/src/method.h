#ifndef VCHI
#define VCHI
#include <stdint.h>
#include <stdlib.h>

#define x_i(i, pb) (pb->h * i)
#define y_j(j, pb) (pb->h * j)
#define u_ij(i, j, pb)                                                         \
  (0.25 * (pb->u[i - 1][j] + pb->u[i + 1][j] + pb->u[i][j - 1] +               \
           pb->u[i][j + 1] - pb->h * pb->h * pb->f[i][j]))

#define CEIL_DIV_UP(x, y) ((x + y - 1) / y)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

typedef double (*func_R2)(double, double);

typedef struct {
  double **u;
  double **f;
  double h;
  int64_t iters;

  double eps;
  double x0;
  double y0;
  double l;
  int64_t bs;
  int64_t size;
  int64_t max_init;
} problem;

problem *approximate(double eps, int64_t sz, int64_t bs, func_R2 f, func_R2 u);

void free_pb(problem *pb);
#endif