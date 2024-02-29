#ifndef VCHI
#define VCHI

#include <stdint.h>
#include <stdlib.h>

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define u_ij(i, j, pb)                                                         \
  0.25 * (pb->u[i - 1][j] + pb->u[i + 1][j] + pb->u[i][j - 1] +                \
          pb->u[i][j + 1] - pb->h * pb->h * pb->f[i][j])

enum signals { SUCCESS, CALCULATION_ERR };

typedef double (*func)(double x, double y);

typedef struct point {
  double x;
  double y;
} point;

typedef struct problem_params {
  /** Point number */
  size_t size;

  /** Size of the grid  */
  double grid_size;

  /** Measurement error */
  double eps;

  /** h := grid_size/(N + 1) */
  double h;

  /** Edge point */
  point shift;

  /** u-function matrix */
  double **u;

  /** function */
  double **f;

} pb_parms;

pb_parms *approximate_values(uint32_t N, double grid_size, double eps,
                             point l_down, func f, func d2f);

void free_results(pb_parms *);

#endif