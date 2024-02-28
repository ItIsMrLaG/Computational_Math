#ifndef VCHI
#define VCHI

#include <stdint.h>
#include <stdlib.h>


#define u_ij(u, i, j, f, h)                                                    \
  = 0.25 *                                                                     \
    (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j])

typedef double (*func)(double x, double y);

typedef struct point {
  double x;
  double y;
} point;

typedef struct problem_params {
  /** Point number */
  size_t N;

  /** Size of the grid  */
  double grid_size;

  /** Measurement error */
  double eps;

  /** h := grid_size/(N + 1) */
  double h;

  /** Edge points */
  point l_down;
  point r_up;

  /** u-function matrix */
  double **u;

  /** Init function */
  func f;
} pb_parms;

pb_parms *approximate_values(uint32_t N, double grid_size, double eps,
                             point l_down, point r_up, func f);

void free_results(pb_parms*);

#endif