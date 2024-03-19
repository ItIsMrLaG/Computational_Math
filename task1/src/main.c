#include <stdio.h>
#include <time.h>

#include "csv_save.h"
#include "functions.h"
#include "method.h"

#define CONST_ZERu "D\\D^0 CONST"
#define RND_CONSTu " D\\D^0 random from [0; const]"
#define AVERu "D\\D^0 average(u(D^0))"

#define MAXu "D\\D^0 max(u(D^0))"
#define RND_AVERu "D\\D^0 random from [0; average(u(D^0))]"

int main(void) {
  int64_t fc = 0;
  int64_t fc_ = 4;
  func_R2 test_f[] = {f_kx3_p_2ky3, f_x3_p_y3, f_linear, f_linearH};
  func_R2 test_d[] = {d_kx3_p_2ky3, d_x3_p_y3, d_linear, d_linearH};
  char *name[] = {n_kx3_p_2ky3, n_x3_p_y3, n_linear, n_linearH};

  int64_t tc = 0;
  int64_t tc_ = 1;
  int threads[] = {12};

  int64_t ec = 0;
  int64_t ec_ = 1;
  double eps[] = {0.1};

  int64_t Nc = 0;
  int64_t Nc_ = 1;
  int N[] = {500};

  int64_t bsc = 0;
  int64_t bsc_ = 1;
  int bs[] = {64};

  for (fc = 0; fc < fc_; fc++) {
    for (tc = 0; tc < tc_; tc++) {
      for (ec = 0; ec < ec_; ec++) {
        for (Nc = 0; Nc < Nc_; Nc++) {
          for (bsc = 0; bsc < bsc_; bsc++) {
            for (int k = 0; k < 20; k++) {
              omp_set_num_threads(threads[tc]);

              double start_t = omp_get_wtime();
              problem *pb =
                  approximate(eps[ec], N[Nc], bs[bsc], test_d[fc], test_f[fc]);
              double end_t = omp_get_wtime();

              int64_t flg = fc + 10 * tc + 100 * ec + 1000 * Nc + 10000 * bsc +
                            10000000 * k;
              save_csv(pb, test_f[fc], end_t - start_t, threads[tc], flg,
                       name[fc]);

              printf("%d) %ld\n", k, flg);
            }
          }
        }
      }
    }
  }
  return 0;
}