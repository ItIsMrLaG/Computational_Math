#include <stdio.h>
#include <time.h>

#include "csv_save.h"
#include "functions.h"
#include "method.h"

int main(void) {
  int64_t fc = 0;
  func_R2 test_f[] = {f_kx3_p_2ky3, f_sin, f_linear};
  func_R2 test_d[] = {d_kx3_p_2ky3, d_sin, d_linear};
  char* name[] = {n_kx3_p_2ky3, n_sin, n_linear};

  int64_t tc = 0;
  int threads[] = {1, 4, 8, 12};

  int64_t ec = 0;
  double eps[] = {0.1, 0.01, 0.001};

  int64_t Nc = 0;
  int N[] = {10, 50, 100, 500, 1000};

  int64_t bsc = 0;
  int bs[] = {4, 16, 64, 128};


  for(fc = 0; fc<3; fc++){
    for(tc = 0; tc<4; tc++){
      for(ec = 0; ec<3; ec++){
         for(Nc = 0; Nc<5; Nc++){
            for(bsc = 0; bsc<4; bsc++){
              omp_set_num_threads(threads[tc]);

              double start_t = omp_get_wtime();
              problem *pb = approximate(eps[ec], N[Nc], bs[bsc], test_d[fc], test_f[fc]);
              double end_t = omp_get_wtime();
              
              int64_t flg = fc + 10*tc + 100*ec + 1000*Nc + 10000*bsc;
              save_csv(pb, test_f[fc], end_t - start_t, tc, flg, name[fc]);

              printf("%ld\n", flg);
            }
         }
      }
    }
  }
  return 0;
}