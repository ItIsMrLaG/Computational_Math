#include <stdio.h>
#include <time.h>

#include "csv_save.h"
#include "functions.h"
#include "method.h"

int main(void) {
  int64_t fc = 0;
  int64_t fc_ = 1;
  func_R2 test_f[] = {f_linear};
  func_R2 test_d[] = {d_linear};
  char* name[] = {n_linear};

  int64_t tc = 0;
  int64_t tc_ = 1;
  int threads[] = {8};

  int64_t ec = 0;
  int64_t ec_ = 3;
  double eps[] = {0.1, 0.01, 0.001};

  int64_t Nc = 0;
  int64_t Nc_ = 5;
  int N[] = {10, 50, 100, 500, 1000};

  int64_t bsc = 0;
  int64_t bsc_ = 1;
  int bs[] = {64};


  for(fc = 0; fc<fc_; fc++){
    for(tc = 0; tc<tc_; tc++){
      for(ec = 0; ec<3; ec++){
         for(Nc = 0; Nc<Nc_; Nc++){
            for(bsc = 0; bsc<bsc_; bsc++){
              for(int k = 0; k < 10; k++){
                omp_set_num_threads(threads[tc]);

                double start_t = omp_get_wtime();
                problem *pb = approximate(eps[ec], N[Nc], bs[bsc], test_d[fc], test_f[fc]);
                double end_t = omp_get_wtime();
                
                int64_t flg = fc + 10*tc + 100*ec + 1000*Nc + 10000*bsc + 10000000*k;
                save_csv(pb, test_f[fc], end_t - start_t, threads[tc], flg, name[fc]);

              printf("%d) %ld\n",k , flg);
              }
            }
         }
      }
    }
  }
  return 0;
}