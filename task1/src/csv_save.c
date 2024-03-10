#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "csv_save.h"
#include "method.h"

void save_result(problem *pb, int index) {
  char name[100];
  snprintf(name, 100, "%s/res_%d.csv", RESULT_DIR_NAME, index);

  FILE *fl = fopen(name, "w");
  for (int64_t i = 0; i < pb->size; i++) {
    for (int64_t j = 0; j < pb->size - 1; j++) {
      fprintf(fl, "%f,", pb->u[i][j]);
    }
    fprintf(fl, "%f\n", pb->u[i][pb->size - 1]);
  }
  fclose(fl);
}

void save_answer(problem *pb, func_R2 u, int index) {
  char name[100];
  snprintf(name, 100, "%s/ans_%d.csv", RESULT_DIR_NAME, index);

  FILE *ans = fopen(name, "w");

  for (int64_t i = 0; i < pb->size; i++) {
    for (int64_t j = 0; j < pb->size - 1; j++) {
      fprintf(ans, "%f,", u(x_i(i, pb), y_j(j, pb)));
    }
    fprintf(ans, "%f\n", u(x_i(i, pb), y_j((pb->size - 1), pb)));
  }
  fclose(ans);
}

void save_meta(problem *pb, double t, int th_n, int index, char *info) {
  char name[100];
  snprintf(name, 100, "%s/met_%d.json", RESULT_DIR_NAME, index);

  FILE *cfg = fopen(name, "w");

  fprintf(cfg, "{");

  fprintf(cfg, "\"h\":%f,", pb->h);
  fprintf(cfg, "\"eps\":%f,", pb->eps);
  fprintf(cfg, "\"time\":%f,", t);
  fprintf(cfg, "\"N\":%ld,", pb->size);
  fprintf(cfg, "\"iters\":%ld,", pb->iters);
  fprintf(cfg, "\"thr_n\":%d,", th_n);
  fprintf(cfg, "\"x_y\":[%f, %f],", pb->x0, pb->y0);
  fprintf(cfg, "\"max_init\":%ld,", pb->max_init);
  fprintf(cfg, "\"side_l\":%f,", pb->l);
  fprintf(cfg, "\"spec_info\":\"%s\",", info);
  fprintf(cfg, "\"bs\":%ld", pb->bs);

  fprintf(cfg, "}");
  fclose(cfg);
}

void save_csv(problem *pb, func_R2 u, double t, int th_n, int index,
              char *info) {
  save_result(pb, index);
  save_answer(pb, u, index);
  save_meta(pb, t, th_n, index, info);
}