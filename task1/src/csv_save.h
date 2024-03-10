#ifndef CSV_SAVE
#define CSV_SAVE

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "method.h"

#define RESULT_DIR_NAME "experiments"

void save_result(problem *pb, int index);

void save_answer(problem *pb, func_R2 u, int index);

void save_meta(problem *pb, double t, int th_n, int index, char *info);

void save_csv(problem *pb, func_R2 u, double t, int th_n, int index,
              char *info);

#endif