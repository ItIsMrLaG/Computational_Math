#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "method.h"

double f_kx3_p_2ky3(double x, double y) {
  return 1000 * pow(x, 3) + 2000 * pow(y, 3);
}
double d_kx3_p_2ky3(double x, double y) { return 6000 * x + 12000 * y; }

double f_sin(double x, double y) { return sin(x) + 0 * y; }
double d_sin(double x, double y) { return f_sin(x, y); }

double f_sin_xy(double x, double y) { return x * sin(x) + cos(y) / y; }
double d_f_sin_xy(double x, double y) {
  double hm = pow(y, 3);
  if (hm == 0) {
    return 2 * cos(x) - x * sin(x) +
           (2 * y * sin(y) - (y * y - 2) * cos(y)) / 1e-15;
  }
  return 2 * cos(x) - x * sin(x) + (2 * y * sin(y) - (y * y - 2) * cos(y)) / hm;
}

double f_linear(double x, double y) { return x + y; }
double d_linear(double x, double y) { return 0 * x + 0 * y; }

double f_sin2(double x, double y) { return sin(x) + sin(y); }
double d_sin2(double x, double y) { return f_sin2(x, y); }
