#ifndef FUNCTIONS
#define FUNCTIONS

#include <stdint.h>
#include <stdlib.h>

/* f(x, y) := 1000*x^3 + 2000*y^3 */
static const char *n_kx3_p_2ky3 = "1000*x^3+2000*y^3";
double f_kx3_p_2ky3(double x, double y);
double d_kx3_p_2ky3(double x, double y);

/* f(x, y) := sin(x) */
static const char *n_sin = "sin(x)";
double f_sin(double x, double y);
double d_sin(double x, double y);

/* f(x, y) := x*sin(x) + cos(y)/y */
static const char *n_sin_xy = "x*sin(x)+cos(y)/y";
double f_sin_xy(double x, double y);
double d_sin_xy(double x, double y);

/* f(x, y) := x + y */
static const char *n_linear = "x+y";
double f_linear(double x, double y);
double d_linear(double x, double y);

/* f(x, y) := sin(x) + sin(y) */
static const char *n_sin2 = "sin(x)+sin(y)";
double f_sin2(double x, double y);
double d_sin2(double x, double y);

#endif