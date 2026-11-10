#pragma once
#include "internal.hpp"
namespace quadratic_pair {
constexpr double rt = 1.7320508075688772935274463415058723669;
static void forms(const double *a, const double *v, double q[5][3], double qa[5][3],
                  double qv[5][3]) {
  for (int m = 0; m < 5; m++) {
    if (m == 0) {
      q[m][0] = rt * a[0] * a[1];
      q[m][1] = rt * (a[0] * v[1] + a[1] * v[0]);
      q[m][2] = rt * v[0] * v[1];
    }
    if (m == 1) {
      q[m][0] = rt * a[1] * a[2];
      q[m][1] = rt * (a[1] * v[2] + a[2] * v[1]);
      q[m][2] = rt * v[1] * v[2];
    }
    if (m == 2) {
      q[m][0] = a[2] * a[2] - .5 * (a[0] * a[0] + a[1] * a[1]);
      q[m][1] = 2 * a[2] * v[2] - (a[0] * v[0] + a[1] * v[1]);
      q[m][2] = v[2] * v[2] - .5 * (v[0] * v[0] + v[1] * v[1]);
    }
    if (m == 3) {
      q[m][0] = rt * a[0] * a[2];
      q[m][1] = rt * (a[0] * v[2] + a[2] * v[0]);
      q[m][2] = rt * v[0] * v[2];
    }
    if (m == 4) {
      q[m][0] = (.5 * rt) * (a[0] * a[0] - a[1] * a[1]);
      q[m][1] = rt * (a[0] * v[0] - a[1] * v[1]);
      q[m][2] = (.5 * rt) * (v[0] * v[0] - v[1] * v[1]);
    }
  }
  for (int which = 0; which < 2; which++) {
    const double *x = which ? v : a;
    double(*t)[3] = which ? qv : qa;
    t[0][0] = (.5 * rt) * x[1];
    t[0][1] = (.5 * rt) * x[0];
    t[0][2] = 0;
    t[1][0] = 0;
    t[1][1] = (.5 * rt) * x[2];
    t[1][2] = (.5 * rt) * x[1];
    t[2][0] = -.5 * x[0];
    t[2][1] = -.5 * x[1];
    t[2][2] = x[2];
    t[3][0] = (.5 * rt) * x[2];
    t[3][1] = 0;
    t[3][2] = (.5 * rt) * x[0];
    t[4][0] = (.5 * rt) * x[0];
    t[4][1] = (-.5 * rt) * x[1];
    t[4][2] = 0;
  }
}
}
