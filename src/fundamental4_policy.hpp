#pragma once
#include "internal.hpp"
#include "boys_table.hpp"
namespace quartic_boys {
using gto::Real;

struct BoysTable {
  double f[65][19]{};
  constexpr BoysTable() {
    for (int i = 0; i < 65; i++)
      for (int j = 0; j < 19; j++)
        f[i][j] = double(gto::boys_table[i][j]);
  }
};
static constexpr BoysTable boys_table64;
// Bounded high-T compensated FP64: explicit FMA residuals, no fast-math.
// Compensation retains accuracy across the large-T recurrence.
static void high_boys(double t, double *f, bool omit_exp) {
  constexpr gto::Real cw = 0.8862269254527580136490837416705725913987747L;
  constexpr double ch = double(cw), cl = double(cw - gto::Real(ch));
  double r = std::sqrt(t), relative = std::fma(-r, r, t) / (2 * t), q = ch / r;
  double lo = (std::fma(-q, r, ch) + cl) / r - q * relative;
  double hi = q + lo;
  lo = (q - hi) + lo;
  if (t < 64) {
    double e = std::erfc(r), s = std::fma(-hi, e, hi), prod = hi * e;
    double residual = ((hi - s) - prod) - std::fma(hi, e, -prod) + lo * (1 - e);
    hi = s + residual;
    lo = (s - hi) + residual;
  }
  f[0] = hi;
  double e = omit_exp && t >= 64 ? 0 : std::exp(-t), den = 2 * t;
  for (int m = 0; m < 4; m++) {
    double a = 2 * m + 1, u = a * hi, mulerr = std::fma(a, hi, -u) + a * lo, s = u - e,
           sl = ((u - s) - e) + mulerr;
    double v = s / den, vl = (std::fma(-v, den, s) + sl) / den;
    hi = v + vl;
    lo = (v - hi) + vl;
    f[m + 1] = hi;
  }
}
static void boys64(double t, double *f, int policy) {
  if (t >= 32) {
    high_boys(t, f, policy >= 3);
    return;
  }
  if (t == 0) {
    for (int m = 0; m < 5; m++)
      f[m] = 1. / (2 * m + 1);
    return;
  }
  unsigned cell = unsigned(2 * t + .5);
  double d = t - .5 * cell;
  double v = boys_table64.f[cell][18] * double(gto::inverse_factorial[14]);
  for (int k = 13; k >= 0; k--)
    v = boys_table64.f[cell][4 + k] * double(gto::inverse_factorial[k]) - d * v;
  f[4] = v;
  double e = std::exp(-t);
  for (int m = 4; m > 0; m--)
    f[m - 1] = (2 * t * f[m] + e) / (2 * m - 1);
}
static bool evaluate(double t, double *f) {
  if (!(t >= 0 && (t < 16 || (t >= 32 && t < 1e8))))
    return false;
  boys64(t, f, 2);
  return true;
}
}
