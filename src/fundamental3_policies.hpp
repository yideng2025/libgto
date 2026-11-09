#pragma once
#include "internal.hpp"
#include "boys_table.hpp"
namespace cubic_boys {
using gto::Real;
static void wide(double t, double *F) {
  Real T(t), v[4];
  if (T == 0) {
    for (int m = 0; m < 4; m++)
      v[m] = Real(1) / (2 * m + 1);
  } else if (T < 32) {
    unsigned cell = unsigned(2 * T + Real(.5));
    Real delta = T - Real(cell) * Real(.5),
         x = gto::boys_table[cell][17] * gto::inverse_factorial[14];
    for (int k = 13; k >= 0; k--)
      x = gto::boys_table[cell][3 + k] * gto::inverse_factorial[k] - delta * x;
    v[3] = x;
    Real e = std::exp(-T);
    for (int m = 3; m > 0; m--)
      v[m - 1] = (2 * T * v[m] + e) / (2 * m - 1);
  } else {
    Real root = std::sqrt(T), e = std::exp(-T);
    v[0] = std::sqrt(gto::pi) * std::erf(root) / (2 * root);
    for (int m = 0; m < 3; m++)
      v[m + 1] = ((2 * m + 1) * v[m] - e) / (2 * T);
  }
  for (int m = 0; m < 4; m++)
    F[m] = double(v[m]);
}
struct BoysTable {
  double f[65][18]{};
  constexpr BoysTable() {
    for (int i = 0; i < 65; i++)
      for (int j = 0; j < 18; j++)
        f[i][j] = double(gto::boys_table[i][j]);
  }
};
static constexpr BoysTable boys_table64;
// Bounded high-T compensated FP64: explicit FMA residuals, no fast-math.
// Compensated residuals retain accuracy across the large-T recurrence.
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
  for (int m = 0; m < 3; m++) {
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
    for (int m = 0; m < 4; m++)
      f[m] = 1. / (2 * m + 1);
    return;
  }
  unsigned cell = unsigned(2 * t + .5);
  double d = t - .5 * cell;
  double v = boys_table64.f[cell][17] * double(gto::inverse_factorial[14]);
  for (int k = 13; k >= 0; k--)
    v = boys_table64.f[cell][3 + k] * double(gto::inverse_factorial[k]) - d * v;
  f[3] = v;
  double e = std::exp(-t);
  for (int m = 3; m > 0; m--)
    f[m - 1] = (2 * t * f[m] + e) / (2 * m - 1);
}
static void evaluate(int policy, double t, double *f) {
  if (policy && t >= 0 && (t < 32 || (policy >= 2 && t < 1e8)))
    boys64(t, f, policy);
  else
    wide(t, f);
}
}
