// Private cross-electron p-p joint Gaussian moment panel; numerical arrays only.
#include "internal.hpp"
#include "boys_table.hpp"
namespace two_p_moment {
using gto::Real;
struct Table {
  double f[65][18]{};
  constexpr Table() {
    for (int i = 0; i < 65; i++)
      for (int j = 0; j < 18; j++)
        f[i][j] = double(gto::boys_table[i][j]);
  }
};
static constexpr Table table;
static void boys(double T, double *F) {
  if (T == 0) {
    for (int m = 0; m < 3; m++)
      F[m] = 1. / (2 * m + 1);
    return;
  }
  if (T < 32) {
    unsigned cell = unsigned(2 * T + .5);
    double d = T - .5 * cell, v = table.f[cell][16] * double(gto::inverse_factorial[14]);
    for (int k = 13; k >= 0; k--)
      v = table.f[cell][2 + k] * double(gto::inverse_factorial[k]) - d * v;
    F[2] = v;
    double e = std::exp(-T);
    for (int m = 2; m > 0; m--)
      F[m - 1] = (2 * T * F[m] + e) / (2 * m - 1);
    return;
  }
  double root = std::sqrt(T);
  F[0] = double(std::sqrt(gto::pi)) / (2 * root);
  if (T >= 48) {
    for (int m = 0; m < 2; m++)
      F[m + 1] = (2 * m + 1) * F[m] / (2 * T);
  } else {
    F[0] *= std::erf(root);
    double e = std::exp(-T);
    for (int m = 0; m < 2; m++)
      F[m + 1] = ((2 * m + 1) * F[m] - e) / (2 * T);
  }
}
struct Packet {
  double X[3][64], a0[3][64], a1[3][64], c0[3][64], c1[3][64], G[9][64], T[64], rho[64], pref[64],
      budget[64], na[64], nb[64], F[3][64];
  bool valid[64];
};
// Same Boys approximation and same original coefficient-weighted absolute budget.
// Envelope is empirical, not a rigorous error certificate. Reject to original MD.
static void angular(Packet &p, size_t n) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (size_t k = 0; k < n; k++) {
    double f[3]{};
    if (p.valid[k])
      boys(p.T[k], f);
    for (int m = 0; m < 3; m++)
      p.F[m][k] = f[m];
  }
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
      for (size_t k = 0; k < n; k++) {
        double v = (-2 * p.rho[k] * p.a1[i][k]) * p.X[i][k],
               z = (2 * p.rho[k] * p.c1[j][k]) * p.X[j][k];
        double f1 = p.a0[i][k] * z + v * p.c0[j][k];
        if (i == j)
          f1 += (2 * p.rho[k] * p.a1[i][k]) * p.c1[j][k];
        p.G[i + 3 * j][k] = p.pref[k] * ((p.a0[i][k] * p.c0[j][k]) * p.F[0][k] + f1 * p.F[1][k] +
                                         (v * z) * p.F[2][k]);
      }
  for (size_t k = 0; k < n; k++) {
    double ma = 0, mc = 0, da = 0, dc = 0;
    for (int i = 0; i < 3; i++) {
      ma = std::max(ma, std::abs(p.a0[i][k]) + std::abs(2 * p.rho[k] * p.a1[i][k] * p.X[i][k]));
      mc = std::max(mc, std::abs(p.c0[i][k]) + std::abs(2 * p.rho[k] * p.c1[i][k] * p.X[i][k]));
      da = std::max(da, std::abs(p.a1[i][k]));
      dc = std::max(dc, std::abs(p.c1[i][k]));
    }
    double env = ma * mc + (2 * p.rho[k] * da) * dc;
    double bound =
        std::abs(p.pref[k]) * p.F[0][k] * env * (2048 * eps * (1 + std::min(p.T[k], 2.5)));
    bound = bound * (1 + 64 * eps) + std::numeric_limits<double>::min();
    p.valid[k] &= std::isfinite(bound) && bound <= p.budget[k];
    for (int d = 0; d < 9; d++)
      if (!std::isfinite(p.G[d][k]) || (p.G[d][k] != 0 && !std::isnormal(p.G[d][k])))
        p.valid[k] = false;
  }
}
}
