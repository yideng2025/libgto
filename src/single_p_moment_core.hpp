#include "internal.hpp"
#include "boys_table.hpp"
#include <cstdint>
namespace single_p {
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
static void boys(double T, double &f0, double &f1) {
  if (T == 0) {
    f0 = 1;
    f1 = 1. / 3;
    return;
  }
  if (T < 32) {
    unsigned cell = unsigned(2 * T + .5);
    double d = T - .5 * cell, v = table.f[cell][15] * double(gto::inverse_factorial[14]);
    for (int k = 13; k >= 0; k--)
      v = table.f[cell][1 + k] * double(gto::inverse_factorial[k]) - d * v;
    f1 = v;
    f0 = 2 * T * f1 + std::exp(-T);
    return;
  }
  double root = std::sqrt(T);
  f0 = double(std::sqrt(gto::pi)) / (2 * root);
  if (T >= 48)
    f1 = f0 / (2 * T);
  else {
    f0 *= std::erf(root);
    f1 = (f0 - std::exp(-T)) / (2 * T);
  }
}

struct Packet {
  double X[3][64], E0[3][64], E1[3][64], M[3][64], G[3][64], T[64], rho[64], pref[64], F0[64],
      F1[64], M0[64], err[64], maxm[64], na[64], nb[64], budget[64];
  Real weight[64]{};
  const libgto_eri_task *task[64]{};
  bool ket[64]{}, valid[64]{};
};

static void angular_moment(Packet &p, size_t n) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (size_t k = 0; k < n; k++) {
    if (p.valid[k])
      boys(p.T[k], p.F0[k], p.F1[k]);
    else
      p.F0[k] = p.F1[k] = 0;
  }
  for (int d = 0; d < 3; d++)
    for (size_t k = 0; k < n; k++) {
      double v = ((p.ket[k] ? 2. : -2.) * p.rho[k] * p.E1[d][k]) * p.X[d][k];
      p.G[d][k] = p.pref[k] * (p.E0[d][k] * p.F0[k] + v * p.F1[k]);
    }
  for (size_t k = 0; k < n; k++) {
    double h = 0;
    for (int d = 0; d < 3; d++) {
      double v = ((p.ket[k] ? 2. : -2.) * p.rho[k] * p.E1[d][k]) * p.X[d][k];
      h = std::max(h, std::abs(p.E0[d][k]) + std::abs(v));
    }
    double bound = std::abs(p.pref[k]) * p.F0[k] * h * (2048 * eps * (1 + std::min(p.T[k], 1.5)));
    bound = bound * (1 + 64 * eps) + std::numeric_limits<double>::min();
    p.valid[k] &= std::isfinite(bound) && bound <= p.budget[k];
    for (int d = 0; d < 3; d++)
      if (!std::isfinite(p.G[d][k]) || (p.G[d][k] != 0 && !std::isnormal(p.G[d][k])))
        p.valid[k] = false;
  }
}

static void angular(Packet &p, size_t n) {
  angular_moment(p, n);
}

}
