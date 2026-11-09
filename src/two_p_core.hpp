// Private autonomous one-primitive/channel p-s | s-p family. C++17, scalar fallback.
#include "internal.hpp"
#include "boys_table.hpp"
namespace two_p {
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
  double X[3][64], a0[3][64], a1[3][64], c0[3][64], c1[3][64], R[15][64], ER[15][64], S[10][64],
      U[12][64], G[9][64], T[64], rho[64], pref[64], budget[64], na[64], nb[64], maxm[64], err[64],
      F[3][64];
  Real weight[64];
  const libgto_eri_task *task[64];
  bool valid[64];
};
struct Step {
  int dst, first, second, axis, coefficient;
};
static constexpr Step steps[] = {{11, 14, 14, 2, 0}, {12, 14, 14, 1, 0}, {13, 14, 14, 0, 0},
                                 {1, 10, 10, 2, 0},  {2, 10, 10, 1, 0},  {3, 10, 10, 0, 0},
                                 {4, 11, 10, 2, 1},  {5, 11, 11, 1, 0},  {6, 12, 10, 1, 1},
                                 {7, 11, 11, 0, 0},  {8, 12, 12, 0, 0},  {9, 13, 10, 0, 1}};
static constexpr int map[4][4] = {{0, 1, 2, 3}, {1, 4, 5, 7}, {2, 5, 6, 8}, {3, 7, 8, 9}};
static void angular(Packet &p, size_t n) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (size_t k = 0; k < n; k++) {
    double F[3]{};
    if (p.valid[k])
      boys(p.T[k], F);
    for (int m = 0; m < 3; m++)
      p.F[m][k] = F[m];
    double factor = 1;
    for (int m = 0; m < 3; m++) {
      int dst = m == 0 ? 0 : m == 1 ? 10 : 14;
      p.R[dst][k] = factor * F[m];
      p.ER[dst][k] = std::abs(p.R[dst][k]) * eps * (128 * (1 + std::min(p.T[k], m + .5)) + 16 * m);
      factor *= -2 * p.rho[k];
    }
    p.maxm[k] = p.err[k] = 0;
  }
  for (auto s : steps)
    for (size_t k = 0; k < n; k++) {
      double v = p.X[s.axis][k] * p.R[s.first][k],
             e = std::abs(p.X[s.axis][k]) * p.ER[s.first][k] + 3 * eps * std::abs(v);
      if (s.coefficient) {
        double u = s.coefficient * p.R[s.second][k];
        v += u;
        e += s.coefficient * p.ER[s.second][k] + 2 * eps * (std::abs(u) + std::abs(v));
      }
      p.R[s.dst][k] = v;
      p.ER[s.dst][k] = e;
    }
  for (int h = 0; h < 10; h++)
    for (size_t k = 0; k < n; k++) {
      double value = p.pref[k] * p.R[h][k];
      p.S[h][k] = value;
      if (value != 0 && !std::isnormal(value))
        p.valid[k] = false;
      p.maxm[k] = std::max(p.maxm[k], std::abs(value));
      p.err[k] = std::max(p.err[k], std::abs(p.pref[k]) * p.ER[h][k] + 48 * eps * std::abs(value));
    }
  for (size_t k = 0; k < n; k++) {
    double bound = (p.err[k] + 24 * eps * p.maxm[k]) * p.na[k] * p.nb[k];
    bound = bound * (1 + 32 * eps) + std::numeric_limits<double>::min();
    p.valid[k] &= std::isfinite(bound) && bound <= p.budget[k];
  }
  for (int h = 0; h < 4; h++)
    for (int j = 0; j < 3; j++)
      for (size_t k = 0; k < n; k++)
        p.U[h * 3 + j][k] = p.S[h][k] * p.c0[j][k] + (-p.S[map[h][3 - j]][k]) * p.c1[j][k];
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
      for (size_t k = 0; k < n; k++)
        p.G[i + 3 * j][k] = p.a0[i][k] * p.U[j][k] + p.a1[i][k] * p.U[(3 - i) * 3 + j][k];
  for (size_t k = 0; k < n; k++)
    for (int d = 0; d < 9; d++)
      if (!std::isfinite(p.G[d][k]))
        p.valid[k] = false;
}
}
