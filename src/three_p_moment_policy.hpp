// Bounded pp|ps/sp joint-Gaussian degree-three moment panel.
#include "internal.hpp"
#include "boys_table.hpp"
#include "fundamental3_policies.hpp"
namespace three_p_moment {
using gto::Real;

struct Packet {
  double a[3][64], b[3][64], c[3][64], v[3][64], z[3][64], H[9][64], J[9][64], A[3][64], B[3][64],
      G[27][64], F[4][64], T[64], pref[64], cov0[64], cov1[64], cross[64], budget[64], na[64],
      nb[64];
  bool valid[64];
};
static void angular(Packet &p, size_t n) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const int policy = 2;
  for (size_t k = 0; k < n; k++) {
    double f[4]{};
    if (p.valid[k])
      cubic_boys::evaluate(policy, p.T[k], f);
    for (int m = 0; m < 4; m++)
      p.F[m][k] = f[m];
  }
  for (int d = 0; d < 3; d++)
    for (size_t k = 0; k < n; k++) {
      p.A[d][k] = p.a[d][k] * p.F[1][k] + p.v[d][k] * p.F[2][k];
      p.B[d][k] = p.b[d][k] * p.F[1][k] + p.v[d][k] * p.F[2][k];
    }
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
      for (size_t k = 0; k < n; k++) {
        double q0 = p.a[i][k] * p.b[j][k], q1 = p.v[i][k] * p.b[j][k] + p.a[i][k] * p.v[j][k],
               q2 = p.v[i][k] * p.v[j][k];
        if (i == j) {
          q0 += p.cov0[k];
          q1 += p.cov1[k];
        }
        p.H[i + 3 * j][k] = (q0 * p.F[0][k] + q1 * p.F[1][k]) + q2 * p.F[2][k];
        p.J[i + 3 * j][k] = (q0 * p.F[1][k] + q1 * p.F[2][k]) + q2 * p.F[3][k];
      }
  for (int d = 0; d < 3; d++)
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++)
        for (size_t k = 0; k < n; k++) {
          double v = p.c[d][k] * p.H[i + 3 * j][k] + p.z[d][k] * p.J[i + 3 * j][k];
          if (i == d)
            v += p.cross[k] * p.B[j][k];
          if (j == d)
            v += p.cross[k] * p.A[i][k];
          p.G[i + 3 * j + 9 * d][k] = p.pref[k] * v;
        }
  for (size_t k = 0; k < n; k++) {
    double ma = 0, mb = 0, mc = 0;
    for (int d = 0; d < 3; d++) {
      ma = std::max(ma, std::abs(p.a[d][k]) + std::abs(p.v[d][k]));
      mb = std::max(mb, std::abs(p.b[d][k]) + std::abs(p.v[d][k]));
      mc = std::max(mc, std::abs(p.c[d][k]) + std::abs(p.z[d][k]));
    }
    double env = (ma * mb + std::abs(p.cov0[k]) + std::abs(p.cov1[k])) * mc +
                 std::abs(p.cross[k]) * (ma + mb);
    double bound = std::abs(p.pref[k]) * p.F[0][k] * env *
                   ((2048 + (policy && (p.T[k] < 32 || (policy >= 2 && p.T[k] < 1e8)) ? 256 : 0)) *
                    eps * (1 + std::min(p.T[k], 3.5)));
    bound = bound * (1 + 64 * eps) + std::numeric_limits<double>::min();
    p.valid[k] &= std::isfinite(bound) && bound <= p.budget[k];
    for (int d = 0; d < 27; d++)
      if (!std::isfinite(p.G[d][k]) || (p.G[d][k] != 0 && !std::isnormal(p.G[d][k])))
        p.valid[k] = false;
  }
}
}
