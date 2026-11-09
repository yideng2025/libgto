// Pure d + p Coulomb moment panel. Isotropic same-electron covariance cancels
// before evaluation because each real d polynomial is harmonic (trace free).
#pragma once
#include "internal.hpp"
#include <utility>
#include "three_p_moment_policy.hpp"
namespace quadratic_linear {
constexpr double rt = 1.7320508075688772935274463415058723669;
struct Packet {
  double a[3][64], c[3][64], v[3][64], z[3][64], B[3][64], G[15][64], F[4][64], T[64], pref[64],
      cross[64], budget[64];
  bool valid[64];
};
template <int M, int K> static double gradient(const Packet &p, size_t k) {
  if constexpr (M == 0) {
    if constexpr (K == 0)
      return rt * p.B[1][k];
    if constexpr (K == 1)
      return rt * p.B[0][k];
  }
  if constexpr (M == 1) {
    if constexpr (K == 1)
      return rt * p.B[2][k];
    if constexpr (K == 2)
      return rt * p.B[1][k];
  }
  if constexpr (M == 2)
    return K == 2 ? 2 * p.B[2][k] : -p.B[K][k];
  if constexpr (M == 3) {
    if constexpr (K == 0)
      return rt * p.B[2][k];
    if constexpr (K == 2)
      return rt * p.B[0][k];
  }
  if constexpr (M == 4) {
    if constexpr (K == 0)
      return rt * p.B[0][k];
    if constexpr (K == 1)
      return -rt * p.B[1][k];
  }
  return 0;
}
template <int M, int K> static void emit(Packet &p, size_t k, double h, double j) {
  double value = p.c[K][k] * h + p.z[K][k] * j;
  if constexpr ((M == 0 && K != 2) || (M == 1 && K != 0) || M == 2 || (M == 3 && K != 1) ||
                (M == 4 && K != 2))
    value += p.cross[k] * gradient<M, K>(p, k);
  p.G[M + 5 * K][k] = p.pref[k] * value;
}
template <int M> static void component(Packet &p, size_t n) {
  for (size_t k = 0; k < n; k++) {
    double ax = p.a[0][k], ay = p.a[1][k], az = p.a[2][k], vx = p.v[0][k], vy = p.v[1][k],
           vz = p.v[2][k], q0, q1, q2;
    if constexpr (M == 0) {
      q0 = rt * (ax * ay);
      q1 = rt * (ax * vy + ay * vx);
      q2 = rt * (vx * vy);
    }
    if constexpr (M == 1) {
      q0 = rt * (ay * az);
      q1 = rt * (ay * vz + az * vy);
      q2 = rt * (vy * vz);
    }
    if constexpr (M == 2) {
      q0 = az * az - .5 * (ax * ax + ay * ay);
      q1 = 2 * az * vz - (ax * vx + ay * vy);
      q2 = vz * vz - .5 * (vx * vx + vy * vy);
    }
    if constexpr (M == 3) {
      q0 = rt * (ax * az);
      q1 = rt * (ax * vz + az * vx);
      q2 = rt * (vx * vz);
    }
    if constexpr (M == 4) {
      q0 = (.5 * rt) * (ax * ax - ay * ay);
      q1 = rt * (ax * vx - ay * vy);
      q2 = (.5 * rt) * (vx * vx - vy * vy);
    }
    double h = (q0 * p.F[0][k] + q1 * p.F[1][k]) + q2 * p.F[2][k],
           j = (q0 * p.F[1][k] + q1 * p.F[2][k]) + q2 * p.F[3][k];
    emit<M, 0>(p, k, h, j);
    emit<M, 1>(p, k, h, j);
    emit<M, 2>(p, k, h, j);
  }
}
static void angular(Packet &p, size_t n) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const int policy = 2;
  const bool fast = policy != 0;
  for (size_t k = 0; k < n; k++) {
    double f[4]{};
    if (p.valid[k])
      cubic_boys::evaluate(policy, p.T[k], f);
    for (int m = 0; m < 4; m++)
      p.F[m][k] = f[m];
  }
  for (int d = 0; d < 3; d++)
    for (size_t k = 0; k < n; k++)
      p.B[d][k] = p.a[d][k] * p.F[1][k] + p.v[d][k] * p.F[2][k];
  component<0>(p, n);
  component<1>(p, n);
  component<2>(p, n);
  component<3>(p, n);
  component<4>(p, n);
  for (size_t k = 0; k < n; k++) {
    double ma = 0, mc = 0;
    for (int d = 0; d < 3; d++) {
      ma = std::max(ma, std::abs(p.a[d][k]) + std::abs(p.v[d][k]));
      mc = std::max(mc, std::abs(p.c[d][k]) + std::abs(p.z[d][k]));
    }
    double env = 2 * ma * ma * mc + 2 * p.cross[k] * ma;
    double bound = std::abs(p.pref[k]) * p.F[0][k] * env *
                   ((2048 + (fast && (p.T[k] < 32 || (policy >= 2 && p.T[k] < 1e8)) ? 256 : 0)) *
                    eps * (1 + std::min(p.T[k], 3.5)));
    bound = bound * (1 + 64 * eps) + std::numeric_limits<double>::min();
    p.valid[k] &= std::isfinite(bound) && bound <= p.budget[k];
    for (int v = 0; v < 15; v++)
      if (!std::isfinite(p.G[v][k]) || (p.G[v][k] != 0 && !std::isnormal(p.G[v][k])))
        p.valid[k] = false;
  }
}
}
