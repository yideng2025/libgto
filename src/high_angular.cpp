// Hermite Coulomb evaluation for f/g shells with raw-input reconstruction.
// Real regular solid harmonics use m=-l..l without the Condon--Shortley phase.
#include "internal.hpp"
#include "r_dependency.hpp"
#include "pure_output.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <type_traits>
#include "angular_products.hpp"
#include <cfenv>
namespace gto {
using HighMP = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<512, boost::multiprecision::digit_base_2>>;
template <class S> static S root(const S &x) {
  using std::sqrt;
  return sqrt(x);
}
template <class S> static S ex(const S &x) {
  using std::exp;
  return exp(x);
}
template <class S> static S ab(const S &x) {
  using std::abs;
  return abs(x);
}
static long double fac(int n) {
  long double x = 1;
  for (int j = 2; j <= n; j++)
    x *= j;
  return x;
}
static Real odd_df(int l) {
  Real x = 1;
  for (int j = 2 * l - 1; j > 1; j -= 2)
    x *= j;
  return x;
}
template <class S> static std::vector<S> harmonic(int l) {
  auto cart = orbitals(l);
  int nc = ncart(l);
  std::vector<S> t((2 * l + 1) * nc);
  if (l == 0) {
    t[0] = 1;
    return t;
  }
  if (l == 1) {
    for (int i = 0; i < 3; i++)
      t[i * 3 + i] = 1;
    return t;
  } // internal p=x,y,z
  for (int m = -l; m <= l; m++) {
    int am = std::abs(m);
    S scale = root(S(fac(l - am)) / S(fac(l + am))) * (am ? root(S(2)) : S(1));
    for (int k = 0; k <= (l - am) / 2; k++) {
      S zc = S(fac(2 * l - 2 * k)) /
             (S(std::ldexp(1.L, l)) * S(fac(k)) * S(fac(l - k)) * S(fac(l - am - 2 * k)));
      if (k & 1)
        zc = -zc;
      for (int iy = 0; iy <= am; iy++) {
        int phase = iy % 4;
        int ri = m < 0 ? (phase == 1   ? 1
                          : phase == 3 ? -1
                                       : 0)
                       : (phase == 0   ? 1
                          : phase == 2 ? -1
                                       : 0);
        if (!ri)
          continue;
        S cc = scale * zc * S(ri * fac(am) / (fac(iy) * fac(am - iy)));
        for (int kx = 0; kx <= k; kx++)
          for (int ky = 0; ky <= k - kx; ky++) {
            int kz = k - kx - ky;
            Triple xyz = {am - iy + 2 * kx, iy + 2 * ky, l - am - 2 * k + 2 * kz};
            auto pos = std::find(cart.begin(), cart.end(), xyz);
            require(pos != cart.end());
            int c = pos - cart.begin();
            t[(m + l) * nc + c] += cc * S(fac(k) / (fac(kx) * fac(ky) * fac(kz)));
          }
      }
    }
  }
  return t;
}
template <class S> struct HPrimitive {
  S p, K;
  std::array<S, 3> offset;
};
template <class S> struct RawPair {
  int A = 0, B = 0, H = 0, N = 0;
  size_t P = 0, C = 0;
  std::array<S, 3> center{};
  std::vector<HPrimitive<S>> primitives;
  std::vector<S> E, pure_E, weights;
  void build(const libgto_pair &r) {
    A = r.a.l;
    B = r.b.l;
    H = nh(A + B);
    N = ncart(A) * ncart(B);
    P = mul(r.a.k, r.b.k);
    C = mul(r.a.c, r.b.c);
    for (int d = 0; d < 3; d++)
      center[d] = S(r.a.r[d]);
    primitives.resize(P);
    E.assign(mul(mul(P, H), N), S(0));
    pure_E.assign(mul(mul(P, H), (2 * A + 1) * (2 * B + 1)), S(0));
    weights.resize(mul(P, C));
    auto oa = orbitals(A), ob = orbitals(B), hs = hermites(A + B);
    auto ta = harmonic<S>(A), tb = harmonic<S>(B);
    S pi = boost::math::constants::pi<S>();
    std::vector<S> wa(r.a.k * r.a.c), wb(r.b.k * r.b.c);
    auto shell_weights = [&](const Shell &s, std::vector<S> &v) {
      for (size_t p = 0; p < s.k; p++) {
        S a = S(s.a[p]), norm = 1;
        if (s.norm == LIBGTO_NORMALIZED_PRIMITIVES) {
          using std::pow;
          norm = pow(2 * a / pi, S(0.75)) * pow(4 * a, S(s.l) / 2) / root(S(odd_df(s.l)));
        }
        for (size_t c = 0; c < s.c; c++)
          v[p + s.k * c] = S(s.raw_coefficients[p + s.k * c]) * norm;
      }
    };
    shell_weights(r.a, wa);
    shell_weights(r.b, wb);
    S delta[3], dist = 0;
    for (int d = 0; d < 3; d++) {
      delta[d] = S(r.b.r[d]) - S(r.a.r[d]);
      dist += delta[d] * delta[d];
    }
    for (size_t j = 0; j < r.b.k; j++)
      for (size_t i = 0; i < r.a.k; i++) {
        size_t p = i + r.a.k * j;
        auto &v = primitives[p];
        S alpha = S(r.a.a[i]), beta = S(r.b.a[j]);
        v.p = alpha + beta;
        v.K = ex(S(-alpha / v.p * beta * dist));
        for (int d = 0; d < 3; d++)
          v.offset[d] = beta / v.p * delta[d];
        S e[3][5][5][10]{};
        for (int d = 0; d < 3; d++) {
          e[d][0][0][0] = 1;
          for (int a = 0; a <= A; a++)
            for (int b = 0; b <= B; b++) {
              if (!a && !b)
                continue;
              int pa = a ? a - 1 : 0, pb = a ? b : b - 1;
              S shift = a ? v.offset[d] : S(-alpha / v.p * delta[d]);
              for (int t = 0; t <= a + b; t++)
                e[d][a][b][t] = (t ? S(e[d][pa][pb][t - 1] / (2 * v.p)) : S(0)) +
                                shift * e[d][pa][pb][t] +
                                (t < 8 ? S((t + 1) * e[d][pa][pb][t + 1]) : S(0));
            }
        }
        for (int h = 0; h < H; h++)
          for (size_t b = 0; b < ob.size(); b++)
            for (size_t a = 0; a < oa.size(); a++) {
              S z = 1;
              for (int d = 0; d < 3; d++)
                z *= e[d][oa[a][d]][ob[b][d]][hs[h][d]];
              E[(p * H + h) * N + a + oa.size() * b] = z;
            }
        for (int h = 0; h < H; h++)
          for (int b = 0; b < 2 * B + 1; b++)
            for (int a = 0; a < 2 * A + 1; a++) {
              S z = 0;
              for (size_t j = 0; j < ob.size(); j++)
                if (tb[b * ob.size() + j] != 0)
                  for (size_t i = 0; i < oa.size(); i++)
                    if (ta[a * oa.size() + i] != 0)
                      z += ta[a * oa.size() + i] * tb[b * ob.size() + j] *
                           E[(p * H + h) * N + i + oa.size() * j];
              pure_E[(p * H + h) * (2 * A + 1) * (2 * B + 1) + a + (2 * A + 1) * b] = z;
            }
        for (size_t cb = 0; cb < r.b.c; cb++)
          for (size_t ca = 0; ca < r.a.c; ca++)
            weights[p + P * (ca + r.a.c * cb)] = wa[i + r.a.k * ca] * wb[j + r.b.k * cb];
      }
  }
  size_t bytes() const {
    return primitives.capacity() * sizeof(HPrimitive<S>) +
           (E.capacity() + pure_E.capacity() + weights.capacity()) * sizeof(S);
  }
  const S *ep(bool pure, size_t p) const {
    return (pure ? pure_E.data() : E.data()) + p * H * (pure ? (2 * A + 1) * (2 * B + 1) : N);
  }
};
struct HighScratch {
  RawPair<HighMP> a, b;
  std::vector<HighMP> arena;
  std::array<size_t, 14> shape{};
};
size_t high_scratch_bytes(const libgto_workspace &w) noexcept {
  return w.high ? sizeof(HighScratch) + w.high->a.bytes() + w.high->b.bytes() +
                      w.high->arena.capacity() * sizeof(HighMP)
                : 0;
}
size_t high_shape_mp_bytes(const QuartetShape &s, const Layout &l) {
  size_t n = add(l.total, 4096), temporary = 0;
  for (int side = 0; side < 2; side++) {
    int a = 2 * side, b = a + 1;
    size_t P = mul(s.k[a], s.k[b]);
    n = add(n, mul(P, add(6, add(nh(s.l[a] + s.l[b]) * (ncart(s.l[a]) * ncart(s.l[b]) +
                                                        (2 * s.l[a] + 1) * (2 * s.l[b] + 1)),
                                 s.c[a] * s.c[b]))));
    temporary = std::max(temporary, add(mul(s.k[a], s.c[a]), mul(s.k[b], s.c[b])));
  }
  return add(sizeof(HighScratch), mul(add(n, temporary), sizeof(HighMP)));
}
size_t high_mp_bytes(const libgto_pair &a, const libgto_pair &b, const Layout &l) {
  return high_shape_mp_bytes(quartet_shape(a, b), l);
}
struct AngularView {
  const std::vector<Real> *E;
  const std::vector<double> *rounded;
  const std::vector<double> *norm;
  int components;
};
struct WidePair {
  const libgto_pair &r;
  int A, B, H, N;
  size_t P, C;
  std::array<Real, 3> center;
  AngularView cart, pure;
  WidePair(const libgto_pair &p)
      : r(p), A(p.a.l), B(p.b.l), H(p.H), N(p.N), P(p.primitives.size()), C(p.a.c * p.b.c),
        center(p.a.r), cart{&p.E, &p.E64, &p.norm_cart, p.N},
        pure{p.pure_E.empty() ? &p.E : &p.pure_E, p.pure_E.empty() ? &p.E64 : &p.pure_E64,
             (1 && p.pure_E.empty()) ? &p.norm_cart : &p.norm_pure, (2 * A + 1) * (2 * B + 1)} {}
  const auto &prim(size_t i) const {
    return r.primitives[i];
  }
  const Real *ep(bool projected, size_t p) const {
    const auto &v = projected ? pure : cart;
    return v.E->data() + p * H * v.components;
  }
  const double *ep64(bool projected, size_t p) const {
    const auto &v = projected ? pure : cart;
    return v.rounded->data() + p * H * v.components;
  }
  Real weight(size_t p, size_t c) const {
    return r.weights[p + P * c];
  }
  Real weight_sum(size_t p) const {
    Real z = 0;
    for (size_t c = 0; c < C; c++)
      z += std::abs(weight(p, c));
    return z;
  }
  Real weight_max(size_t p) const {
    Real z = 0;
    for (size_t c = 0; c < C; c++)
      z = std::max(z, std::abs(weight(p, c)));
    return z;
  }
  Real output_scale() const {
    Real factor = 1;
    for (const Shell *s : {&r.a, &r.b})
      if (s->norm == LIBGTO_NORMALIZED_PRIMITIVES) {
        Real minimum = std::numeric_limits<Real>::max();
        for (int x = 0; x <= s->l; x++)
          for (int y = 0; y <= s->l - x; y++) {
            int z = s->l - x - y;
            Real df = odd_df(x) * odd_df(y) * odd_df(z);
            minimum = std::min(minimum, df);
          }
        factor *= std::sqrt(odd_df(s->l) / minimum);
      }
    return factor;
  }
  Real norm(bool projected, size_t p) const {
    const auto &v = projected ? pure : cart;
    return v.norm->empty() || ((A <= 2 && B <= 2) && !r.fp64_safe)
               ? std::numeric_limits<Real>::infinity()
               : Real((*v.norm)[p * (v.components + 1)]);
  }
};
template <class S> struct MPView {
  const RawPair<S> &r;
  int A, B, H, N;
  size_t P, C;
  std::array<S, 3> center;
  MPView(const RawPair<S> &v)
      : r(v), A(v.A), B(v.B), H(v.H), N(v.N), P(v.P), C(v.C), center(v.center) {}
  const auto &prim(size_t i) const {
    return r.primitives[i];
  }
  const S *ep(bool pure, size_t p) const {
    return r.ep(pure, p);
  }
  S weight(size_t p, size_t c) const {
    return r.weights[p + r.P * c];
  }
};
using HighRStep = HermiteRStep;
struct HighRPlan {
  std::vector<int> offset;
  std::vector<HighRStep> steps;
  std::vector<Triple> hs;
  HighRPlan(int L) : offset(L + 1), hs(hermites(L)) {
    for (int m = 1; m <= L; m++)
      offset[m] = offset[m - 1] + nh(L - m + 1);
    emit_r_dependencies(L, offset, [&](HighRStep step) { steps.push_back(step); });
  }
};
static const HighRPlan &hplan(int l) {
  static const std::array<HighRPlan, 17> p = {
      HighRPlan(0),  HighRPlan(1),  HighRPlan(2),  HighRPlan(3),  HighRPlan(4),  HighRPlan(5),
      HighRPlan(6),  HighRPlan(7),  HighRPlan(8),  HighRPlan(9),  HighRPlan(10), HighRPlan(11),
      HighRPlan(12), HighRPlan(13), HighRPlan(14), HighRPlan(15), HighRPlan(16)};
  return p[l];
}
static const std::vector<unsigned short> &coupling(int A, int B) {
  static const std::array<std::vector<unsigned short>, 81> maps = []() {
    std::array<std::vector<unsigned short>, 81> r;
    for (int a = 0; a <= 8; a++)
      for (int b = 0; b <= 8; b++) {
        auto &v = r[a * 9 + b];
        v.reserve(nh(a) * nh(b));
        emit_hermite_coupling(
            a, b, [](int h) { return hermite_triple(h); },
            [&](unsigned short code) { v.push_back(code); });
      }
    return r;
  }();
  return maps[A * 9 + B];
}
template <class S> static void fundamental(int m, S T, S *f) {
  if (T == 0) {
    for (int i = 0; i <= m; i++)
      f[i] = S(1) / (2 * i + 1);
    return;
  }
  if (T < S(2 * m + 64)) {
    S term = 1, sum = 1;
    int k = 0;
    for (; k < 2048; k++) {
      term *= 2 * T / (2 * m + 2 * k + 3);
      sum += term;
      S ratio = 2 * T / (2 * m + 2 * k + 5);
      if (ratio < 1 && term * ratio / (1 - ratio) <= std::numeric_limits<S>::epsilon() * sum)
        break;
    }
    require(k < 2048, LIBGTO_NUMERICAL_FAILURE);
    S e = ex(S(-T));
    f[m] = e * sum / (2 * m + 1);
    for (int n = m; n > 0; n--)
      f[n - 1] = (2 * T * f[n] + e) / (2 * n - 1);
  } else {
    S e = ex(S(-T));
    if constexpr (std::is_same_v<S, Real>)
      f[0] = root(boost::math::constants::pi<S>()) * std::erf(root(T)) / (2 * root(T));
    else
      f[0] = root(boost::math::constants::pi<S>()) * boost::math::erf(root(T)) / (2 * root(T));
    for (int n = 0; n < m; n++)
      f[n + 1] = ((2 * n + 1) * f[n] - e) / (2 * T);
  }
}

bool high_angular(const libgto_pair &a, const libgto_pair &b) {
  return a.a.l > 2 || a.b.l > 2 || b.a.l > 2 || b.b.l > 2;
}
void prepare_high_pair(libgto_pair &r) {
  { coupling(r.a.l + r.b.l, 0); }
  RawPair<Real> p;
  p.build(r);
  r.H = p.H;
  r.N = p.N;
  r.primitives.resize(p.P);
  for (size_t j = 0; j < p.P; j++) {
    auto &v = r.primitives[j];
    v.p = p.primitives[j].p;
    v.K = p.primitives[j].K;
    v.offset = p.primitives[j].offset;
  }
  r.E = std::move(p.E);
  r.pure_E = std::move(p.pure_E);
  r.weights = std::move(p.weights);
  r.fp64_safe = true;
  rounded_view(r.E, r.E64, r.fp64_safe);
  rounded_view(r.pure_E, r.pure_E64, r.fp64_safe);
  rounded_view(r.weights, r.weights64, r.fp64_safe);
  auto ta = harmonic<Real>(r.a.l), tb = harmonic<Real>(r.b.l);
  int na = ncart(r.a.l), nb = ncart(r.b.l), pa = 2 * r.a.l + 1, pb = 2 * r.b.l + 1;
  r.norm_cart.resize(p.P * (r.N + 1));
  r.norm_pure.resize(p.P * (pa * pb + 1));
  for (size_t z = 0; z < p.P; z++) {
    Real cartmax = 0, puremax = 0;
    for (int c = 0; c < r.N; c++) {
      Real sum = 0;
      for (int h = 0; h < r.H; h++)
        sum += std::abs(r.E[(z * r.H + h) * r.N + c]);
      cartmax = std::max(cartmax, sum);
    }
    for (int b = 0; b < pb; b++)
      for (int a = 0; a < pa; a++) {
        Real sum = 0;
        for (int h = 0; h < r.H; h++)
          for (int j = 0; j < nb; j++)
            if (tb[b * nb + j] != 0)
              for (int i = 0; i < na; i++)
                if (ta[a * na + i] != 0)
                  sum += std::abs(ta[a * na + i] * tb[b * nb + j] *
                                  r.E[(z * r.H + h) * r.N + i + na * j]);
        puremax = std::max(puremax, sum);
      }
    auto up = [](Real x) {
      return x > std::numeric_limits<double>::max()
                 ? std::numeric_limits<double>::infinity()
                 : std::nextafter(double(x), std::numeric_limits<double>::infinity()) *
                       (1 + 64 * std::numeric_limits<double>::epsilon());
    };
    r.norm_cart[z * (r.N + 1)] = up(cartmax);
    r.norm_pure[z * (pa * pb + 1)] = up(puremax);
  }
}
template <class S, class View, bool Narrow = false, int Blocked = 0>
static S evaluate(const View &a, const View &b, const libgto_eri_options &o, const Layout &l,
                  S *arena, double *fast = nullptr, size_t *attempted = nullptr) {
  S *R = arena, *M = R + l.r, *U = M + l.m, *G = U + l.u, *tmp = G + l.g, *out = tmp + l.tmp;
  std::fill_n(out, l.contracted, S(0));
  const bool max_component = 1 == 1;
  S magnitude = 0, majorant = 0;
  const auto &rp = hplan(l.L);
  const auto &ah = hplan(a.A + a.B).hs;
  const auto &bh = hplan(b.A + b.B).hs;
  bool pure = o.representation == LIBGTO_PURE;
  S monitor_scale = 1;
  if constexpr (std::is_same_v<S, Real>)
    if (!pure)
      monitor_scale = a.output_scale() * b.output_scale();
  int coupling_strategy = 0;
  const unsigned short *coupled = nullptr;
  if constexpr (std::is_same_v<S, Real>) {
    coupling_strategy = 3;
    if (coupling_strategy)
      coupled = coupling(a.A + a.B, b.A + b.B).data();
  }
  const bool reuse_casts = coupling_strategy == 3 && std::fegetround() == FE_TONEAREST;
  S pi = boost::math::constants::pi<S>();
  auto primitive = [&](size_t p, size_t q) {
    if (attempted)
      ++*attempted;
    const auto &x = a.prim(p);
    const auto &y = b.prim(q);
    S rho = x.p / (x.p + y.p) * y.p, T = 0, X[3];
    for (int d = 0; d < 3; d++) {
      X[d] = (a.center[d] - b.center[d]) + (x.offset[d] - y.offset[d]);
      T += rho * X[d] * X[d];
    }
    S f[17];
    fundamental(l.L, T, f);
    S scale = 1;
    for (int m = 0; m <= l.L; m++) {
      R[rp.offset[m]] = scale * f[m];
      scale *= -2 * rho;
    }
    for (auto s : rp.steps) {
      S v = X[s.axis] * R[s.first];
      if (s.coefficient)
        v += s.coefficient * R[s.second];
      R[s.dst] = v;
    }
    S pref = 2 * pi * pi * root(pi) * x.K * y.K / (x.p * y.p * root(S(x.p + y.p)));
    S maxM = 0;
    if (coupled) {
      // All final triples of degree<=L occur in the unrestricted coupling map.
      // All recurrence consumers have finished: its m=0 prefix can be scaled in place.
      for (int h = 0; h < nh(l.L); h++) {
        R[h] = pref * R[h];
        if constexpr (std::is_same_v<S, Real>)
          maxM = std::max(maxM, ab(R[h]));
      }
      if (!Narrow || (coupling_strategy != 2 && coupling_strategy != 3))
        for (size_t i = 0; i < l.m; i++) {
          unsigned code = coupled[i];
          S z = R[code & 0x7fff];
          M[i] = (code & 0x8000) ? -z : z;
        }
    } else
      for (int h = 0; h < l.hab; h++)
        for (int k = 0; k < l.hcd; k++) {
          const auto &v = ah[h];
          const auto &w = bh[k];
          S z = pref * R[hi(v[0] + w[0], v[1] + w[1], v[2] + w[2])];
          M[h * l.hcd + k] = ((w[0] + w[1] + w[2]) & 1) ? -z : z;
          if constexpr (std::is_same_v<S, Real>)
            maxM = std::max(maxM, ab(z));
        }
    if constexpr (std::is_same_v<S, Real>)
      majorant += maxM * a.norm(pure, p) * b.norm(pure, q) *
                  (max_component ? a.weight_max(p) : a.weight_sum(p)) *
                  (max_component ? b.weight_max(q) : b.weight_sum(q)) *
                  (max_component ? S(1) : S(l.g));
    if constexpr (Narrow) {
      static_assert(std::is_same_v<S, Real>);
      Real factor = 4 * Real(std::numeric_limits<double>::epsilon()) * (2 * l.hab + 2 * l.hcd + 16);
      if (!std::isfinite(majorant) || factor * majorant * monitor_scale > 0.0625e-10L)
        return false;
      double *Md = fast, *Ud = Md + l.m, *Gd = Ud + l.u;
      if (coupled && coupling_strategy == 3 && reuse_casts) {
        // Every final spatial index occurs in the unrestricted coupling map.
        // Nearest rounding commutes with a sign flip, including signed zero.
        // Check/cast each unique value once; other rounding modes keep the old path.
        double scaled64[nh(16)];
        for (int h = 0; h < nh(l.L); h++) {
          Real z = R[h];
          if (!std::isfinite(z) || std::abs(z) > std::numeric_limits<double>::max())
            return false;
          double v = double(z);
          if (z != 0 && !std::isnormal(v))
            return false;
          scaled64[h] = v;
        }
        // Exact two-term ps/sp ket support, pure fs/sf bra only. The general
        // recurrence, unique-value conversion and complete old error guard above
        // are unchanged. Keep the original E64 view and ascending nonzero k order.
        if (1 && pure && l.hab == 20 && l.nab == 7 && l.hcd == 4 && l.ncd == 3 &&
            ((a.A == 3 && a.B == 0) || (a.A == 0 && a.B == 3)) &&
            ((b.A == 1 && b.B == 0) || (b.A == 0 && b.B == 1))) {
          const double *ec = b.ep64(pure, q);
          for (int h = 0; h < 20; h++)
            for (int c = 0; c < 3; c++) {
              unsigned z = coupled[h * 4], d = coupled[h * 4 + 3 - c];
              double s0 = scaled64[z & 0x7fff], s1 = scaled64[d & 0x7fff];
              if (z & 0x8000)
                s0 = -s0;
              if (d & 0x8000)
                s1 = -s1;
              double sum = 0;
              sum += s0 * ec[c];
              sum += s1 * ec[(3 - c) * 3 + c];
              Ud[h * 3 + c] = sum;
            }
          angular_bra_ordered(20, 7, 3, a.ep64(pure, p), Ud, Gd);
          for (size_t v = 0; v < l.g; v++) {
            if (!std::isfinite(Gd[v]))
              return false;
            G[v] = Real(Gd[v]);
          }
          return true;
        }
        if (1 && pure &&
            angular_unique_R(l.hab, l.hcd, l.nab, l.ncd, coupled, scaled64, a.ep64(pure, p),
                             b.ep64(pure, q), Ud, Gd)) {
          for (size_t v = 0; v < l.g; v++) {
            if (!std::isfinite(Gd[v]))
              return false;
            G[v] = Real(Gd[v]);
          }
          return true;
        }
        for (size_t i = 0; i < l.m; i++) {
          unsigned code = coupled[i];
          double v = scaled64[code & 0x7fff];
          Md[i] = (code & 0x8000) ? -v : v;
        }
      } else if (coupled && (coupling_strategy == 2 || coupling_strategy == 3)) {
        for (size_t i = 0; i < l.m; i++) {
          unsigned code = coupled[i];
          Real z = R[code & 0x7fff];
          if (code & 0x8000)
            z = -z;
          if (!std::isfinite(z) || std::abs(z) > std::numeric_limits<double>::max())
            return false;
          double v = double(z);
          if (z != 0 && !std::isnormal(v))
            return false;
          Md[i] = v;
        }
      } else
        for (size_t i = 0; i < l.m; i++) {
          if (!std::isfinite(M[i]) || std::abs(M[i]) > std::numeric_limits<double>::max())
            return false;
          double v = double(M[i]);
          if (M[i] != 0 && !std::isnormal(v))
            return false;
          Md[i] = v;
        }
      if constexpr (Blocked == 2)
        angular_products_ordered(l.hab, l.hcd, l.nab, l.ncd, Md, a.ep64(pure, p), b.ep64(pure, q),
                                 Ud, Gd);
      else if constexpr (Blocked == 1)
        angular_products_block4(l.hab, l.hcd, l.nab, l.ncd, Md, a.ep64(pure, p), b.ep64(pure, q),
                                Ud, Gd);
      else
        angular_products(l.hab, l.hcd, l.nab, l.ncd, Md, a.ep64(pure, p), b.ep64(pure, q), Ud, Gd);
      for (size_t v = 0; v < l.g; v++) {
        if (!std::isfinite(Gd[v]))
          return false;
        G[v] = Real(Gd[v]);
      }
    } else {
      const S *EA = a.ep(pure, p), *EB = b.ep(pure, q);
      for (int h = 0; h < l.hab; h++)
        for (int v = 0; v < l.ncd; v++) {
          S z = 0;
          for (int k = 0; k < l.hcd; k++)
            if (EB[k * l.ncd + v] != 0)
              z += M[h * l.hcd + k] * EB[k * l.ncd + v];
          U[h * l.ncd + v] = z;
        }
      for (int v = 0; v < l.ncd; v++)
        for (int u = 0; u < l.nab; u++) {
          S z = 0;
          for (int h = 0; h < l.hab; h++)
            if (EA[h * l.nab + u] != 0)
              z += EA[h * l.nab + u] * U[h * l.ncd + v];
          G[u + l.nab * v] = z;
        }
    }
    return true;
  };
  if (l.cab == 1 && l.ccd == 1) {
    for (size_t q = 0; q < b.P; q++)
      for (size_t p = 0; p < a.P; p++) {
        if (!primitive(p, q))
          return std::numeric_limits<S>::infinity();
        S wt = a.weight(p, 0) * b.weight(q, 0);
        for (size_t v = 0; v < l.g; v++) {
          S term = wt * G[v];
          out[v] += term;
          magnitude += ab(term);
        }
      }
  } else {
    size_t inner = l.abfirst ? a.P : b.P, outer = l.abfirst ? b.P : a.P,
           ni = l.abfirst ? l.cab : l.ccd, no = l.abfirst ? l.ccd : l.cab;
    for (size_t q = 0; q < outer; q++) {
      std::fill_n(tmp, l.tmp, S(0));
      for (size_t p = 0; p < inner; p++) {
        if (!primitive(l.abfirst ? p : q, l.abfirst ? q : p))
          return std::numeric_limits<S>::infinity();
        for (size_t c = 0; c < ni; c++) {
          S wt = l.abfirst ? a.weight(p, c) : b.weight(p, c);
          for (size_t v = 0; v < l.g; v++)
            tmp[v + l.g * c] += wt * G[v];
        }
      }
      for (size_t d = 0; d < no; d++)
        for (size_t c = 0; c < ni; c++) {
          S wt = l.abfirst ? b.weight(q, d) : a.weight(q, d);
          size_t slot = l.abfirst ? c + l.cab * d : d + l.cab * c;
          for (size_t v = 0; v < l.g; v++) {
            S term = wt * tmp[v + l.g * c];
            out[v + l.g * slot] += term;
            magnitude += ab(term);
          }
        }
    }
  }
  if constexpr (std::is_same_v<S, Real>)
    if (!std::isfinite(majorant) || !std::isfinite(magnitude))
      return std::numeric_limits<Real>::infinity();
  return (max_component ? majorant : std::max(magnitude, majorant)) * monitor_scale;
}
template <class S>
static void publish(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
                    const Layout &l, const S *from, double *out) {
  if (o.representation == LIBGTO_PURE) {
    store_pure_contracted(a, b, from, out, l);
    return;
  }
  const Shell *ss[4] = {&a.a, &a.b, &b.a, &b.b};
  size_t nc[4], dim[4];
  S scales[4][15];
  for (int s = 0; s < 4; s++) {
    nc[s] = o.representation == LIBGTO_PURE ? 2 * ss[s]->l + 1 : ncart(ss[s]->l);
    dim[s] = nc[s] * ss[s]->c;
    auto orbs = orbitals(ss[s]->l);
    for (size_t c = 0; c < nc[s]; c++) {
      scales[s][c] = 1;
      if (o.representation == LIBGTO_CARTESIAN && ss[s]->norm == LIBGTO_NORMALIZED_PRIMITIVES) {
        Real df = 1;
        for (int x : orbs[c])
          for (int k = 2 * x - 1; k > 1; k -= 2)
            df *= k;
        scales[s][c] = root(S(odd_df(ss[s]->l)) / S(df));
      }
    }
  }
  for (size_t cb = 0; cb < l.ccd; cb++)
    for (size_t ca = 0; ca < l.cab; ca++)
      for (size_t d = 0; d < nc[3]; d++)
        for (size_t c = 0; c < nc[2]; c++)
          for (size_t j = 0; j < nc[1]; j++)
            for (size_t i = 0; i < nc[0]; i++) {
              auto perm = [&](size_t v, int axis) {
                return o.representation == LIBGTO_PURE && ss[axis]->l == 1 ? (v + 1) % 3 : v;
              };
              size_t src =
                  perm(i, 0) + nc[0] * (perm(j, 1) + nc[1] * (perm(c, 2) + nc[2] * perm(d, 3)));
              size_t dst =
                  i + nc[0] * (ca % a.a.c) +
                  dim[0] *
                      (j + nc[1] * (ca / a.a.c) +
                       dim[1] * (c + nc[2] * (cb % b.a.c) + dim[2] * (d + nc[3] * (cb / b.a.c))));
              S v = from[src + l.g * (ca + l.cab * cb)] * scales[0][i] * scales[1][j] *
                    scales[2][c] * scales[3][d];
              require(v == v && ab(v) <= S(std::numeric_limits<double>::max()),
                      LIBGTO_NUMERICAL_FAILURE);
              out[dst] = static_cast<double>(v);
            }
}
void compute_high(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
                  libgto_workspace &w, double *out, const Layout &l) {
  if (1 && w.policy == LIBGTO_PRECISION_AUTO && (l.L & 1) && a.a.r == a.b.r && a.a.r == b.a.r &&
      a.a.r == b.b.r) {
    std::fill_n(out, l.public_size.elements, 0.);
    w.stats = {};
    ++w.precision.wide_calls;
    return;
  }
  if (w.policy == LIBGTO_PRECISION_REFERENCE) {
    // REFERENCE remains a raw, unmerged wide calculation, not a truth oracle.
    w.high.reset();
    size_t base = add(mul(w.arena.capacity(), sizeof(Real)),
                      mul(add(w.fast.capacity(), w.borrowed.capacity()), sizeof(double)));
    if (w.precise)
      base = add(base, sensitive_workspace_bytes());
    require(add(base, high_mp_bytes(a, b, l)) <= w.capacity_limit, LIBGTO_RESOURCE_LIMIT);
    libgto_pair aa, bb;
    aa.a = a.a;
    aa.b = a.b;
    bb.a = b.a;
    bb.b = b.b;
    prepare_high_pair(aa);
    prepare_high_pair(bb);
    evaluate<Real>(WidePair(aa), WidePair(bb), o, l, w.arena.data());
    publish(a, b, o, l, w.arena.data() + l.r + l.m + l.u + l.g + l.tmp, out);
    w.stats = {};
    w.stats.primitive_quartets = a.primitives.size() * b.primitives.size();
    ++w.precision.wide_calls;
    return;
  }
  w.stats = {};
  uint64_t primitive_work = 0;
  bool sensitive = a.sensitive || b.sensitive;
  Real magnitude = 0;
  bool took_narrow = false;
  if (!sensitive) {
    const auto &aa = a.optimized ? *a.optimized : a;
    const auto &bb = b.optimized ? *b.optimized : b;
    if (w.policy == LIBGTO_PRECISION_AUTO && aa.fp64_safe && bb.fp64_safe) {
      size_t tried = 0;
      require(w.fast.size() >= l.m + l.u + l.g, LIBGTO_RESOURCE_LIMIT);
      {
        magnitude = evaluate<Real, WidePair, true, 2>(WidePair(aa), WidePair(bb), o, l,
                                                      w.arena.data(), w.fast.data(), &tried);
      }
      primitive_work += tried;
      w.precision.fp64_primitive_quartets += tried;
      Real error = magnitude * (4 * Real(std::numeric_limits<double>::epsilon()) *
                                (2 * l.hab + 2 * l.hcd + 16));
      took_narrow = std::isfinite(magnitude) && error <= 0.0625e-10L;
      if (!took_narrow)
        ++w.precision.fp64_fallbacks;
    }
    if (!took_narrow) {
      magnitude = evaluate<Real>(WidePair(aa), WidePair(bb), o, l, w.arena.data());
      primitive_work += aa.primitives.size() * bb.primitives.size();
    }
    Real risk = magnitude * std::numeric_limits<Real>::epsilon() * Real(8192) *
                (1 + a.primitives.size() * b.primitives.size());
    sensitive = !std::isfinite(magnitude) || risk > 1e-11L;
  }
  if (sensitive) {
    size_t base = add(mul(w.arena.capacity(), sizeof(Real)),
                      mul(add(w.fast.capacity(), w.borrowed.capacity()), sizeof(double)));
    if (w.precise)
      base = add(base, sensitive_workspace_bytes());
    require(add(base, high_mp_bytes(a, b, l)) <= w.capacity_limit, LIBGTO_RESOURCE_LIMIT);
    std::array<size_t, 14> shape{};
    const Shell *shells[4] = {&a.a, &a.b, &b.a, &b.b};
    for (int i = 0; i < 4; i++) {
      shape[3 * i] = shells[i]->l;
      shape[3 * i + 1] = shells[i]->k;
      shape[3 * i + 2] = shells[i]->c;
    }
    shape[12] = o.representation;
    shape[13] = o.contraction_order;
    if (!w.high || w.high->shape != shape) {
      w.high.reset();
      w.high = std::make_shared<HighScratch>();
      w.high->shape = shape;
    }
    auto &h = *w.high;
    h.a.build(a);
    h.b.build(b);
    h.arena.resize(l.total);
    evaluate<HighMP>(MPView<HighMP>(h.a), MPView<HighMP>(h.b), o, l, h.arena.data());
    primitive_work += a.primitives.size() * b.primitives.size();
    publish(a, b, o, l, h.arena.data() + l.r + l.m + l.u + l.g + l.tmp, out);
    ++w.sensitive_calls;
  } else
    publish(a, b, o, l, w.arena.data() + l.r + l.m + l.u + l.g + l.tmp, out);
  w.stats.primitive_quartets = primitive_work;
  if (took_narrow && !sensitive)
    ++w.precision.fp64_calls;
  else
    ++w.precision.wide_calls;
}
}
