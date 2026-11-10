#include "internal.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>
namespace gto {
// Bounded fixed-precision reconstruction from original immutable input.
// Boost supplies arithmetic; all integral formulas remain autonomous.
// Reconstruct primitive geometry, E and normalization from immutable original
// inputs, rather than extending already-rounded pair coefficients.
using MP = boost::multiprecision::number<boost::multiprecision::backends::cpp_bin_float<
                                             256, boost::multiprecision::backends::digit_base_2>,
                                         boost::multiprecision::et_off>;
static const MP &mp_pi() {
  static const MP p = acos(MP(-1));
  return p;
}

static void fundamental(int L, const MP &T, MP *F) {
  if (T == 0) {
    for (int m = 0; m <= L; m++)
      F[m] = MP(1) / (2 * m + 1);
    return;
  }
  if (T < 256) {
    MP sum = 1, term = 1;
    int k = 0;
    for (; k < 4096; k++) {
      term *= 2 * T / (2 * L + 2 * k + 3);
      sum += term;
      if (abs(term) < abs(sum) * ldexp(MP(1), -264))
        break;
    }
    require(k < 4096, LIBGTO_NUMERICAL_FAILURE);
    MP e = exp(-T);
    F[L] = e * sum / (2 * L + 1);
    for (int m = L; m > 0; m--)
      F[m - 1] = (2 * T * F[m] + e) / (2 * m - 1);
  } else {
    // For m <= 8 and T >= 256, the omitted upper incomplete-Gamma tail is below
    // 1e-96 relative, below 256-bit rounding throughout the s/p/d domain.
    F[0] = sqrt(mp_pi()) / (2 * sqrt(T));
    for (int m = 0; m < L; m++)
      F[m + 1] = (2 * m + 1) * F[m] / (2 * T);
  }
}

static MP coefficient(const Shell &s, size_t p, size_t c) {
  MP v = s.raw_coefficients[p + s.k * c];
  if (s.norm == LIBGTO_UNNORMALIZED)
    return v;
  MP x = 2 * MP(s.a[p]) / mp_pi();
  v *= sqrt(sqrt(x * x * x));
  if (s.l == 1)
    v *= sqrt(4 * MP(s.a[p]));
  else if (s.l == 2)
    v *= 4 * MP(s.a[p]) / sqrt(MP(3));
  return v;
}

struct PairMP {
  MP p, K, P[3], E[1260], weight;
};
struct SensitiveScratch {
  PairMP x, y;
  MP sum[1296], mass[1296], F[9], R[495], U[1260], G[1296], e[3][3][3][5], cart[36],
      transformed[36];
};
static_assert(sizeof(SensitiveScratch) <= 1024 * 1024,
              "fixed precision scratch exceeds supported capacity");
void SensitiveDeleter::operator()(SensitiveScratch *p) const noexcept {
  delete p;
}

size_t sensitive_workspace_bytes() noexcept {
  return sizeof(SensitiveScratch);
}

void reserve_sensitive(libgto_workspace &w) {
  if (!w.precise) {
    auto p = std::unique_ptr<SensitiveScratch, SensitiveDeleter>(new SensitiveScratch);
    (void)mp_pi();
    w.precise = std::move(p);
  }
}

static void build_pair(const libgto_pair &a, size_t p, size_t channel, bool pure, PairMP &v,
                       SensitiveScratch &scratch) {
  size_t i = p % a.a.k, j = p / a.a.k;
  MP alpha = a.a.a[i], beta = a.b.a[j];
  v.p = alpha + beta;
  MP distance = 0, delta[3];
  auto &e = scratch.e;
  for (auto &axis : e)
    for (auto &ia : axis)
      for (auto &ib : ia)
        for (auto &value : ib)
          value = 0;
  for (int d = 0; d < 3; d++) {
    delta[d] = MP(a.b.r[d]) - MP(a.a.r[d]);
    distance += delta[d] * delta[d];
    v.P[d] = beta / v.p * delta[d];
  }
  v.K = exp(-(alpha / v.p) * beta * distance);
  for (int d = 0; d < 3; d++) {
    e[d][0][0][0] = 1;
    for (int ia = 0; ia <= a.a.l; ia++)
      for (int ib = 0; ib <= a.b.l; ib++) {
        if (!ia && !ib)
          continue;
        int pa = ia ? ia - 1 : 0, pb = ia ? ib : ib - 1;
        MP shift = ia ? v.P[d] : -(alpha / v.p) * delta[d];
        for (int h = 0; h <= ia + ib; h++)
          e[d][ia][ib][h] = (h ? e[d][pa][pb][h - 1] / (2 * v.p) : MP(0)) +
                            shift * e[d][pa][pb][h] +
                            (h < 4 ? (h + 1) * e[d][pa][pb][h + 1] : MP(0));
      }
  }
  Triple ca[6], cb[6], hs[35];
  int na = 0, nb = 0, nhh = 0;
  for (int x = a.a.l; x >= 0; x--)
    for (int y = a.a.l - x; y >= 0; y--)
      ca[na++] = {x, y, a.a.l - x - y};
  for (int x = a.b.l; x >= 0; x--)
    for (int y = a.b.l - x; y >= 0; y--)
      cb[nb++] = {x, y, a.b.l - x - y};
  for (int d = 0; d <= a.a.l + a.b.l; d++)
    for (int x = 0; x <= d; x++)
      for (int y = 0; y <= d - x; y++)
        hs[nhh++] = {x, y, d - x - y};
  auto &cart = scratch.cart;
  for (int h = 0; h < nhh; h++) {
    for (int b = 0; b < nb; b++)
      for (int c = 0; c < na; c++) {
        MP z = 1;
        for (int d = 0; d < 3; d++)
          z *= e[d][ca[c][d]][cb[b][d]][hs[h][d]];
        cart[c + na * b] = z;
      }
    if (pure && (a.a.l == 2 || a.b.l == 2)) {
      MP rt = sqrt(MP(3));
      auto transform = [&](const MP *x, int stride, MP *y, int output_stride) {
        y[0] = rt * x[stride];
        y[output_stride] = rt * x[4 * stride];
        y[2 * output_stride] = x[5 * stride] - (x[0] + x[3 * stride]) / 2;
        y[3 * output_stride] = rt * x[2 * stride];
        y[4 * output_stride] = rt * (x[0] - x[3 * stride]) / 2;
      };
      auto &tmp = scratch.transformed;
      int na_view = na;
      if (a.a.l == 2) {
        for (int b = 0; b < nb; b++)
          transform(cart + 6 * b, 1, tmp + 5 * b, 1);
        na_view = 5;
      } else
        for (int i = 0; i < na * nb; i++)
          tmp[i] = cart[i];
      if (a.b.l == 2) {
        for (int c = 0; c < na_view; c++)
          transform(tmp + c, na_view, v.E + h * na_view * 5 + c, na_view);
      } else
        for (int i = 0; i < na_view * nb; i++)
          v.E[h * na_view * nb + i] = tmp[i];
    } else
      for (int c = 0; c < na * nb; c++)
        v.E[h * na * nb + c] = cart[c];
  }
  v.weight = coefficient(a.a, i, channel % a.a.c) * coefficient(a.b, j, channel / a.a.c);
}

void compute_sensitive(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
                       libgto_workspace &w, double *out, const Layout &l) {
  require(l.L <= 8, LIBGTO_UNSUPPORTED);
  require(bool(w.precise), LIBGTO_RESOURCE_LIMIT);
  auto &scratch = *w.precise;
  ++w.precision.wide_calls;
  ++w.sensitive_calls;
  w.stats = {};
  double clock = o.profile ? now() : 0;
  Real *contracted = w.arena.data() + l.r + l.m + l.u + l.g + l.tmp;
  Real *stage = contracted + l.contracted;
  int off[9]{};
  for (int m = 1; m <= l.L; m++)
    off[m] = off[m - 1] + nh(l.L - m + 1);
  size_t P = a.a.k * a.b.k, Q = b.a.k * b.b.k;
  Triple h[165];
  int ih = 0;
  for (int d = 0; d <= l.L; d++)
    for (int x = 0; x <= d; x++)
      for (int y = 0; y <= d - x; y++)
        h[ih++] = {x, y, d - x - y};
  MP constant = 2 * mp_pi() * mp_pi() * sqrt(mp_pi()), anchor[3];
  for (int d = 0; d < 3; d++)
    anchor[d] = MP(a.a.r[d]) - MP(b.a.r[d]);
  const MP absolute_tolerance = MP(1) / MP(10000000000ULL),
           relative_tolerance = MP(1) / MP(100000000000ULL);
  for (size_t cb = 0; cb < l.ccd; cb++)
    for (size_t ca = 0; ca < l.cab; ca++) {
      auto &sum = scratch.sum;
      auto &mass = scratch.mass;
      auto &x = scratch.x;
      auto &y = scratch.y;
      for (size_t i = 0; i < l.g; i++) {
        sum[i] = 0;
        mass[i] = 0;
      }
      for (size_t q = 0; q < Q; q++) {
        build_pair(b, q, cb, o.representation == LIBGTO_PURE, y, scratch);
        for (size_t p = 0; p < P; p++) {
          build_pair(a, p, ca, o.representation == LIBGTO_PURE, x, scratch);
          MP rho = x.p / (x.p + y.p) * y.p, X[3], T = 0;
          for (int d = 0; d < 3; d++) {
            X[d] = anchor[d] + x.P[d] - y.P[d];
            T += rho * X[d] * X[d];
          }
          auto &F = scratch.F;
          auto &R = scratch.R;
          MP factor = 1;
          fundamental(l.L, T, F);
          for (int m = 0; m <= l.L; m++) {
            R[off[m]] = factor * F[m];
            factor *= -2 * rho;
          }
          for (int m = l.L - 1; m >= 0; m--)
            for (int i = 1; i < nh(l.L - m); i++) {
              auto t = h[i];
              int axis = t[0] ? 0 : t[1] ? 1 : 2, degree = t[axis];
              --t[axis];
              MP value = X[axis] * R[off[m + 1] + hi(t[0], t[1], t[2])];
              if (degree > 1) {
                --t[axis];
                value += (degree - 1) * R[off[m + 1] + hi(t[0], t[1], t[2])];
              }
              R[off[m] + i] = value;
            }
          MP prefactor = constant / (x.p * y.p * sqrt(x.p + y.p)) * x.K * y.K;
          auto &U = scratch.U;
          auto &G = scratch.G;
          for (size_t i = 0; i < l.u; i++)
            U[i] = 0;
          for (size_t i = 0; i < l.g; i++)
            G[i] = 0;
          for (int i = 0; i < l.hab; i++)
            for (int j = 0; j < l.hcd; j++) {
              auto u = h[i], v = h[j];
              MP m = prefactor * R[hi(u[0] + v[0], u[1] + v[1], u[2] + v[2])];
              if ((v[0] + v[1] + v[2]) & 1)
                m = -m;
              for (int c = 0; c < l.ncd; c++)
                U[i * l.ncd + c] += m * y.E[j * l.ncd + c];
            }
          for (int i = 0; i < l.hab; i++)
            for (int c = 0; c < l.ncd; c++)
              for (int j = 0; j < l.nab; j++)
                G[j + l.nab * c] += x.E[i * l.nab + j] * U[i * l.ncd + c];
          MP weight = x.weight * y.weight;
          for (size_t i = 0; i < l.g; i++) {
            MP term = weight * G[i];
            sum[i] += term;
            mass[i] += abs(term);
          }
          ++w.stats.primitive_quartets;
        }
      }
      for (size_t i = 0; i < l.g; i++) {
        // The empirical residual-precision guard reserves 48 bits of the working
        // mantissa. It is not an error certificate for arbitrary conditioning.
        MP bound = mass[i] * MP(P) * MP(Q) * ldexp(MP(1), -208),
           tol = absolute_tolerance + relative_tolerance * abs(sum[i]);
        require(bound < tol / 16, LIBGTO_NUMERICAL_FAILURE);
        contracted[i + l.g * (ca + l.cab * cb)] = sum[i].convert_to<Real>();
      }
    }
  if (o.profile)
    w.stats.recurrence_seconds = now() - clock;
  require(store_md_eri_output(a, b, o, w, contracted, stage, out, l), LIBGTO_NUMERICAL_FAILURE);
}
}
