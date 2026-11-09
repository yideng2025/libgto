// Bounded ssss primitive panels. Same scalar Boys/monitor and wide channel order.
namespace scalar_panel {
struct alignas(64) Packet {
  double ap[64], bp[64], sum[64], kx[64], ky[64], T[64], pref[64], G[64];
  bool valid[64];
};
static_assert(sizeof(Packet) <= 8192, "bounded ssss packet");
template <bool Vector> static void prefactors(Packet &p, size_t n) {
  const double C = double(2 * std::pow(gto::pi, 2.5L));
  size_t k = 0;
#if defined(__AVX2__)
  if constexpr (Vector) {
    const auto c = _mm256_set1_pd(C);
    for (; k + 4 <= n; k += 4) {
      auto ap = _mm256_load_pd(p.ap + k), bp = _mm256_load_pd(p.bp + k),
           sum = _mm256_load_pd(p.sum + k);
      auto v = _mm256_div_pd(c, _mm256_mul_pd(_mm256_mul_pd(ap, bp), _mm256_sqrt_pd(sum)));
      v = _mm256_mul_pd(_mm256_mul_pd(v, _mm256_load_pd(p.kx + k)), _mm256_load_pd(p.ky + k));
      _mm256_store_pd(p.pref + k, v);
    }
  }
#endif
  for (; k < n; k++)
    p.pref[k] = C / (p.ap[k] * p.bp[k] * std::sqrt(p.sum[k])) * p.kx[k] * p.ky[k];
}
}
template <bool Vector>
static bool compute_scalar_panel(const libgto_pair &a, const libgto_pair &b,
                                 const libgto_eri_options &o, libgto_workspace &w, double *out,
                                 const Layout &l) {
  if (l.L || o.profile)
    return false;
  const LowContext ctx = low_context(a, b, o.representation == LIBGTO_PURE);
  size_t P = a.primitives.size(), Q = b.primitives.size(), total = mul(P, Q);
  if (total < 8)
    return false;
  bool direct = l.cab == 1 && l.ccd == 1, abfirst = direct ? true : l.abfirst;
  size_t inner = abfirst ? P : Q, ni = abfirst ? l.cab : l.ccd, no = abfirst ? l.ccd : l.cab;
  const auto &wi = abfirst ? a.weights : b.weights;
  const auto &wo = abfirst ? b.weights : a.weights;
  Real *R = w.arena.data(), *M = R + l.r, *U = M + l.m, *G = U + l.u, *tmp = G + l.g,
       *contracted = tmp + l.tmp, *stage = contracted + l.contracted;
  std::fill_n(contracted, l.contracted, Real(0));
  w.stats = {};
  bool local_wide = false;
  double mass = a.weight_mass * b.weight_mass, budget = std::isnormal(mass) ? 0.0625e-10 / mass : 0;
  scalar_panel::Packet z;
  for (size_t start = 0; start < total; start += 64) {
    size_t count = std::min<size_t>(64, total - start), p = start % inner, q = start / inner;
    for (size_t k = 0; k < count; k++) {
      size_t ip = abfirst ? p : q, iq = abfirst ? q : p;
      const auto &x = a.primitives[ip];
      const auto &y = b.primitives[iq];
      z.valid[k] = x.low_normal && y.low_normal;
      double ap = x.p64, bp = y.p64, sum = 0, rho = 0, T = 0;
      if (z.valid[k]) {
        sum = ap + bp;
        z.valid[k] = std::isnormal(sum);
        if (z.valid[k]) {
          rho = ap / sum * bp;
          z.valid[k] = std::isnormal(rho);
        }
      }
      if (z.valid[k]) {
        for (int d = 0; d < 3; d++) {
          double X = double(ctx.delta[d] + (x.offset[d] - y.offset[d]));
          T += rho * X * X;
        }
        z.valid[k] = std::isfinite(T) && T >= 0;
      }
      z.ap[k] = z.valid[k] ? ap : 1;
      z.bp[k] = z.valid[k] ? bp : 1;
      z.sum[k] = z.valid[k] ? sum : 2;
      z.kx[k] = z.valid[k] ? x.K64 : 1;
      z.ky[k] = z.valid[k] ? y.K64 : 1;
      z.T[k] = T;
      if (++p == inner) {
        p = 0;
        ++q;
      }
    }
    scalar_panel::prefactors<Vector>(z, count);
    p = start % inner;
    q = start / inner;
    for (size_t k = 0; k < count; k++) {
      size_t ip = abfirst ? p : q, iq = abfirst ? q : p;
      bool narrow = z.valid[k] && std::isnormal(z.pref[k]);
      double f = 0;
      if (narrow) {
        low_boys_fixed<0>(z.T[k], &f);
        double value = z.pref[k] * f;
        constexpr double eps = std::numeric_limits<double>::epsilon();
        double e = std::abs(f) * eps * (128 * (1 + std::min(z.T[k], .5)));
        double maxerror = std::abs(z.pref[k]) * e + 48 * eps * std::abs(value);
        double bound = (maxerror + 18 * eps * std::abs(value)) * ctx.an[ip * 2] * ctx.bn[iq * 2];
        bound = bound * (1 + 32 * eps) + std::numeric_limits<double>::min();
        narrow = !(value != 0 && !std::isnormal(value)) && std::isfinite(bound) && bound <= budget;
        z.G[k] = value;
      }
      z.valid[k] = narrow;
      if (++p == inner) {
        p = 0;
        ++q;
      }
    }
    w.precision.fp64_primitive_quartets += count;
    p = start % inner;
    q = start / inner;
    for (size_t k = 0; k < count; k++) {
      size_t ip = abfirst ? p : q, iq = abfirst ? q : p;
      if (!direct && p == 0)
        std::fill_n(tmp, l.tmp, Real(0));
      bool narrow = z.valid[k];
      if (!narrow) {
        bool scaled = (6 & 2) && compute_scaled_low<false>(ctx, ip, iq, G, budget);
        if (!scaled) {
          if (!local_wide) {
            ++w.precision.fp64_fallbacks;
            ++w.precision.wide_calls;
            local_wide = true;
          }
          primitive(a, b, ip, iq, l, R, M, U, G, o.representation == LIBGTO_PURE);
          ++w.stats.primitive_quartets;
        }
      }
      ++w.stats.primitive_quartets;
      if (direct) {
        Real weight = a.weights[ip] * b.weights[iq];
        contracted[0] += weight * (narrow ? Real(z.G[k]) : G[0]);
      } else {
        for (size_t c = 0; c < ni; c++)
          tmp[c] += wi[p + inner * c] * (narrow ? Real(z.G[k]) : G[0]);
        if (p + 1 == inner)
          for (size_t d = 0; d < no; d++)
            for (size_t c = 0; c < ni; c++) {
              size_t slot = abfirst ? c + l.cab * d : d + l.cab * c;
              contracted[slot] += wo[q + (abfirst ? Q : P) * d] * tmp[c];
            }
      }
      if (++p == inner) {
        p = 0;
        ++q;
      }
    }
  }
  return store_md_eri_output(a, b, o, w, contracted, stage, out, l);
}
