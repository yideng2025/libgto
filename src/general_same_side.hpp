static void pack_same_pair(same_pair_moment::Packet &p, size_t k, const libgto_pair &a,
                           const libgto_pair &b, size_t ip, size_t iq, double budget) {
  bool flip = a.a.l + a.b.l == 0;
  const auto &pp = flip ? b : a;
  const auto &ss = flip ? a : b;
  size_t ipair = flip ? iq : ip, isp = flip ? ip : iq;
  const auto &x = a.primitives[ip];
  const auto &y = b.primitives[iq];
  const auto &z = pp.primitives[ipair];
  double ap = x.p64, bp = y.p64, sum = ap + bp, rho = ap / sum * bp, pa = z.p64,
         pb = ss.primitives[isp].p64, af = pa / sum, bf = pb / sum;
  bool single_d = pp.N == 6;
  p.valid[k] = x.low_normal && y.low_normal && ap >= 1e-8 && ap <= 1e8 && bp >= 1e-8 && bp <= 1e8 &&
               std::min(af, bf) >= 1e-12;
  p.T[k] = 0;
  const int diagonal[3] = {0, 3, 5};
  for (int d = 0; d < 3; d++) {
    double X = double((a.a.r[d] - b.a.r[d]) + (x.offset[d] - y.offset[d]));
    p.T[k] += rho * X * X;
    if (flip)
      X = -X;
    int other = (d + 1) % 3;
    if (single_d) {
      p.a[d][k] = double(z.p * pp.E[ipair * 60 + (3 - d) * 6 + diagonal[d]]);
      p.b[d][k] = 0;
    } else {
      p.a[d][k] = double(z.offset[d]);
      p.b[d][k] = double(2 * z.p * pp.E[ipair * 90 + (3 - other) * 9 + other + 3 * d]);
    }
    p.v[d][k] = -bf * X;
  }
  p.cov0[k] = 1 / (2 * pa);
  p.cov1[k] = -p.cov0[k] * bf;
  p.pref[k] = double(2 * std::pow(pi, 2.5L)) / (ap * bp * std::sqrt(sum)) * x.K64 * y.K64;
  p.valid[k] &= std::isnormal(sum) && std::isnormal(rho) && std::isnormal(p.pref[k]) &&
                std::isfinite(p.T[k]) && p.T[k] >= 0;
  p.budget[k] = budget;
}

static bool compute_same_pair_panel(const libgto_pair &a, const libgto_pair &b,
                                    const libgto_eri_options &o, libgto_workspace &w, double *out,
                                    const Layout &l) {
  if (l.L != 2 ||
      !((a.a.l + a.b.l == 2 && b.a.l + b.b.l == 0) || (b.a.l + b.b.l == 2 && a.a.l + a.b.l == 0)) ||
      ((a.N == 6 || b.N == 6) && o.representation != LIBGTO_PURE) || o.profile || !a.fp64_safe ||
      !b.fp64_safe || a.sensitive || b.sensitive)
    return false;
  size_t P = a.primitives.size(), Q = b.primitives.size(), total = mul(P, Q);
  bool flip = l.nab == 1;
  bool single_d = (flip ? b : a).N == 6;
  size_t components = single_d ? 5 : 9;

  static_assert(sizeof(same_pair_moment::Packet) <= 131072, "bounded numeric scratch");
  same_pair_moment::Packet packet;
  w.stats = {};
  Real *R = w.arena.data();
  Real *M = R + l.r;
  Real *U = M + l.m;
  Real *G = U + l.u;
  Real *tmp = G + l.g;
  Real *contracted = tmp + l.tmp;
  Real *stage = contracted + l.contracted;
  std::fill_n(contracted, l.contracted, 0);
  double mass = a.weight_mass * b.weight_mass, budget = std::isnormal(mass) ? 0.0625e-10 / mass : 0;
  bool direct = l.cab == 1 && l.ccd == 1, abfirst = direct ? true : l.abfirst, local_wide = false;
  size_t inner = abfirst ? P : Q, outer = abfirst ? Q : P, ni = abfirst ? l.cab : l.ccd,
         no = abfirst ? l.ccd : l.cab;
  const auto &wi = abfirst ? a.weights : b.weights;
  const auto &wo = abfirst ? b.weights : a.weights;
  for (size_t start = 0; start < total; start += 64) {
    size_t count = std::min<size_t>(64, total - start);
    for (size_t k = 0; k < count; k++) {
      size_t flat = start + k, p = flat % inner, q = flat / inner;
      pack_same_pair(packet, k, a, b, abfirst ? p : q, abfirst ? q : p, budget);
    }
    same_pair_moment::angular(packet, count, single_d);
    w.precision.fp64_primitive_quartets += count;
    for (size_t k = 0; k < count; k++) {
      size_t flat = start + k, p = flat % inner, q = flat / inner;
      if (!direct && p == 0)
        std::fill_n(tmp, l.tmp, 0);
      bool narrow = packet.valid[k];
      if (!narrow) {
        if (!local_wide) {
          ++w.precision.fp64_fallbacks;
          ++w.precision.wide_calls;
          local_wide = true;
        }
        primitive(a, b, abfirst ? p : q, abfirst ? q : p, l, R, M, U, G,
                  o.representation == LIBGTO_PURE);
        ++w.stats.primitive_quartets;
      }
      ++w.stats.primitive_quartets;
      if (direct) {
        Real weight = a.weights[p] * b.weights[q];
        for (size_t v = 0; v < components; v++)
          contracted[v] += weight * (narrow ? Real(packet.G[v][k]) : G[v]);
      } else {
        for (size_t c = 0; c < ni; c++) {
          Real weight = wi[p + inner * c];
          for (size_t v = 0; v < components; v++)
            tmp[v + components * c] += weight * (narrow ? Real(packet.G[v][k]) : G[v]);
        }
        if (p + 1 == inner)
          for (size_t d = 0; d < no; d++)
            for (size_t c = 0; c < ni; c++) {
              size_t slot = abfirst ? c + l.cab * d : d + l.cab * c;
              Real weight = wo[q + outer * d];
              for (size_t v = 0; v < components; v++)
                contracted[v + components * slot] += weight * tmp[v + components * c];
            }
      }
    }
  }
  return store_md_eri_output(a, b, o, w, contracted, stage, out, l);
}
