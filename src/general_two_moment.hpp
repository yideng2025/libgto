static void pack_two_p_moment(two_p_moment::Packet &p, size_t k, const libgto_pair &a,
                              const libgto_pair &b, size_t ip, size_t iq, double budget) {
  const auto &x = a.primitives[ip];
  const auto &y = b.primitives[iq];
  p.valid[k] = x.low_normal && y.low_normal;
  double ap = x.p64, bp = y.p64, sum = ap + bp;
  p.rho[k] = ap / sum * bp;
  p.T[k] = 0;
  p.valid[k] &= ap >= 1e-8 && ap <= 1e8 && bp >= 1e-8 && bp <= 1e8 &&
                std::min(ap, bp) / std::max(ap, bp) >= 1e-12;
  for (int d = 0; d < 3; d++) {
    p.X[d][k] = double((a.a.r[d] - b.a.r[d]) + (x.offset[d] - y.offset[d]));
    p.T[k] += p.rho[k] * p.X[d][k] * p.X[d][k];
    p.a0[d][k] = a.E64[ip * 12 + d];
    p.a1[d][k] = a.E64[ip * 12 + (3 - d) * 3 + d];
    p.c0[d][k] = b.E64[iq * 12 + d];
    p.c1[d][k] = b.E64[iq * 12 + (3 - d) * 3 + d];
  }
  p.pref[k] = double(2 * std::pow(pi, 2.5L)) / (ap * bp * std::sqrt(sum)) * x.K64 * y.K64;
  p.valid[k] &= std::isnormal(sum) && std::isnormal(p.rho[k]) && std::isnormal(p.pref[k]) &&
                std::isfinite(p.T[k]) && p.T[k] >= 0;
  p.na[k] = a.norm_cart[ip * (a.N + 1)];
  p.nb[k] = b.norm_cart[iq * (b.N + 1)];
  p.budget[k] = budget;
}
static bool compute_two_p_moment_panel(const libgto_pair &a, const libgto_pair &b,
                                       const libgto_eri_options &o, libgto_workspace &w,
                                       double *out, const Layout &l) {
  if (l.L != 2 || a.a.l + a.b.l != 1 || b.a.l + b.b.l != 1 || o.profile || !a.fp64_safe ||
      !b.fp64_safe || a.sensitive || b.sensitive)
    return false;
  size_t P = a.primitives.size(), Q = b.primitives.size(), total = mul(P, Q);
  if (total < 8)
    return false;
  static_assert(sizeof(two_p_moment::Packet) <= 57344, "bounded numeric scratch");
  two_p_moment::Packet packet;
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
      pack_two_p_moment(packet, k, a, b, abfirst ? p : q, abfirst ? q : p, budget);
    }
    two_p_moment::angular(packet, count);
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
        for (size_t v = 0; v < 9; v++)
          contracted[v] += weight * (narrow ? Real(packet.G[v][k]) : G[v]);
      } else {
        for (size_t c = 0; c < ni; c++) {
          Real weight = wi[p + inner * c];
          for (size_t v = 0; v < 9; v++)
            tmp[v + 9 * c] += weight * (narrow ? Real(packet.G[v][k]) : G[v]);
        }
        if (p + 1 == inner)
          for (size_t d = 0; d < no; d++)
            for (size_t c = 0; c < ni; c++) {
              size_t slot = abfirst ? c + l.cab * d : d + l.cab * c;
              Real weight = wo[q + outer * d];
              for (size_t v = 0; v < 9; v++)
                contracted[v + 9 * slot] += weight * tmp[v + 9 * c];
            }
      }
    }
  }
  return store_md_eri_output(a, b, o, w, contracted, stage, out, l);
}
