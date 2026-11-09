

static void pack_quadratic_linear(quadratic_linear::Packet &p, size_t k, const libgto_pair &a,
                                  const libgto_pair &b, size_t ip, size_t iq, double budget) {
  bool flip = a.a.l + a.b.l == 1;
  const auto &dp = flip ? b : a;
  const auto &pp = flip ? a : b;
  size_t id = flip ? iq : ip, ir = flip ? ip : iq;
  const auto &x = a.primitives[ip];
  const auto &y = b.primitives[iq];
  const auto &d = dp.primitives[id];
  double ap = x.p64, bp = y.p64, sum = ap + bp, rho = ap / sum * bp, pd = d.p64,
         pr = pp.primitives[ir].p64;
  p.valid[k] = x.low_normal && y.low_normal && pd >= 1e-8 && pd <= 1e8 && pr >= 1e-8 && pr <= 1e8 &&
               std::min(pd, pr) / sum >= 1e-12;
  p.T[k] = 0;
  const int diagonal[3] = {0, 3, 5};
  for (int axis = 0; axis < 3; axis++) {
    double X = double((a.a.r[axis] - b.a.r[axis]) + (x.offset[axis] - y.offset[axis]));
    p.T[k] += rho * X * X;
    if (flip)
      X = -X;
    // E_(e_axis),axis^2 = mean_axis/pd, no subtraction of large product offsets.
    p.a[axis][k] = double(d.p * dp.E[id * 60 + (3 - axis) * 6 + diagonal[axis]]);
    p.c[axis][k] = pp.E64[ir * 12 + axis];
    p.v[axis][k] = -pr / sum * X;
    p.z[axis][k] = pd / sum * X;
  }
  p.cross[k] = 1 / (2 * sum);
  p.pref[k] = double(2 * std::pow(pi, 2.5L)) / (ap * bp * std::sqrt(sum)) * x.K64 * y.K64;
  p.valid[k] &= std::isnormal(sum) && std::isnormal(rho) && std::isnormal(p.pref[k]) &&
                std::isfinite(p.T[k]) && p.T[k] >= 0;
  p.budget[k] = budget;
}

static bool compute_quadratic_linear_panel(const libgto_pair &a, const libgto_pair &b,
                                           const libgto_eri_options &o, libgto_workspace &w,
                                           double *out, const Layout &l) {
  if (l.L != 3 || o.representation != LIBGTO_PURE ||
      !((a.a.l + a.b.l == 2 && (a.a.l == 2 || a.b.l == 2) && b.a.l + b.b.l == 1) ||
        (b.a.l + b.b.l == 2 && (b.a.l == 2 || b.b.l == 2) && a.a.l + a.b.l == 1)) ||
      o.profile || !a.fp64_safe || !b.fp64_safe || a.sensitive || b.sensitive)
    return false;
  size_t P = a.primitives.size(), Q = b.primitives.size(), total = mul(P, Q);
  bool flip = l.nab == 3;

  static_assert(sizeof(quadratic_linear::Packet) <= 131072, "bounded numeric scratch");
  quadratic_linear::Packet packet;
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
      pack_quadratic_linear(packet, k, a, b, abfirst ? p : q, abfirst ? q : p, budget);
    }
    quadratic_linear::angular(packet, count);
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
        for (size_t v = 0; v < 15; v++)
          contracted[v] +=
              weight * (narrow ? Real(packet.G[flip ? (v / 3) + 5 * (v % 3) : v][k]) : G[v]);
      } else {
        for (size_t c = 0; c < ni; c++) {
          Real weight = wi[p + inner * c];
          for (size_t v = 0; v < 15; v++)
            tmp[v + 15 * c] +=
                weight * (narrow ? Real(packet.G[flip ? (v / 3) + 5 * (v % 3) : v][k]) : G[v]);
        }
        if (p + 1 == inner)
          for (size_t d = 0; d < no; d++)
            for (size_t c = 0; c < ni; c++) {
              size_t slot = abfirst ? c + l.cab * d : d + l.cab * c;
              Real weight = wo[q + outer * d];
              for (size_t v = 0; v < 15; v++)
                contracted[v + 15 * slot] += weight * tmp[v + 15 * c];
            }
      }
    }
  }
  return store_md_eri_output(a, b, o, w, contracted, stage, out, l);
}
