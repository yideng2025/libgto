

static bool single_p_panel_eligible(const libgto_pair &a, const libgto_pair &b,
                                    const libgto_eri_options &o, const Layout &l) {
  return l.L == 1 && !o.profile && a.fp64_safe && b.fp64_safe && !a.sensitive && !b.sensitive &&
         (l.cab > 1 || l.ccd > 1) && a.primitives.size() <= 64 && b.primitives.size() <= 64 &&
         a.primitives.size() * b.primitives.size() >= 8;
}
static bool contract_single_p_panel(const libgto_pair &a, const libgto_pair &b,
                                    const libgto_eri_options &o, libgto_workspace &w, double *out,
                                    const Layout &l) {
  {
    {}
  }
  size_t P = a.primitives.size(), Q = b.primitives.size(), total = P * Q;
  bool abfirst = l.abfirst;
  size_t inner = abfirst ? P : Q, outer = abfirst ? Q : P, ni = abfirst ? l.cab : l.ccd,
         no = abfirst ? l.ccd : l.cab;
  const auto &wi = abfirst ? a.weights64 : b.weights64;
  const auto &wo = abfirst ? b.weights : a.weights;
  Real *contracted = w.arena.data() + l.r + l.m + l.u + l.g + l.tmp;
  Real *stage = contracted + l.contracted;
  require(w.fast.size() >= l.tmp, LIBGTO_RESOURCE_LIMIT);
  double *tmp = w.fast.data();
  std::fill_n(contracted, l.contracted, Real(0));
  w.stats = {};
  for (const auto &x : a.primitives)
    if (!x.low_normal) {
      return false;
    }
  for (const auto &x : b.primitives)
    if (!x.low_normal) {
      return false;
    }
  double ca[64]{}, cb[64]{};
  for (size_t c = 0; c < l.cab; c++)
    for (size_t p = 0; p < P; p++)
      ca[p] = std::max(ca[p], std::abs(a.weights64[p + P * c]));
  for (size_t c = 0; c < l.ccd; c++)
    for (size_t q = 0; q < Q; q++)
      cb[q] = std::max(cb[q], std::abs(b.weights64[q + Q * c]));
  double mass = a.weight_mass * b.weight_mass, budget = std::isnormal(mass) ? 0.0625e-10 / mass : 0;
  Real magnitude = 0;
  single_p::Packet packet;
  for (size_t start = 0; start < total; start += 64) {
    size_t count = std::min<size_t>(64, total - start);
    for (size_t k = 0; k < count; k++) {
      size_t flat = start + k, p = flat % inner, q = flat / inner;
      pack_single_p(packet, k, a, b, abfirst ? p : q, abfirst ? q : p, budget);
    }
    single_p::angular(packet, count);
    w.precision.fp64_primitive_quartets += count;
    w.stats.primitive_quartets += count;
    for (size_t k = 0; k < count; k++) {
      size_t flat = start + k, p = flat % inner, q = flat / inner;
      if (!packet.valid[k]) {
        return false;
      }
      if (!p)
        std::fill_n(tmp, l.tmp, 0.);
      double g[3], largest = 0;
      for (int v = 0; v < 3; v++) {
        g[v] = packet.G[v][k];
        largest = std::max(largest, std::abs(g[v]));
      }
      magnitude += Real(ca[abfirst ? p : q]) * Real(cb[abfirst ? q : p]) * Real(largest);
      for (size_t c = 0; c < ni; c++)
        channel_accumulation::update(3, wi[p + inner * c], g, tmp + 3 * c);
      if (p + 1 == inner)
        for (size_t d = 0; d < no; d++)
          for (size_t c = 0; c < ni; c++) {
            size_t slot = abfirst ? c + l.cab * d : d + l.cab * c;
            channel_accumulation::outer(3, wo[q + outer * d], tmp + 3 * c, contracted + 3 * slot);
          }
    }
    Real error = channel_accumulation::bound(magnitude, inner, outer, total,
                                             abfirst ? b.weight_mass : a.weight_mass);
    if (!std::isfinite(error) || error > 0.0625e-10L) {
      return false;
    }
  }

  return store_md_eri_output(a, b, o, w, contracted, stage, out, l);
}
