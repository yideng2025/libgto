#include "internal.hpp"
namespace gto {
static Layout make_layout(const QuartetShape &s, const libgto_eri_options &o, bool projected) {
  require(o.representation == LIBGTO_CARTESIAN || o.representation == LIBGTO_PURE);
  require(o.contraction_order >= LIBGTO_CONTRACT_AUTO && o.contraction_order <= LIBGTO_CD_FIRST);
  Layout l{};
  l.L = s.l[0] + s.l[1] + s.l[2] + s.l[3];
  l.hab = nh(s.l[0] + s.l[1]);
  l.hcd = nh(s.l[2] + s.l[3]);
  l.nab = projected ? (2 * s.l[0] + 1) * (2 * s.l[1] + 1) : (ncart(s.l[0]) * ncart(s.l[1]));
  l.ncd = projected ? (2 * s.l[2] + 1) * (2 * s.l[3] + 1) : (ncart(s.l[2]) * ncart(s.l[3]));
  l.cab = mul(s.c[0], s.c[1]);
  l.ccd = mul(s.c[2], s.c[3]);
  long double P = mul(s.k[0], s.k[1]), Q = mul(s.k[2], s.k[3]), I = l.cab, O = l.ccd;
  l.abfirst =
      o.contraction_order == LIBGTO_AB_FIRST || (o.contraction_order == LIBGTO_CONTRACT_AUTO &&
                                                 (P * Q * I + Q * I * O <= P * Q * O + P * I * O));
  size_t n = 1;
  for (int i = 0; i < 4; i++) {
    l.public_size.dimensions[i] =
        mul(o.representation == LIBGTO_PURE ? 2 * s.l[i] + 1 : ncart(s.l[i]), s.c[i]);
    n = mul(n, l.public_size.dimensions[i]);
  }
  l.public_size.elements = n;
  l.public_size.output_bytes = mul(n, sizeof(double));
  constexpr size_t r_counts[9] = {1, 5, 15, 35, 70, 126, 210, 330, 495};
  l.r = l.L <= 8 ? r_counts[l.L] : size_t((l.L + 1) * (l.L + 2) * (l.L + 3) * (l.L + 4) / 24);
  l.m = mul(l.hab, l.hcd);
  l.u = mul(l.hab, l.ncd);
  l.g = mul(l.nab, l.ncd);
  l.tmp = (l.cab == 1 && l.ccd == 1) ? 0 : mul(l.g, l.abfirst ? l.cab : l.ccd);
  l.contracted = mul(mul(l.g, l.cab), l.ccd);
  l.stage = l.contracted;
  l.borrowed = n;
  l.total = 0;
  for (size_t x : {l.r, l.m, l.u, l.g, l.tmp, l.contracted, l.stage, l.stage})
    l.total = add(l.total, x);
  l.fast = add(add(mul(2, l.u), mul(3, l.g)), mul(2, add(l.tmp, l.contracted)));
  l.fast = add(add(l.fast, l.m), mul(2, nh(l.L)));
  if (l.hab == 35 && l.hcd == 35 && l.nab >= 25 && l.ncd >= 25)
    l.fast = add(l.fast, add(l.m, add(mul(l.hab, l.nab), mul(l.hcd, l.ncd))));
  l.public_size.workspace_bytes =
      add(mul(l.total, sizeof(Real)), mul(add(l.borrowed, l.fast), sizeof(double)));
  return l;
}

void require_md_angular_momentum(int l) {
  require(l >= 0 && l <= 4, LIBGTO_UNSUPPORTED);
}

QuartetShape quartet_shape(const libgto_pair &a, const libgto_pair &b) {
  const Shell *ss[4] = {&a.a, &a.b, &b.a, &b.b};
  QuartetShape s;
  for (int i = 0; i < 4; i++) {
    s.l[i] = ss[i]->l;
    s.k[i] = ss[i]->k;
    s.c[i] = ss[i]->c;
    s.normalization[i] = ss[i]->norm;
  }
  return s;
}

MathRequest eri_request(const libgto_eri_options &o, libgto_precision_policy p) {
  MathRequest r;
  r.representation = o.representation;
  r.precision = p;
  r.contraction_order = o.contraction_order;
  return r;
}

static Layout storage_for(const QuartetShape &s, const libgto_eri_options &o) {
  auto l = make_layout(s, o, o.representation == LIBGTO_PURE);
  if (o.representation == LIBGTO_PURE && *std::max_element(s.l.begin(), s.l.end()) <= 2) {
    size_t g = mul(ncart(s.l[0]) * ncart(s.l[1]), ncart(s.l[2]) * ncart(s.l[3])),
           u = mul(l.hab, ncart(s.l[2]) * ncart(s.l[3])),
           tmp = (l.cab == 1 && l.ccd == 1) ? 0 : mul(g, l.abfirst ? l.cab : l.ccd),
           out = mul(mul(g, l.cab), l.ccd), total = 0;
    for (size_t n : {l.r, l.m, u, g, tmp, mul(3, out)})
      total = add(total, n);
    l.total = std::max(l.total, total);
  }
  if (*std::max_element(s.l.begin(), s.l.end()) > 2)
    l.fast = add(l.m, add(l.u, l.g));
  l.public_size.workspace_bytes =
      add(mul(l.total, sizeof(Real)), mul(add(l.borrowed, l.fast), sizeof(double)));
  if (*std::max_element(s.l.begin(), s.l.end()) > 2)
    l.public_size.workspace_bytes = add(l.public_size.workspace_bytes, high_shape_mp_bytes(s, l));
  return l;
}

FamilyPlan resolve_plan(const MathRequest &r, const QuartetShape &s) {
  require(r.op == OperatorKind::Coulomb && r.rank == 4 && r.derivative_order == 0,
          LIBGTO_UNSUPPORTED);
  require(r.precision == LIBGTO_PRECISION_AUTO || r.precision == LIBGTO_PRECISION_REFERENCE,
          LIBGTO_UNSUPPORTED);
  for (int i = 0; i < 4; i++) {
    require_md_angular_momentum(s.l[i]);
    require(s.k[i] && s.c[i]);
    require(s.normalization[i] == LIBGTO_UNNORMALIZED ||
            s.normalization[i] == LIBGTO_NORMALIZED_PRIMITIVES);
  }
  return {r, s, storage_for(s, {r.representation, r.contraction_order, 0})};
}

Layout layout_reference(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o) {
  return make_layout(quartet_shape(a, b), o, false);
}

Layout layout(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
              libgto_precision_policy p) {
  (void)p;
  return storage_for(quartet_shape(a, b), o);
}

const FamilyPlan *cached_plan(libgto_batch &b, const libgto_eri_task &t,
                              libgto_precision_policy precision) {
  auto request = eri_request(t.options, precision);
  // Validate enum fields before compact-key packing; collisions never grant support.
  require(t.options.representation == LIBGTO_CARTESIAN || t.options.representation == LIBGTO_PURE);
  require(t.options.contraction_order >= LIBGTO_CONTRACT_AUTO &&
          t.options.contraction_order <= LIBGTO_CD_FIRST);
  PlanKey key;
  key.semantics = unsigned(request.representation) | (unsigned(precision) << 1) |
                  (unsigned(request.contraction_order) << 2) | (t.ab->shape_bits << 4) |
                  (t.cd->shape_bits << 12);
  for (int i = 0; i < 2; i++) {
    key.sizes[i] = t.ab->shape_sizes[i];
    key.sizes[i + 2] = t.cd->shape_sizes[i];
    key.sizes[i + 4] = t.ab->shape_sizes[i + 2];
    key.sizes[i + 6] = t.cd->shape_sizes[i + 2];
  }
  size_t hash =
      t.ab->shape_hash ^ (t.cd->shape_hash * size_t(0x9e3779b97f4a7c15ull)) ^ key.semantics;
  hash ^= hash >> 32;

  size_t bucket = hash % b.plan_index.size();
  while (b.plan_index[bucket]) {
    auto &e = b.plans[b.plan_index[bucket] - 1];
    if (e.key == key) {
      ++b.binding_stats.plan_hits;
      return &e.plan;
    }
    bucket = (bucket + 1) % b.plan_index.size();
  }
  // A full per-batch pool uses the compatible preflight path; never overwrite
  // a plan referenced by another binding in the same call.
  if (b.plan_count == b.plans.size())
    return nullptr;
  auto &e = b.plans[b.plan_count];
  e.key = key;
  e.plan = resolve_plan(request, quartet_shape(*t.ab, *t.cd));
  b.plan_index[bucket] = static_cast<unsigned short>(++b.plan_count);
  ++b.binding_stats.plan_builds;
  return &e.plan;
}
}
