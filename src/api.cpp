#include "internal.hpp"
extern "C" {
const char *libgto_status_string(libgto_status s) {
  switch (s) {
  case LIBGTO_SUCCESS:
    return "success";
  case LIBGTO_INVALID_ARGUMENT:
    return "invalid argument or size overflow";
  case LIBGTO_UNSUPPORTED:
    return "unsupported angular momentum or configuration";
  case LIBGTO_ALLOCATION_FAILED:
    return "allocation failed";
  case LIBGTO_BUFFER_TOO_SMALL:
    return "output buffer too small";
  case LIBGTO_RESOURCE_LIMIT:
    return "workspace or thread resource limit";
  default:
    return "nonfinite numerical result or internal error";
  }
}

libgto_eri_options libgto_default_eri_options(void) {
  return {LIBGTO_CARTESIAN, LIBGTO_CONTRACT_AUTO, 0};
}

libgto_status libgto_pair_create(const libgto_shell *a, const libgto_shell *b, libgto_pair **out) {
  if (out)
    *out = nullptr;
  return gto::protect([&] {
    gto::require(a && b && out);
    *out = gto::make_pair(*a, *b).release();
  });
}

void libgto_pair_destroy(libgto_pair *p) {
  delete p;
}

size_t libgto_pair_bytes(const libgto_pair *p) {
  if (!p)
    return 0;
  return libgto_pair_bytes(p->optimized.get()) +
         (p->norm_cart.capacity() + p->norm_pure.capacity()) * sizeof(double) +
         (p->E_mask.capacity() + p->pure_mask.capacity()) * sizeof(uint64_t) + sizeof(*p) +
         p->primitives.capacity() * sizeof(gto::Primitive) +
         (p->E.capacity() + p->pure_E.capacity() + p->weights.capacity() + p->a.a.capacity() +
          p->b.a.capacity() + p->a.weights.capacity() + p->b.weights.capacity()) *
             sizeof(gto::Real) +
         (p->E64.capacity() + p->pure_E64.capacity() + p->weights64.capacity() +
          p->a.raw_coefficients.capacity() + p->b.raw_coefficients.capacity()) *
             sizeof(double);
}

libgto_status libgto_eri_requirements(const libgto_pair *a, const libgto_pair *b,
                                      const libgto_eri_options *o, libgto_requirements *out) {
  return gto::protect([&] {
    gto::require(a && b && o && out);
    auto l = gto::layout(*a, *b, *o);
    *out = l.public_size;
    if (!gto::high_angular(*a, *b) && gto::needs_sensitive(*a, *b, l, LIBGTO_PRECISION_AUTO))
      out->workspace_bytes = gto::add(out->workspace_bytes, gto::sensitive_workspace_bytes());
  });
}

libgto_status libgto_workspace_create(libgto_workspace **w) {
  if (w)
    *w = nullptr;
  return gto::protect([&] {
    gto::require(w);
    *w = new libgto_workspace;
  });
}

void libgto_workspace_destroy(libgto_workspace *w) {
  delete w;
}

libgto_status libgto_workspace_reserve(libgto_workspace *w, const libgto_pair *a,
                                       const libgto_pair *b, const libgto_eri_options *o) {
  return gto::protect([&] {
    gto::require(w && a && b && o);
    auto l = gto::layout(*a, *b, *o, w->policy);
    gto::reserve(*w, l.total, l.borrowed, l.fast, gto::needs_sensitive(*a, *b, l, w->policy));
  });
}

size_t libgto_workspace_bytes(const libgto_workspace *w) {
  return w ? gto::high_scratch_bytes(*w) + w->arena.capacity() * sizeof(gto::Real) +
                 (w->borrowed.capacity() + w->fast.capacity()) * sizeof(double) +
                 (w->precise ? gto::sensitive_workspace_bytes() : 0)
           : 0;
}

libgto_status libgto_workspace_set_precision(libgto_workspace *w, libgto_precision_policy p) {
  if (!w || (p != LIBGTO_PRECISION_AUTO && p != LIBGTO_PRECISION_REFERENCE))
    return LIBGTO_INVALID_ARGUMENT;
  w->policy = p;
  return LIBGTO_SUCCESS;
}

libgto_status libgto_batch_set_precision(libgto_batch *b, libgto_precision_policy p) {
  if (!b || (p != LIBGTO_PRECISION_AUTO && p != LIBGTO_PRECISION_REFERENCE))
    return LIBGTO_INVALID_ARGUMENT;
  for (auto &w : b->work)
    w.policy = p;
  return LIBGTO_SUCCESS;
}

libgto_status libgto_workspace_precision_statistics(const libgto_workspace *w,
                                                    libgto_precision_statistics *out) {
  if (!w || !out)
    return LIBGTO_INVALID_ARGUMENT;
  *out = w->precision;
  return LIBGTO_SUCCESS;
}

libgto_status libgto_batch_precision_statistics(const libgto_batch *b,
                                                libgto_precision_statistics *out) {
  if (!b || !out)
    return LIBGTO_INVALID_ARGUMENT;
  *out = {};
  for (const auto &w : b->work) {
    out->wide_calls += w.precision.wide_calls;
    out->fp64_calls += w.precision.fp64_calls;
    out->fp64_fallbacks += w.precision.fp64_fallbacks;
    out->fp64_primitive_quartets += w.precision.fp64_primitive_quartets;
  }
  return LIBGTO_SUCCESS;
}

libgto_status libgto_workspace_reconstruction_count(const libgto_workspace *w, uint64_t *out) {
  if (!w || !out)
    return LIBGTO_INVALID_ARGUMENT;
  *out = w->sensitive_calls;
  return LIBGTO_SUCCESS;
}

libgto_status libgto_batch_reconstruction_count(const libgto_batch *b, uint64_t *out) {
  if (!b || !out)
    return LIBGTO_INVALID_ARGUMENT;
  *out = 0;
  for (const auto &w : b->work)
    *out += w.sensitive_calls;
  return LIBGTO_SUCCESS;
}

libgto_status libgto_workspace_statistics(const libgto_workspace *w, libgto_statistics *out) {
  if (!w || !out)
    return LIBGTO_INVALID_ARGUMENT;
  *out = w->stats;
  return LIBGTO_SUCCESS;
}

libgto_status libgto_eri(const libgto_pair *a, const libgto_pair *b, const libgto_eri_options *o,
                         libgto_workspace *w, double *out, size_t n) {
  return gto::protect([&] {
    gto::require(a && b && o && w && out);
    auto l = gto::layout(*a, *b, *o, w->policy);
    gto::require(n >= l.public_size.elements, LIBGTO_BUFFER_TOO_SMALL);
    gto::reserve(*w, l.total, l.borrowed, l.fast, gto::needs_sensitive(*a, *b, l, w->policy));
    gto::compute(*a, *b, *o, *w, out, l);
  });
}

libgto_status libgto_eri_view(const libgto_pair *a, const libgto_pair *b,
                              const libgto_eri_options *o, libgto_workspace *w, const double **out,
                              size_t *n) {
  if (out)
    *out = nullptr;
  if (n)
    *n = 0;
  return gto::protect([&] {
    gto::require(a && b && o && w && out && n);
    auto l = gto::layout(*a, *b, *o, w->policy);
    gto::reserve(*w, l.total, l.borrowed, l.fast, gto::needs_sensitive(*a, *b, l, w->policy));
    double *p = w->borrowed.data();
    gto::compute(*a, *b, *o, *w, p, l);
    *out = p;
    *n = l.borrowed;
  });
}

libgto_status libgto_boys(unsigned m, double t, double *out, size_t n) {
  return gto::protect([&] {
    gto::require(m <= 8 && std::isfinite(t) && t >= 0 && out);
    gto::require(n > m, LIBGTO_BUFFER_TOO_SMALL);
    gto::boys(m, t, out);
  });
}

libgto_batch_options libgto_default_batch_options(void) {
  return {1, 64, 1024 * 1024, 64 * 1024 * 1024};
}

libgto_status libgto_batch_create(const libgto_batch_options *o, libgto_batch **out) {
  if (out)
    *out = nullptr;
  return gto::protect([&] {
    gto::require(o && out && o->threads > 0 && o->max_tile_tasks && o->max_tile_output_bytes &&
                 o->max_workspace_bytes);
#ifndef _OPENMP
    gto::require(o->threads == 1, LIBGTO_UNSUPPORTED);
#endif
    auto b = std::make_unique<libgto_batch>();
    b->options = *o;
    b->work.resize(o->threads);
    for (auto &w : b->work)
      w.capacity_limit = o->max_workspace_bytes;
    *out = b.release();
  });
}

void libgto_batch_destroy(libgto_batch *b) {
  delete b;
}

size_t libgto_batch_workspace_bytes(const libgto_batch *b) {
  size_t n = 0;
  if (b)
    for (const auto &w : b->work)
      n += libgto_workspace_bytes(&w);
  return n;
}

libgto_status libgto_batch_compute(libgto_batch *b, const libgto_eri_task *t, size_t n) {
  return gto::protect([&] { gto::execute_batch(b, t, n); });
}

size_t libgto_batch_binding_bytes(const libgto_batch *b) {
  return b ? (sizeof(b->plans) + sizeof(b->plan_index)) +
                 b->bindings.capacity() * sizeof(gto::BlockBinding) +
                 b->ranges.capacity() * sizeof(gto::OutputRange) +
                 b->parallel_layouts.capacity() * sizeof(gto::Layout)
           : 0;
}

libgto_status libgto_batch_set_binding_limits(libgto_batch *b, size_t count, size_t bytes) {
  return gto::protect([&] {
    gto::require(b);
    gto::require(bytes >= (sizeof(b->plans) + sizeof(b->plan_index)), LIBGTO_RESOURCE_LIMIT);
    gto::require(gto::add((sizeof(b->plans) + sizeof(b->plan_index)),
                          gto::mul(count, sizeof(gto::BlockBinding) + sizeof(gto::OutputRange))) <=
                     bytes,
                 LIBGTO_RESOURCE_LIMIT);
    if (libgto_batch_binding_bytes(b) > bytes) {
      std::vector<gto::Layout>().swap(b->parallel_layouts);
      std::vector<gto::BlockBinding>().swap(b->bindings);
      std::vector<gto::OutputRange>().swap(b->ranges);
    }
    b->binding_limit = count;
    b->binding_byte_limit = bytes;
  });
}

libgto_status libgto_batch_binding_statistics(const libgto_batch *b,
                                              libgto_binding_statistics *out) {
  if (!b || !out)
    return LIBGTO_INVALID_ARGUMENT;
  *out = b->binding_stats;
  return LIBGTO_SUCCESS;
}
}
