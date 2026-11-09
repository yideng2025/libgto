#include "angular_products.hpp"
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define GTO_AVX2 1
#define TARGET __attribute__((target("avx2,fma")))
#endif
namespace gto {
#if defined(GTO_AVX2)
// Four rows by eight contiguous columns: eight accumulators, two right vectors,
// one broadcast. FMA is explicit only inside this ISA-dispatched microkernel.
// There is no input-dependent reassociation and no fast-math policy change.
template <int Rows, int Inner, int Cols, bool TransposeLeft>
TARGET static void product_avx2(const double *left, const double *right, double *out) {
  constexpr int RI = Rows / 4 * 4, CJ = Cols / 8 * 8;
  for (int i = 0; i < RI; i += 4) {
    for (int j = 0; j < CJ; j += 8) {
      __m256d s00 = _mm256_setzero_pd(), s01 = s00, s10 = s00, s11 = s00, s20 = s00, s21 = s00,
              s30 = s00, s31 = s00;
      for (int k = 0; k < Inner; k++) {
        __m256d b0 = _mm256_loadu_pd(right + k * Cols + j),
                b1 = _mm256_loadu_pd(right + k * Cols + j + 4);
        __m256d a = _mm256_broadcast_sd(left + (TransposeLeft ? k * Rows + i : i * Inner + k));
        s00 = _mm256_fmadd_pd(a, b0, s00);
        s01 = _mm256_fmadd_pd(a, b1, s01);
        a = _mm256_broadcast_sd(left + (TransposeLeft ? k * Rows + i + 1 : (i + 1) * Inner + k));
        s10 = _mm256_fmadd_pd(a, b0, s10);
        s11 = _mm256_fmadd_pd(a, b1, s11);
        a = _mm256_broadcast_sd(left + (TransposeLeft ? k * Rows + i + 2 : (i + 2) * Inner + k));
        s20 = _mm256_fmadd_pd(a, b0, s20);
        s21 = _mm256_fmadd_pd(a, b1, s21);
        a = _mm256_broadcast_sd(left + (TransposeLeft ? k * Rows + i + 3 : (i + 3) * Inner + k));
        s30 = _mm256_fmadd_pd(a, b0, s30);
        s31 = _mm256_fmadd_pd(a, b1, s31);
      }
      _mm256_storeu_pd(out + i * Cols + j, s00);
      _mm256_storeu_pd(out + i * Cols + j + 4, s01);
      _mm256_storeu_pd(out + (i + 1) * Cols + j, s10);
      _mm256_storeu_pd(out + (i + 1) * Cols + j + 4, s11);
      _mm256_storeu_pd(out + (i + 2) * Cols + j, s20);
      _mm256_storeu_pd(out + (i + 2) * Cols + j + 4, s21);
      _mm256_storeu_pd(out + (i + 3) * Cols + j, s30);
      _mm256_storeu_pd(out + (i + 3) * Cols + j + 4, s31);
    }
    // Exact column tails; no padding or unrequested component is read or stored.
    for (int row = i; row < i + 4; row++)
      for (int j = CJ; j < Cols; j++) {
        double v = 0;
        for (int k = 0; k < Inner; k++)
          v += left[TransposeLeft ? k * Rows + row : row * Inner + k] * right[k * Cols + j];
        out[row * Cols + j] = v;
      }
  }
  for (int i = RI; i < Rows; i++) {
    for (int j = 0; j < CJ; j += 8) {
      __m256d s0 = _mm256_setzero_pd(), s1 = s0;
      for (int k = 0; k < Inner; k++) {
        auto a = _mm256_broadcast_sd(left + (TransposeLeft ? k * Rows + i : i * Inner + k));
        s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(right + k * Cols + j), s0);
        s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(right + k * Cols + j + 4), s1);
      }
      _mm256_storeu_pd(out + i * Cols + j, s0);
      _mm256_storeu_pd(out + i * Cols + j + 4, s1);
    }
    for (int j = CJ; j < Cols; j++) {
      double v = 0;
      for (int k = 0; k < Inner; k++)
        v += left[TransposeLeft ? k * Rows + i : i * Inner + k] * right[k * Cols + j];
      out[i * Cols + j] = v;
    }
  }
}
static bool available() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif
void angular_products(int H, int K, int A, int C, const double *M, const double *ea,
                      const double *ec, double *U, double *G) {
#if defined(GTO_AVX2)
  if (available()) {
    if (H == 35 && K == 35 && A == 25 && C == 25) {
      product_avx2<35, 35, 25, false>(M, ec, U);
      product_avx2<25, 35, 25, true>(U, ea, G);
      return;
    }
    if (H == 35 && K == 35 && A == 36 && C == 36) {
      product_avx2<35, 35, 36, false>(M, ec, U);
      product_avx2<36, 35, 36, true>(U, ea, G);
      return;
    }
  }
#endif
  for (int h = 0; h < H; h++)
    for (int c = 0; c < C; c++) {
      double v = 0;
      for (int k = 0; k < K; k++)
        v += M[h * K + k] * ec[k * C + c];
      U[h * C + c] = v;
    }
  for (int c = 0; c < C; c++)
    for (int a = 0; a < A; a++) {
      double v = 0;
      for (int h = 0; h < H; h++)
        v += U[h * C + c] * ea[h * A + a];
      G[a + A * c] = v;
    }
}
// Portable four-column register tile; each dot retains ascending inner order.
// No FMA or reassociation is introduced here; compiler contraction stays off.
void angular_products_block4(int H, int K, int A, int C, const double *M, const double *ea,
                             const double *ec, double *U, double *G) {
  for (int h = 0; h < H; h++) {
    int c = 0;
    for (; c + 4 <= C; c += 4) {
      double sum[4] = {};
      for (int k = 0; k < K; k++) {
        double m = M[h * K + k];
        for (int j = 0; j < 4; j++)
          sum[j] += m * ec[k * C + c + j];
      }
      for (int j = 0; j < 4; j++)
        U[h * C + c + j] = sum[j];
    }
    for (; c < C; c++) {
      double sum = 0;
      for (int k = 0; k < K; k++)
        sum += M[h * K + k] * ec[k * C + c];
      U[h * C + c] = sum;
    }
  }
  for (int c = 0; c < C; c++) {
    int a = 0;
    for (; a + 4 <= A; a += 4) {
      double sum[4] = {};
      for (int h = 0; h < H; h++) {
        double u = U[h * C + c];
        for (int j = 0; j < 4; j++)
          sum[j] += u * ea[h * A + a + j];
      }
      for (int j = 0; j < 4; j++)
        G[a + j + A * c] = sum[j];
    }
    for (; a < A; a++) {
      double sum = 0;
      for (int h = 0; h < H; h++)
        sum += U[h * C + c] * ea[h * A + a];
      G[a + A * c] = sum;
    }
  }
}

#if defined(GTO_AVX2)
// Ordered non-FMA microkernel. SIMD spans four independent output columns;
// four row accumulators hide add latency without changing any dot order.
template <bool TransposeLeft, bool Masked = false>
__attribute__((target("avx2"))) static void product_ordered_avx2(int rows, int inner, int cols,
                                                                 const double *left,
                                                                 const double *right, double *out) {
  const int RI = rows / 4 * 4, CJ = cols / 4 * 4;
  for (int row = 0; row < RI; row += 4) {
    for (int col = 0; col < CJ; col += 4) {
      __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
      for (int k = 0; k < inner; k++) {
        __m256d b = _mm256_loadu_pd(right + k * cols + col);
        __m256d a0 = _mm256_broadcast_sd(left + (TransposeLeft ? k * rows + row : row * inner + k));
        __m256d a1 = _mm256_broadcast_sd(
            left + (TransposeLeft ? k * rows + row + 1 : (row + 1) * inner + k));
        __m256d a2 = _mm256_broadcast_sd(
            left + (TransposeLeft ? k * rows + row + 2 : (row + 2) * inner + k));
        __m256d a3 = _mm256_broadcast_sd(
            left + (TransposeLeft ? k * rows + row + 3 : (row + 3) * inner + k));
        s0 = _mm256_add_pd(s0, _mm256_mul_pd(a0, b));
        s1 = _mm256_add_pd(s1, _mm256_mul_pd(a1, b));
        s2 = _mm256_add_pd(s2, _mm256_mul_pd(a2, b));
        s3 = _mm256_add_pd(s3, _mm256_mul_pd(a3, b));
      }
      _mm256_storeu_pd(out + row * cols + col, s0);
      _mm256_storeu_pd(out + (row + 1) * cols + col, s1);
      _mm256_storeu_pd(out + (row + 2) * cols + col, s2);
      _mm256_storeu_pd(out + (row + 3) * cols + col, s3);
    }
    if constexpr (Masked) {
      const int tail = cols - CJ;
      if (tail) {
        const __m256i mask = _mm256_set_epi64x(tail > 3 ? -1LL : 0, tail > 2 ? -1LL : 0,
                                               tail > 1 ? -1LL : 0, tail > 0 ? -1LL : 0);
        __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
        for (int k = 0; k < inner; k++) {
          __m256d b = _mm256_maskload_pd(right + k * cols + CJ, mask);
          auto a0 = _mm256_broadcast_sd(left + (TransposeLeft ? k * rows + row : row * inner + k));
          auto a1 = _mm256_broadcast_sd(
              left + (TransposeLeft ? k * rows + row + 1 : (row + 1) * inner + k));
          auto a2 = _mm256_broadcast_sd(
              left + (TransposeLeft ? k * rows + row + 2 : (row + 2) * inner + k));
          auto a3 = _mm256_broadcast_sd(
              left + (TransposeLeft ? k * rows + row + 3 : (row + 3) * inner + k));
          s0 = _mm256_add_pd(s0, _mm256_mul_pd(a0, b));
          s1 = _mm256_add_pd(s1, _mm256_mul_pd(a1, b));
          s2 = _mm256_add_pd(s2, _mm256_mul_pd(a2, b));
          s3 = _mm256_add_pd(s3, _mm256_mul_pd(a3, b));
        }
        _mm256_maskstore_pd(out + row * cols + CJ, mask, s0);
        _mm256_maskstore_pd(out + (row + 1) * cols + CJ, mask, s1);
        _mm256_maskstore_pd(out + (row + 2) * cols + CJ, mask, s2);
        _mm256_maskstore_pd(out + (row + 3) * cols + CJ, mask, s3);
      }
    } else {
      for (int i = row; i < row + 4; i++)
        for (int col = CJ; col < cols; col++) {
          double sum = 0;
          for (int k = 0; k < inner; k++)
            sum += left[TransposeLeft ? k * rows + i : i * inner + k] * right[k * cols + col];
          out[i * cols + col] = sum;
        }
    }
  }
  for (int row = RI; row < rows; row++) {
    for (int col = 0; col < CJ; col += 4) {
      __m256d sum = _mm256_setzero_pd();
      for (int k = 0; k < inner; k++) {
        auto a = _mm256_broadcast_sd(left + (TransposeLeft ? k * rows + row : row * inner + k));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(a, _mm256_loadu_pd(right + k * cols + col)));
      }
      _mm256_storeu_pd(out + row * cols + col, sum);
    }
    if constexpr (Masked) {
      const int tail = cols - CJ;
      if (tail) {
        const __m256i mask = _mm256_set_epi64x(tail > 3 ? -1LL : 0, tail > 2 ? -1LL : 0,
                                               tail > 1 ? -1LL : 0, tail > 0 ? -1LL : 0);
        __m256d sum = _mm256_setzero_pd();
        for (int k = 0; k < inner; k++) {
          auto a = _mm256_broadcast_sd(left + (TransposeLeft ? k * rows + row : row * inner + k));
          sum =
              _mm256_add_pd(sum, _mm256_mul_pd(a, _mm256_maskload_pd(right + k * cols + CJ, mask)));
        }
        _mm256_maskstore_pd(out + row * cols + CJ, mask, sum);
      }
    } else {
      for (int col = CJ; col < cols; col++) {
        double sum = 0;
        for (int k = 0; k < inner; k++)
          sum += left[TransposeLeft ? k * rows + row : row * inner + k] * right[k * cols + col];
        out[row * cols + col] = sum;
      }
    }
  }
}
#endif
void angular_products_ordered(int H, int K, int A, int C, const double *M, const double *ea,
                              const double *ec, double *U, double *G) {
#if defined(GTO_AVX2)
  if (__builtin_cpu_supports("avx2")) {
    {
      product_ordered_avx2<false, true>(H, K, C, M, ec, U);
      product_ordered_avx2<true, true>(C, H, A, U, ea, G);
    }
    return;
  }
#endif
  angular_products_block4(H, K, A, C, M, ea, ec, U, G);
}

}

namespace gto {
void angular_bra_ordered(int H, int A, int C, const double *ea, const double *U, double *G) {
#if defined(GTO_AVX2)
  if (__builtin_cpu_supports("avx2")) {
    { product_ordered_avx2<true, true>(C, H, A, U, ea, G); }
    return;
  }
#endif
  for (int c = 0; c < C; c++) {
    int a = 0;
    for (; a + 4 <= A; a += 4) {
      double sum[4] = {};
      for (int h = 0; h < H; h++) {
        double u = U[h * C + c];
        for (int j = 0; j < 4; j++)
          sum[j] += u * ea[h * A + a + j];
      }
      for (int j = 0; j < 4; j++)
        G[a + j + A * c] = sum[j];
    }
    for (; a < A; a++) {
      double sum = 0;
      for (int h = 0; h < H; h++)
        sum += U[h * C + c] * ea[h * A + a];
      G[a + A * c] = sum;
    }
  }
}
}

namespace gto {
#if defined(GTO_AVX2)
template <int Cols>
__attribute__((target("avx2"))) static void unique_rows(int H, int K, const unsigned short *index,
                                                        const double *values, const double *D,
                                                        double *U) {
  constexpr int blocks = (Cols + 3) / 4, tail = Cols % 4;
  const __m256i mask = _mm256_set_epi64x(tail > 3 ? -1LL : 0, tail > 2 ? -1LL : 0,
                                         tail > 1 ? -1LL : 0, tail > 0 ? -1LL : 0);
  for (int h = 0; h < H; h++) {
    __m256d sums[blocks];
#pragma GCC unroll 32
    for (int j = 0; j < blocks; j++)
      sums[j] = _mm256_setzero_pd();
    for (int k = 0; k < K; k++) {
      unsigned code = index[h * K + k];
      double x = values[code & 0x7fff];
      if (code & 0x8000)
        x = -x;
      __m256d a = _mm256_set1_pd(x);
#pragma GCC unroll 32
      for (int j = 0; j < blocks; j++) {
        __m256d b;
        if constexpr (tail) {
          if (j + 1 == blocks)
            b = _mm256_maskload_pd(D + k * Cols + 4 * j, mask);
          else
            b = _mm256_loadu_pd(D + k * Cols + 4 * j);
        } else
          b = _mm256_loadu_pd(D + k * Cols + 4 * j);
        sums[j] = _mm256_add_pd(sums[j], _mm256_mul_pd(a, b));
      }
    }
#pragma GCC unroll 32
    for (int j = 0; j < blocks; j++) {
      if constexpr (tail) {
        if (j + 1 == blocks)
          _mm256_maskstore_pd(U + h * Cols + 4 * j, mask, sums[j]);
        else
          _mm256_storeu_pd(U + h * Cols + 4 * j, sums[j]);
      } else
        _mm256_storeu_pd(U + h * Cols + 4 * j, sums[j]);
    }
  }
}
#endif
bool angular_unique_R(int H, int K, int A, int C, const unsigned short *index, const double *R,
                      const double *E, const double *D, double *U, double *G) {
#if defined(GTO_AVX2)
  if (!__builtin_cpu_supports("avx2"))
    return false;
  switch (C) {
  case 3:
    unique_rows<3>(H, K, index, R, D, U);
    break;
  case 5:
    unique_rows<5>(H, K, index, R, D, U);
    break;
  case 7:
    unique_rows<7>(H, K, index, R, D, U);
    break;
  case 9:
    unique_rows<9>(H, K, index, R, D, U);
    break;
  case 15:
    unique_rows<15>(H, K, index, R, D, U);
    break;
  case 21:
    unique_rows<21>(H, K, index, R, D, U);
    break;
  case 25:
    unique_rows<25>(H, K, index, R, D, U);
    break;
  case 35:
    unique_rows<35>(H, K, index, R, D, U);
    break;
  default:
    return false;
  }
  angular_bra_ordered(H, A, C, E, U, G);
  return true;
#else
  (void)H;
  (void)K;
  (void)A;
  (void)C;
  (void)index;
  (void)R;
  (void)E;
  (void)D;
  (void)U;
  (void)G;
  return false;
#endif
}
}

namespace gto {
namespace {
inline double signed_unique(const double *v, unsigned code, int nr) {
  return v[(code & 0x7fff) + ((code & 0x8000) ? nr : 0)];
}
inline unsigned compact_first_bit(std::uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
  return unsigned(__builtin_ctzll(v));
#else
  unsigned b = 0;
  while (!(v & 1)) {
    v >>= 1;
    ++b;
  }
  return b;
#endif
}
} // namespace
void angular_ps_dd_bra_unique(const unsigned short *codes, const double *values, const double *E,
                              const double *D, double *V, double *G) {
  for (int k = 0; k < 35; k++)
    for (int i = 0; i < 3; i++) {
      double m0 = signed_unique(values, codes[k], 56),
             m1 = signed_unique(values, codes[(3 - i) * 35 + k], 56);
      double z = 0;
      z += m0 * E[i];
      z += m1 * E[(3 - i) * 3 + i];
      V[k * 3 + i] = z;
    }
  angular_bra_ordered(35, 3, 25, V, D, G);
}
void angular_dd_ps_ket_unique(const unsigned short *codes, const double *values, const double *E,
                              const double *D, double *U, double *G) {
  for (int h = 0; h < 35; h++)
    for (int j = 0; j < 3; j++) {
      double m0 = signed_unique(values, codes[h * 4], 56),
             m1 = signed_unique(values, codes[h * 4 + 3 - j], 56);
      double z = 0;
      if (D[j] != 0)
        z += m0 * D[j];
      if (D[(3 - j) * 3 + j] != 0)
        z += m1 * D[(3 - j) * 3 + j];
      U[h * 3 + j] = z;
    }
  angular_bra_ordered(35, 25, 3, E, U, G);
}
void angular_ds_bra_unique(int K, const unsigned short *codes, const double *values,
                           const double *E, const double *D, const std::uint64_t *mask, double *V,
                           double *G) {
  const int nr = K == 20 ? 56 : 84, C = K == 20 ? 15 : 25;
  for (int k = 0; k < K; k++) {
    double *row = V + k * 5;
    for (int i = 0; i < 5; i++)
      row[i] = 0;
    for (int h = 0; h < 10; h++) {
      std::uint64_t bits = mask[h];
      if (!bits)
        continue;
      double m = signed_unique(values, codes[h * K + k], nr);
      while (bits) {
        unsigned i = compact_first_bit(bits);
        bits &= bits - 1;
        row[i] += m * E[h * 5 + i];
      }
    }
  }
  angular_bra_ordered(K, 5, C, V, D, G);
}
}
