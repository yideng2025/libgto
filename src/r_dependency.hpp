#pragma once
#include "internal.hpp"
namespace gto {
struct HermiteRStep {
  unsigned short dst, first, second;
  unsigned char axis, coefficient;
};
constexpr Triple hermite_triple(int index) {
  int degree = 0;
  while (index >= nh(degree))
    ++degree;
  int row = index - (degree ? nh(degree - 1) : 0);
  for (int x = 0; x <= degree; x++) {
    int width = degree - x + 1;
    if (row < width)
      return {x, row, degree - x - row};
    row -= width;
  }
  return {0, 0, 0};
}

template <class Offsets, class Emit>
constexpr void emit_r_dependencies(int L, const Offsets &offset, Emit emit) {
  for (int m = L - 1; m >= 0; m--)
    for (int h = 1; h < nh(L - m); h++) {
      auto x = hermite_triple(h);
      int axis = x[0] ? 0 : x[1] ? 1 : 2, degree = x[axis];
      --x[axis];
      int first = offset[m + 1] + hi(x[0], x[1], x[2]), second = first;
      if (degree > 1) {
        --x[axis];
        second = offset[m + 1] + hi(x[0], x[1], x[2]);
      }
      emit(HermiteRStep{static_cast<unsigned short>(offset[m] + h),
                        static_cast<unsigned short>(first), static_cast<unsigned short>(second),
                        static_cast<unsigned char>(axis), static_cast<unsigned char>(degree - 1)});
    }
}
// Shared mathematical coupling map: final spatial R index plus ket phase.
template <class TripleAt, class Emit>
constexpr void emit_hermite_coupling(int A, int B, TripleAt triple, Emit emit) {
  for (int h = 0; h < nh(A); h++)
    for (int k = 0; k < nh(B); k++) {
      auto x = triple(h), y = triple(k);
      unsigned index = hi(x[0] + y[0], x[1] + y[1], x[2] + y[2]);
      if ((y[0] + y[1] + y[2]) & 1)
        index |= 0x8000;
      emit(static_cast<unsigned short>(index));
    }
}

}
