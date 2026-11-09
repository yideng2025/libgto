#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
// Standalone C++17 generator, no production dependency. Positive convergent
// series at m=22, compensated accumulation, then stable downward recurrence.
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  std::ofstream out(argv[1]);
  if (!out)
    return 3;
  out << "// Reproduce with tools/generate_boys.cpp. Scalar MD Boys orders 0..8.\n#ifndef "
         "LIBGTO_BOYS_TABLE_HPP\n#define LIBGTO_BOYS_TABLE_HPP\nnamespace gto {\nstatic constexpr "
         "long double inverse_factorial[15]={\n";
  long double f = 1;
  for (int k = 0; k <= 14; k++) {
    if (k)
      f /= k;
    out << std::hexfloat << f << "L,";
  }
  out << "};\nstatic constexpr long double boys_table[65][23]={\n";
  for (unsigned i = 0; i <= 64; i++) {
    long double T = i * .5L, a[23];
    if (i == 0) {
      for (int m = 0; m <= 22; m++)
        a[m] = 1.L / (2 * m + 1);
    } else {
      long double term = 1, sum = 1, c = 0;
      int k = 0;
      for (; k < 512; k++) {
        term *= 2 * T / (47 + 2 * k);
        long double v = sum + term;
        c += std::abs(sum) >= std::abs(term) ? (sum - v) + term : (term - v) + sum;
        sum = v;
        long double ratio = 2 * T / (49 + 2 * k);
        if (ratio < 1 && term * ratio / (1 - ratio) <
                             std::numeric_limits<long double>::epsilon() * (sum + c) / 16)
          break;
      }
      if (k == 512)
        return 4;
      long double e = std::exp(-T);
      a[22] = e * (sum + c) / 45;
      for (int m = 22; m > 0; m--)
        a[m - 1] = (2 * T * a[m] + e) / (2 * m - 1);
    }
    out << "{";
    for (auto v : a)
      out << std::hexfloat << v << "L,";
    out << "},\n";
  }
  out << "};\n}\n#endif\n";
  return !out;
}
