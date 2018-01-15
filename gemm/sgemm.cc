#include "error_checking.h"

#include <cstring>

namespace gemm {

// Computes C = alpha*op(A)*op(B) + beta*C, where
// A is an m x k matrix, and B is a k x n. A, B, C
// are store in column major format
void cblas_sgemm_0(const bool transa, const bool transb,
    const int m, const int n, const int k,
    const float alpha, const float *a, const int lda,
    const float *b, const int ldb, const float beta,
    float *c, const int ldc) {
  ASSERT(!transa);
  ASSERT(!transb);
  ASSERT(a != nullptr);
  ASSERT(b != nullptr);
  ASSERT(c != nullptr);
  ASSERT(m > 0);
  ASSERT(n > 0);
  ASSERT(k > 0);
  ASSERT(lda == m);
  ASSERT(ldb == k);
  ASSERT(ldc == m);
  ASSERT(beta == 0);
  ASSERT(alpha == 1);

  memset(c, 0, sizeof(float)*m*n);
  
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < k; ++l) {
        c[i + j * m] += a[i + l * m] * b[l + j * k];
      }
    }
  }
}

} // namespace gemm
