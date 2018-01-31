#include "error_checking.h"

#include <cstring>

#include <iostream>
using std::cout;
using std::endl;

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

/**
 * Because we are working in column major, this permutation of the loops 
 * is slower. We step through the inner loop non-contiguously, which
 * yields worse cache utilization
 */
void cblas_sgemm_1(const bool transa, const bool transb,
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

  for (int l = 0; l < k; ++l) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        c[i + j * m] += a[i + l * m] * b[l + j * k];
      }
    }
  }
}

/**
 * This permutation of the loops accesses mats A, B, and C
 * consecutively in memory (in column-major, w/ no transposition).
 */
void cblas_sgemm_2(const bool transa, const bool transb,
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

  for (int j = 0; j < n; ++j) {
    for (int l = 0; l < k; ++l) {
      for (int i = 0; i < m; ++i) {
        c[i + j * m] += a[i + l * m] * b[l + j * k];
      }
    }
  }
}

/**
 * GEMM w/ tiled M & N dimensions
 */
void cblas_sgemm_3(const bool transa, const bool transb,
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

  // Tuning parameters
  constexpr int m_tile = 64;
  constexpr int n_tile = 64;
  
  for (int i = 0; i < m; i += m_tile) {
    for (int j = 0; j < n; j += n_tile) {
      // Calculate the size of the tiles we will
      // be working with
      int m_tile_size = std::min(m-i, m_tile);
      int n_tile_size = std::min(n-j, n_tile);
      for (int q = 0; q < n_tile_size; ++q) {
        for (int l = 0; l < k; ++l) {
          for (int p = 0; p < m_tile_size; ++p) {
            int row = i + p;
            int col = j + q;
            c[col*m + row] += a[l*m + row] * b[col*k + l];
          }
        }
      }
    }
  }
}

} // namespace gemm
