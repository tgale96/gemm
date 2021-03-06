#include <Accelerate/Accelerate.h>
#include <benchmark/benchmark.h>

#include <iostream>

#include "gemm/gemm_benchmark.h"

namespace gemm {
namespace benchmark {

class Blas : public GemmBenchmark {};

BENCHMARK_DEFINE_F(Blas, Sgemm)(::benchmark::State &st) {
  int m = st.range(0);
  int k = st.range(1);
  int n = st.range(2);
  
  float *a = this->CreateRandomMatrix<float>(m, k);
  float *b = this->CreateRandomMatrix<float>(k, n);
  float *c = this->CreateZeroMatrix<float>(m, n);

  for (auto _ : st) {
    cblas_sgemm(CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        m, n, k,
        1.f,
        a, m,
        b, k,
        0.f,
        c, m);
  }

  delete[] a;
  delete[] b;
  delete[] c;
}

BENCHMARK_REGISTER_F(Blas, Sgemm)
  ->Apply(DeepBenchMatrixDims)
  ->Unit(::benchmark::kMicrosecond)
  ->Iterations(10)
  ->UseRealTime();

} // namespace benchmark
} // namespace gemm
