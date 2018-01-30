#include <benchmark/benchmark.h>

#include <iostream>

#include "gemm/gemm_benchmark.h"
#include "gemm/sgemm.h"

namespace gemm {
namespace benchmark {

class Cpu : public GemmBenchmark {};

BENCHMARK_DEFINE_F(Cpu, Sgemm_0)(::benchmark::State &st) {
  int m = st.range(0);
  int k = st.range(1);
  int n = st.range(2);
  
  float *a = this->CreateRandomMatrix<float>(m, k);
  float *b = this->CreateRandomMatrix<float>(k, n);
  float *c = this->CreateZeroMatrix<float>(m, n);
  
  for (auto _ : st) {
    gemm::cblas_sgemm_0(
        false,
        false,
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

BENCHMARK_REGISTER_F(Cpu, Sgemm_0)
  ->Apply(DeepBenchMatrixDims)
  ->Unit(::benchmark::kMicrosecond)
  ->Iterations(10)
  ->UseRealTime();

BENCHMARK_DEFINE_F(Cpu, Sgemm_1)(::benchmark::State &st) {
  int m = st.range(0);
  int k = st.range(1);
  int n = st.range(2);
  
  float *a = this->CreateRandomMatrix<float>(m, k);
  float *b = this->CreateRandomMatrix<float>(k, n);
  float *c = this->CreateZeroMatrix<float>(m, n);
  
  for (auto _ : st) {
    gemm::cblas_sgemm_1(
        false,
        false,
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

BENCHMARK_REGISTER_F(Cpu, Sgemm_1)
  ->Apply(DeepBenchMatrixDims)
  ->Unit(::benchmark::kMicrosecond)
  ->Iterations(10)
  ->UseRealTime();

BENCHMARK_DEFINE_F(Cpu, Sgemm_2)(::benchmark::State &st) {
  int m = st.range(0);
  int k = st.range(1);
  int n = st.range(2);
  
  float *a = this->CreateRandomMatrix<float>(m, k);
  float *b = this->CreateRandomMatrix<float>(k, n);
  float *c = this->CreateZeroMatrix<float>(m, n);
  
  for (auto _ : st) {
    gemm::cblas_sgemm_2(
        false,
        false,
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

BENCHMARK_REGISTER_F(Cpu, Sgemm_2)
  ->Apply(DeepBenchMatrixDims)
  ->Unit(::benchmark::kMicrosecond)
  ->Iterations(10)
  ->UseRealTime();

} // namespace benchmark
} // namespace gemm
