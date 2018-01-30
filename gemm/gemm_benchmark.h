#include <benchmark/benchmark.h>

#include <chrono>
#include <random>

namespace gemm {
namespace benchmark {

#ifdef FULLDB
inline void DeepBenchMatrixDims(::benchmark::internal::Benchmark *b) {
  std::vector<int> ms = {16, 32, 64, 128, 7000};
  std::vector<int> nks = {1760, 2048, 2560, 4096};
  for (auto &m : ms) {
    for (auto &nk : nks) {
      b->Args({m, nk, nk});
    }
  }
}

#else
inline void DeepBenchMatrixDims(::benchmark::internal::Benchmark *b) {
  std::vector<int> ms = {16, 32, 64, 128};
  std::vector<int> nks = {1760, 2048, 2560};
  for (auto &m : ms) {
    for (auto &nk : nks) {
      b->Args({m, nk, nk});
    }
  }
}

#endif // FULLDB

class GemmBenchmark : public ::benchmark::Fixture {
public:
  inline GemmBenchmark() : gen_(time(nullptr)) {}

  inline ~GemmBenchmark() = default;
  
  template <typename T>
  inline T* CreateZeroMatrix(int a, int b) {
    T *mat = new T[a*b];
    memset(mat, 0, sizeof(T)*a*b);
    return mat;
  }
  
  template <typename T>
  inline T* CreateRandomMatrix(int a, int b) {
    std::uniform_real_distribution<> d(1.0, 10000.0);
    T *mat = new T[a*b];
    for (int i = 0; i < a*b; ++i) {
      mat[i] = static_cast<float>(d(gen_));
    }
    return mat;
  }

  inline std::chrono::high_resolution_clock::time_point GetTime() {
    return std::chrono::high_resolution_clock::now();
  }
  
  inline float ElapsedTime(std::chrono::high_resolution_clock::time_point start,
      std::chrono::high_resolution_clock::time_point end) {
    return float(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1000000000;
  }

private:
  std::mt19937 gen_;
};

} // namespace benchmark
} // namespace gemm
