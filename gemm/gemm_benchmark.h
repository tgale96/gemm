#include <benchmark/benchmark.h>

#include <random>

namespace gemm {
namespace benchmark {

class GemmBenchmark : public ::benchmark::Fixture {
public:
  GemmBenchmark() : gen_(time(nullptr)) {}

  ~GemmBenchmark() = default;
  
  template <typename T>
  T* CreateZeroMatrix(int a, int b) {
    T *mat = new T[a*b];
    memset(mat, 0, sizeof(T)*a*b);
    return mat;
  }
  
  template <typename T>
  T* CreateRandomMatrix(int a, int b) {
    std::uniform_real_distribution<> d(1.0, 10000.0);
    T *mat = new T[a*b];
    for (int i = 0; i < a*b; ++i) {
      mat[i] = static_cast<float>(d(gen_));
    }
    return mat;
  }

private:
  std::mt19937 gen_;
};

} // namespace benchmark
} // namespace gemm
