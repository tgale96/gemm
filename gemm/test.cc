#include "sgemm.h"

#include <Accelerate/Accelerate.h>

#include <chrono>
#include <iostream>
using namespace std;
using namespace chrono;

high_resolution_clock::time_point GetTime() {
  return high_resolution_clock::now();
}

float ElapsedTime(high_resolution_clock::time_point start,
    high_resolution_clock::time_point end) {
  return float(duration_cast<nanoseconds>(end - start).count()) / 1000000000;
}

int main() {
  int m = 16;
  int k = 1760;
  int n = 1760;
  
  float *a = new float[m*k];
  float *b = new float[k*n];
  float *c = new float[m*n];

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      a[i + j*m] = 2.f;
    }
  }

  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      b[i + j*k] = 2.f;
    }
  }

  cout << m << "x" << k << " * " << k << "x" << n << endl;
  
  auto t1 = GetTime();
  cblas_sgemm(CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      m, n, k,
      1.f,
      a, m,
      b, k,
      0.f,
      c, m);
  auto total_seconds = ElapsedTime(t1, GetTime());
  cout << "Reference: " << total_seconds << " seconds" << endl;
  
  t1 = GetTime();
  cpu::cblas_sgemm(false,
      false,
      m, n, k,
      1.0,
      a, m,
      b, k,
      0.0,
      c, m);
  total_seconds = ElapsedTime(t1, GetTime());
  cout << "Schedule 0: " << total_seconds << " seconds" << endl;
  
  // for (int i = 0; i < m; ++i) {
  //   for (int j = 0; j < n; ++j) {
  //     cout << c[i + j*m] << " ";
  //   }
  //   cout << endl;
  // }
  
  return 0;
}
