
namespace cpu {

void cblas_sgemm(const bool transa, const bool transb,
    const int m, const int n, const int k,
    const float alpha, const float *a, const int lda,
    const float *b, const int ldb, const float beta,
    float *c, const int ldc);

} // namespace cpu
