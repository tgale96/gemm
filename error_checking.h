#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
using std::string;
using std::to_string;
using std::runtime_error;

#define CUDA_CALL(code)                         \
  do {                                          \
    cudaError_t status = code;                  \
    if (status != cudaSuccess) {                \
      string file = __FILE__;                   \
      string line = to_string(__LINE__);        \
      string error = "[" + file + ":" + line +  \
        "]: CUDA error \"" +                    \
        cudaGetErrorString(status) + "\"";      \
      throw runtime_error(error);               \
    }                                           \
  } while (0)

#define ASSERT(code)                            \
  do {                                          \
    if (!(code)) {                              \
      string test = #code;                      \
      string file = __FILE__;                   \
      string line = to_string(__LINE__);        \
      string error = "[" + file + ":" + line +  \
        "]: assert on \"" + test + "\"";        \
      throw runtime_error(error);               \
    }                                           \
  } while (0)
