#include <cuda.h>
#include "cufft_error.h"

#define CHECK_CUDA_ERROR(ival) check_err((ival), #ival, __FILE__, __LINE__)
void check_err(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
   if (err != cudaSuccess)
   {
      std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
       std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
       // We exit when we encounter CUDA errors
       std::exit(EXIT_FAILURE);
   }
}


#define CHECK_LAST_CUDA_ERROR() checkLast_err(__FILE__, __LINE__)
void checkLast_err(const char* const file, const int line)
{
   cudaError_t const err{cudaGetLastError()};
   if (err != cudaSuccess)
   {
       std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                 << std::endl;
       std::cerr << cudaGetErrorString(err) << std::endl;
       // We exit when we encounter CUDA errors
       std::exit(EXIT_FAILURE);
    }
}


