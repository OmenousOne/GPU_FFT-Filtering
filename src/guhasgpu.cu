#include <stdio.h>

#include "gpu_fft_filtering.h"

int32 guhasgpu(void)
{
   int32 ngpus;
   int32 i=0;
   cudaDeviceProp prop;

   cudaGetDeviceCount(&ngpus);
   // printf("Number of GPU Devices: %d\n", ngpus);

  
   for (i=0; i < ngpus; i++) 
   {
      cudaGetDeviceProperties(&prop, i);
      //printf("Device Number: %d\n", i);
      printf("CUDA    Device name: %s\n", prop.name);
      //printf("  Device Compute Major: %d Minor: %d\n", prop.major, prop.minor);
      //printf("  Max Thread Dimensions: [%d][%d][%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      //printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
      //printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
      //printf("  Device Clock Rate (KHz): %d\n", prop.clockRate);
      //printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
      //printf("  Registers Per Block: %d\n", prop.regsPerBlock);
      //printf("  Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
      //printf("  Shared Memory Per Block: %zu\n", prop.sharedMemPerBlock);
      //printf("  Shared Memory Per Multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
      //printf("  Total Constant Memory (bytes): %zu\n", prop.totalConstMem);
      //printf("  Total Global Memory (bytes): %zu\n", prop.totalGlobalMem);
      //printf("  Warp Size: %d\n", prop.warpSize);
      //printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    
  }

  return ngpus;
}
