#include <iostream>
#include <cstring>
#include <complex>
#include <cuda.h>
#include "cuda_error.h"
#include "helper_cuda.h"
#include "cufft.h"

#include "gpu_fft_filtering.h"

typedef float2 Complex;

#define BATCH    1

__global__ void MinMaxFilt(Complex *a, int32 nminf, int32 nmaxf);


void gufftfilt_32f_I(int32 nmin,int32 nmax,int32 nsampst,int32 nsamprt,float *Fdata,int32 nsize)
{
   int32   i=0;
   int32   ielem;             // number of elements
   int32   nchunk=0,nleft=0;
   int32   nyquist=0;
   int32   nfreqs=0;
   int32   nminf=0,nmaxf=0;
   float   df=0.0f;
   cufftReal    *cu_rFdata;
   cufftComplex *cu_Fdata;
   cufftHandle  cu_plan;

   ielem=nsize/sizeof(float);
   nchunk=nsampst;    // need to process one trace at a time for filter to work correctly
   if (ielem<nchunk)  nchunk=ielem;
   nleft=ielem;

   // std::cout << "cudaMallocing " << (nchunk/2+1)*BATCH*sizeof(cufftComplex) << " bytes" << std::endl;
   CHECK_CUDA_ERROR(cudaMalloc((void**)&cu_rFdata,nchunk*sizeof(cufftReal)));    // device
   CHECK_CUDA_ERROR(cudaMalloc((void**)&cu_Fdata,(nchunk/2+1)*BATCH*sizeof(cufftComplex)));    // device

   // std::cout << "nchunk: " << nchunk << std::endl;

   for (i=0; i<ielem;)
   {
      // copy to device memory
      CHECK_CUDA_ERROR(cudaMemcpy((void*)cu_rFdata,(cufftReal*)&Fdata[i],nchunk*sizeof(cufftReal),cudaMemcpyHostToDevice));

      // CUFFT plan
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      CHECK_CUFFT_ERROR(cufftPlan1d(&cu_plan,nchunk, CUFFT_R2C, BATCH));
      // transform 
      CHECK_CUFFT_ERROR(cufftExecR2C(cu_plan, (cufftReal *)cu_rFdata, cu_Fdata));
      // free plan memory
      cufftDestroy(cu_plan);

      // apply  filter
      // calculate nyquist value nfreqs and  df
      // nyquist = 500 / sample rate in ms, segy stores sample rate in microseconds
      nyquist = 500 / (nsamprt/1000);

      // number of frequencies = (number of samples per trace / sample rate in ms) + 1
      nfreqs = (nsampst / (nsamprt/1000)) + 1;

      // df or frequency interval = nyquist / (number of frequecies - 1)
      df = (float)nyquist / (float)(nfreqs - 1);
 
      // calculate min and max freq to keep
      nminf = (int32)((float)nmin / df);
      nmaxf = (int32)((float)nmax / df);

      MinMaxFilt <<< nchunk/2,1 >>>(cu_Fdata, nminf, nmaxf);

      // transform back
      CHECK_CUFFT_ERROR(cufftPlan1d(&cu_plan, nchunk, CUFFT_C2R, BATCH));
      CHECK_CUFFT_ERROR(cufftExecC2R(cu_plan, cu_Fdata, (cufftReal *)cu_rFdata));
      // free plan memory
      cufftDestroy(cu_plan);

      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      // copy to host memory
      CHECK_CUDA_ERROR(cudaMemcpy((cufftReal*)&Fdata[i],(void*)cu_rFdata,nchunk*sizeof(cufftReal),cudaMemcpyDeviceToHost));

      i+=nchunk;
      nleft-=nchunk;
      if (nleft<nchunk) nchunk=nleft;
   }
   

   // free memory
   cudaFree(cu_Fdata);
   cudaFree(cu_rFdata);

   return;
}

__global__ void MinMaxFilt(Complex *a, int32 nminf, int32 nmaxf)
{
   int32  ithread;

   ithread=blockIdx.x * blockDim.x + threadIdx.x;

   // printf("\n\nHello from thread id %d nminf(%d)  nmaxf(%d)\n\n",ithread,nminf,nmaxf);
   // if (ithread>=0 && ithread<8)  printf("Values(%d)  x %0.8f   y %0.8f \n",ithread,a[ithread].x,a[ithread].y);

   // zero out amplitude and phases of freqs not in the range we want to keep 
   if (ithread<nminf || ithread>nmaxf)
   {
      a[ithread].x=0.0f;
      a[ithread].y=0.0f;
      // printf("\nZEROED thread id %d nminf(%d)  nmaxf(%d) a[].x(%0.8f) a[%0.8f].y\n",
      //        ithread,nminf,nmaxf,a[ithread].x,a[ithread].y);
   }
}

