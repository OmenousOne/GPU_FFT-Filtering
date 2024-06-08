#include "gpu_fft_filtering.h"
#include <iostream>
#include <cstring>

void   float_2tracebuff(float *Fbuff,char *Cbuff,int32 numt)
{
   int32 i=0,j=0;
   int32 icloc=0,ifloc=0;
   float *Fp;
   
   for (i=0; i<numt; i++)
   {
      icloc=i*((nbpsamp*nsamp)+THEAD)+THEAD;
      Fp=(float*)&Cbuff[icloc];
      for(j=0; j<nsamp; j++)
      {
         *Fp= Fbuff[ifloc];
         ifloc++;
         Fp++;
      }
   }
   
   return;
}

