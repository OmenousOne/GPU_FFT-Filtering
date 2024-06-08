#include "gpu_fft_filtering.h"
#include <iostream>
#include <cstring>

void   trace_2floatbuff(char *Cbuff,float *Fbuff,int32 numt)
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
         Fbuff[ifloc] = *Fp;
         ifloc++;
         Fp++;
      }
   }
   
   return;
}

