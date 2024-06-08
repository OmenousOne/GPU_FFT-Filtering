#include "gpu_fft_filtering.h"
#include <iostream>
#include <cstring>


int32 write_trace(char *Cbuff,int32 numt,FILE *Fo)
{
   // write trace(s) to FILE
   int32 nsize=0;
   int32 nwrite=0;
   int32 i=0,j=0,iloc=0;
   float *Fp;
   
   // byte swap data if needed
   if (iswapd)
   {
      for(i=0; i<numt; i++)
      {
         // only byte swap data not headers
         iloc=(i*((nbpsamp*nsamp)+THEAD))+THEAD;
         Fp=(float*)&Cbuff[iloc];
         for(j=0; j<nsamp; j++)
         {
            *Fp = bswapf(*Fp);
            Fp++;  // move pointer to next sample
         }
      }
   }

   // write number of traces * trace size 
   //                         ((bytes per sample * number of samples )+trace header)
   nsize = numt * ((nbpsamp*nsamp)+THEAD);
   nwrite = fwrite(Cbuff,1,nsize,Fo);
   if (nwrite==0)
   {
      std::cout << "err - wrote 0 traces, disk full?" << std::endl;
      return(0);
   }
   // convert to traces read
   nwrite = nwrite/((nbpsamp*nsamp)+THEAD);
   if (nwrite != numt)
   {
      std::cout << "err - Wrote " << nwrite << "Traces, expected " << numt << std::endl;
      return(-1);
   }

   return(nwrite);
}
