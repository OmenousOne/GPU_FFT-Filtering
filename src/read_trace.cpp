#include "gpu_fft_filtering.h"
#include <iostream>
#include <cstring>


int32 read_trace(char *Cbuff,int32 numt,FILE *Fi)
{
   // read trace(s) into buffer
   int32 nsize=0;
   int32 nread=0;
   int32 i=0,j=0,iloc=0;
   float *Fp;
   
   // read number of traces * trace size 
   //                         ((bytes per sample * number of samples )+trace header)
   nsize = numt * ((nbpsamp*nsamp)+THEAD);
   nread = fread(Cbuff,1,nsize,Fi);
   if (nread==0)
   {
      std::cout << "EOF - end of file, 0 traces read" << std::endl;
      return(0);
   }
   // convert to traces read
   nread = nread/((nbpsamp*nsamp)+THEAD);
   if (nread < 1)
   {
      std::cout << "err - less than a full trace read" << std::endl;
      return(-1);
   }

   // byte swap data if needed
   if (iswapd)
   {
      for(i=0; i<nread; i++)
      {
         // only byte swap data not headers
         iloc=i*(((nbpsamp*nsamp)+THEAD))+THEAD;
         Fp=(float*)&Cbuff[iloc];
         for(j=0; j<nsamp; j++)
         {
            *Fp = bswapf(*Fp);
            Fp++;  // move pointer to next sample
         }
      }
   }

   return(nread);
}
