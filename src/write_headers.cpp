#include "gpu_fft_filtering.h"
#include <iostream>
#include <cstring>


int32  write_headers(char *Ce, char *Cb, FILE *Fo)
{
   int32 nwrite;

   nwrite=fwrite(Ce,1,EHEAD,Fo);
   if (nwrite!=EHEAD)
   {
      std::cout << "err - Wrote " << nwrite << "Bytes, expected " << EHEAD << std::endl;
      return(-1);
   }
   nwrite=fwrite(Cb,1,BHEAD,Fo);
   if (nwrite!=BHEAD)
   {
      std::cout << "err - Wrote " << nwrite << "Bytes, expected " << BHEAD << std::endl;
      return(-1);
   }
   
   return(0);
}
