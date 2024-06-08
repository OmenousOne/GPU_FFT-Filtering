#include "gpu_fft_filtering.h"
#include <iostream>
#include <cstring>


int32  read_headers(char *Ce, char *Cb, FILE *Fi)
{
   int32   nread=0;

   // test if it is SEGY data
   nread=fread(Ce,1,EHEAD,Fi);
   if (nread!=EHEAD)
   {
      std::cout << "err - Read " << nread << "Bytes, expected " << EHEAD << std::endl;
      return(-1);
   }
   std::cout << "EBCDIC Header Read " << nread << " Bytes" << std::endl; 
   nread=fread(Cb,1,BHEAD,Fi);
   if (nread!=BHEAD)
   {
      std::cout << "err - Read " << nread << "Bytes, expected " << BHEAD << std::endl;
      return(-1);
   }
   std::cout << "Binary Header Read " << nread << " Bytes" << std::endl; 

   // test if it is a supported SEGY type
   if (segy_check(&Ce[0],&Cb[0]) != 0)
   {
      std::cout << "err - not a valid/supported SEGY type data" << std::endl;
      return(-1);
   }




   return(0);
}
