#include "gpu_fft_filtering.h"
#include <iostream>
#include <cstring>


int32 segy_check(char *Ce, char *Cb)
{
   // check if valid/supported segy type data
   int32  i,na=0,ne=0;
   int32  *Lp;
   int16  *Sp;
   char   ch;

   for (i=0; i<180; i++)
   {
      ch=Ce[i];
      if (ch == '\x40') ne++;
      if (ch == '\x20') na++;
   }
   if (na+ne>0)
   {
      netype=0;
      if (ne>na) netype=1;
      if(netype) std::cout << "Ebcdic Header: EBCDIC" << std::endl;
      else       std::cout << "Ebcdic Header: ASCII" << std::endl;
   }
   else
   {
      std::cout << "Ebcdic Header: Type Not Found" << std::endl;
      return(-1); 
   }
   // grab info from binary header
   // revision
   Lp = (int32*)&Cb[96];
   nrev = *Lp;
   if ((!iendian && nbtype) || (iendian && !nbtype)) nrev = bswaps(nrev);
   if (nrev==16909060)
   {
      // this data sets endian matches this CPU
      std::cout << "Revision >= 2.0 Detected, Endian is correct for this CPU" << std::endl;
      nrevd=2;
   }
   else if(nrev==67305985)
   {
      // this data sets endian does NOT match this CPU
      std::cout << "Revision >= 2.0 Detected, Endian is NOT correct for this CPU" << std::endl;
      nrevd=-2;
   }
   else
   {
      std::cout << "Revision < 2.0 Detected" << std::endl;
      nrevd=0;
   }

   // format code
   Sp = (int16*)&Cb[24];
   nformat = *Sp;
   if (! iendian) nformat = bswaps(nformat); 
   if (nformat>255)
   {
      nformat = *Sp;
      nbtype=0;
      std::cout << "Binary Header: PC ORDER" << std::endl;
      
      if ((!iendian && nbtype) || (iendian && !nbtype)) nformat = bswaps(nformat);
   }
   else
   {
      nbtype=1;
      std::cout << "Binary Header: IBM ORDER" << std::endl;
   }
   if (nformat>20)
   {
      std::cout << "err - " << nformat << " is not a known format" << std::endl;
      return(-1);
   }
   if ((nformat!=6 && nformat!=11 && nrevd!=0 ) && (nformat!=5 && nrevd!=2 && nrevd!=-2))
   {
      std::cout << "err - Format: " << nformat << " Revision: " << nrevd << " is not supported currently" << std::endl;
      return(-1);
   }
   //assuming IEEE 4 bytes Floats
   nbpsamp=sizeof(float);

   // get some information about this data
   Sp = (int16*)&Cb[12];
   ntrac = *Sp;
   if ((!iendian && nbtype) || (iendian && !nbtype)) ntrac = bswaps(ntrac);
   Sp = (int16*)&Cb[14];
   ntaux = *Sp;
   if ((!iendian && nbtype) || (iendian && !nbtype)) ntaux = bswaps(ntaux);
   Sp = (int16*)&Cb[16];
   nsint = *Sp;
   if ((!iendian && nbtype) || (iendian && !nbtype)) nsint = bswaps(nsint);
   Sp = (int16*)&Cb[20];
   nsamp = *Sp;
   if ((!iendian && nbtype) || (iendian && !nbtype)) nsamp = bswaps(nsamp);
   
   printf("Num Traces:      %4d (%04X)  bytes 3213-3214\n",ntrac,ntrac);
   printf("Num Aux Tr:      %4d (%04X)  bytes 3215-3216\n",ntaux,ntaux);
   printf("Sample Interval: %4d (%04X)  bytes 3217-3218\n",nsint,nsint);
   printf("Num Samples:     %4d (%04X)  bytes 3221-3222\n",nsamp,nsamp);
   printf("Revision:  (%2d)  %4d (%04X)  bytes 3297-3300\n",nrevd,nrev,nrev);
   printf("Format:          %4d (%04X)  bytes 3225-3226\n\n",nformat,nformat);
   printf("Calculated Number of Traces:          %ld\n",(lfilesz-3600)/((nsamp*sizeof(float))+THEAD));

   if (nrevd==-2) iswapd=1;
   if (nrevd==0 && nformat==6 && iendian==0) iswapd=1;
   if (nrevd==0 && nformat==11 && iendian==1) iswapd=1;
   if (iswapd)  std::cout << "Data Byte Swap Required" << std::endl;

   return(0);
}
		
