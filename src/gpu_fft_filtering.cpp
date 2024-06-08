#define  _GPU_FFT_FILTERING
#include "gpu_fft_filtering.h"
#include <iostream>
#include <cstring>


int32 main(int nargc, char *Cargv[])
{
   // declare variables
   int32  i=0,isize,numproc=0;
   int32  nread=0,nwrite=0,numt=0;
   int32  iret=0;
   int32  ngpus=0;
   int32  ncores=0;
   int64  loc=0;
   int32  nmin=0,nmax=0;
   float  *Fdata;
   char   Cin[256],Cout[256];
   char   *Cdata;
   char   Cehead[EHEAD],Cbhead[BHEAD];  // binary and ebcdic headers
   FILE   *Fin=NULL,*Fout=NULL;

   ngpus=guhasgpu();
   if (ngpus>0)
   {
      printf("GPU Count: (%d)\n",ngpus);
   }
   else
   {
      printf("err - no gpus found(%d)\n",ngpus);
      return(-1);
   }
   ncores=gucores();
   printf("Core Count: (%d)\n",ncores);



   // determine endian of machine   
   iendian = endian(1);

   // usage information
   if (nargc<=1)
   {
      std::cout << "Usage:  gpu_fft_filtering.exe infile outfile min max traces" << std::endl;
      std::cout << "" << std::endl;
      std::cout << "        infile      - SEGY IEEE 4byte Floating Point File, big or little endian" << std::endl;
      std::cout << "        outfile     - Signal Processed SEGY IEEE 4byte Floating Point File, same endian as input" << std::endl;
      std::cout << "        min         - minimum frequency in hertz to keep" << std::endl;
      std::cout << "        max         - maximum frequency in hertz to keep" << std::endl;
      std::cout << "        traces      - number of traces to read/write simultaneously" << std::endl;
      std::cout << "" << std::endl;
      std::cout << "        example: gpu_fft_filtering.exe ../data/Line_1_IEEE_Float_LE.sgy output.sgy 30 60 1000" << std::endl;
      std::cout << "                 - filters input to output 1000 traces at a time and only keeps" << std::endl;
      std::cout << "                   frequencies between 30 and 60 hertz" << std::endl;
      std::cout << "" << std::endl;
      return(1);
   }

   // parse command line args
   for(i=1; i<nargc; i++)
   {
      if (nargc<6)
      {  
         std::cout << "err - missing argument(s)" << std::endl;
         std::cout << "      run with no arguments to view usage information" << std::endl;
         std::cout << "" << std::endl;
         return(-1);
      }
      if (i==1)  
      {
         strcpy(Cin,Cargv[i]);
         std::cout << "Input File: " << Cin << std::endl;
      } 
      else if (i==2)  
      {
         strcpy(Cout,Cargv[i]);
         std::cout << "Output File: " << Cout << std::endl;
      } 
      else if (i==3)  
      {
         nmin = atoi(Cargv[i]);
         std::cout << "Minimum Frequency: " << nmin << std::endl;
      } 
      else if (i==4)  
      {
         nmax = atoi(Cargv[i]);
         std::cout << "Maximum Frequency: " << nmax << std::endl;
      } 
      else if (i==5)  
      {
         numt = atoi(Cargv[i]);
         std::cout << "Simultaneous Traces: " << numt << std::endl;
      } 
   }
   std::cout << "" << std::endl;

   // start timing
   etime();

   // open input file
   Fin = fopen(Cin,"rb");   // open for read binary 
   if (!Fin)
   {
      std::cout << "err - failed to open for read: " << Cin << std::endl;
      return(0);
   }
   std::cout << "Open for read: " << Cin << std::endl;
   // get file size
   lfilesz = fseeko(Fin,0,SEEK_END);
   lfilesz = ftello(Fin);
   rewind(Fin);

   // read and check segy headers
   iret = read_headers(Cehead,Cbhead,Fin);
   if (iret != 0)
   {
      return(iret);
   } 

   // data type okay so prepare output file with headers
   // open output file
   Fout = fopen(Cout,"wb");   // open for write binary 
   if (!Fout)
   {
      std::cout << "err - failed to open for write: " << Cout << std::endl;
      return(0);
   }
   std::cout << "Open for write: " << Cout << std::endl;
   
   // write output segy headers
   iret = write_headers(Cehead,Cbhead,Fout); 
   if (iret != 0)
   {
      return(iret);
   } 


   // set variables and allocate buffers to hold traces 
   // numt=1;
   // fscale=2.0f;
   isize=nsamp*sizeof(float);
   Cdata=(char*)calloc((numt*(isize+THEAD)),1);    // traces including header
   Fdata=(float*)calloc((numt*isize),1);         // just float samples from traces



   ///////////////////////////////////////////////
   //
   // loop through data until all file processed
   //
   ///////////////////////////////////////////////
   for (loc=3600; loc<lfilesz; )
   {
      // read in trace data, should be either big or little endian ieee floating point 4 byte data
      nread = read_trace(Cdata,numt,Fin);
      if (nread<0)  return(-1); 
      if (nread==0) break;                 // found end of file
      loc+=nread*(isize+THEAD);  // set location in file
      // transfer trace data to float buffer
      trace_2floatbuff(Cdata,Fdata,nread);
      if (numproc==0)
      {
         std::cout << "" << std::endl;
         std::cout << "First/Last 8 data samples for QC:" << std::endl;
         // output some before values
         for(i=0; i<8; i++)
         {
            printf("Input %d: %0.8f\n",i,Fdata[i]);
         }
         for(i=(numt*isize/sizeof(float))-8; (uint32)i<(numt*isize/sizeof(float)); i++)
         {
            printf("Input %d: %0.8f\n",i,Fdata[i]);
         }
      }

      // apply fft filtering processing
      std::cout << "#### gufftfilt_32f_i   (" << numt*isize << ")  " << loc << "/" << lfilesz << "\r" << std::flush;
     
      gufftfilt_32f_I(nmin,nmax,nsamp,nsint,Fdata,numt*isize);


      if (numproc==0)
      {
         // output some after values
         std::cout << "" << std::endl;
         for(i=0; i<8; i++)
         {
            printf("Output %d: %0.8f\n",i,Fdata[i]);
         }
         for(i=(numt*isize/sizeof(float))-8; (uint32)i<(numt*isize/sizeof(float)); i++)
         {
            printf("Output %d: %0.8f\n",i,Fdata[i]);
         }
         std::cout << "" << std::endl;
      }
      // transfer trace data to float buffer
      float_2tracebuff(Fdata,Cdata,nread);

      // write out data to output file
      nwrite = write_trace(Cdata,nread,Fout);
      if (nwrite<=0)  break; 
      numproc+=nwrite;
   }

   // free memory
   free(Cdata);
   free(Fdata);

   // close files
   if (Fin!=NULL)  fclose(Fin);
   if (Fout!=NULL) fclose(Fout);

   std::cout << "Total Traces Processed: "<< numproc << std::endl;
   std::cout << "Total Samples Processed (float numbers): "<< numproc*nsamp << std::endl;
   std::cout << "Elapsed Time: " << etime() << " microseconds\n";

   return(0);
}


