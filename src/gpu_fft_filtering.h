#include <iostream>
/*
*/
#define EHEAD 3200
#define BHEAD 400
#define THEAD 240


#ifndef _GPU_TYPE
   #define _GPU_TYPE

   typedef   int8_t    int8;
   typedef  uint8_t   uint8;
   typedef  int16_t   int16;
   typedef uint16_t  uint16;
   typedef  int32_t   int32;
   typedef uint32_t  uint32;
   typedef  int64_t   int64;
   typedef uint64_t  uint64;

#endif


// globals declared here
#ifdef   _GPU_FFT_FILTERING

int32   netype,nbtype,nformat;
int32   iendian=-1;
int32   iswapd=0;
int32   ntrac,ntaux,nsamp,nsint,nrev,nrevd,nbpsamp,ntraces;
int64   lfilesz=0;
timeval tStart;

#else

extern  int32   netype,nbtype,nformat;
extern  int32   iendian;
extern  int32   iswapd;
extern  int32   ntrac,ntaux,nsamp,nsint,nrev,nrevd,nbpsamp,ntraces;
extern  int64   lfilesz;
extern  timeval tStart;

#endif

int32  endian(int32 idump);
int32  segy_check(char *Ce, char *Cb);
int16  bswaps(int16 snum);
int32  bswapl(int32 lnum);
float  bswapf(float fnum);
int32  read_trace(char *Cbuff,int32 numt,FILE *Fi);
int32  write_trace(char *Cbuff,int32 numt,FILE *Fo);
void   trace_2floatbuff(char *Cbuff,float *Fbuff,int32 numt);
void   float_2tracebuff(float *Fbuff,char *Cbuff,int32 numt);
int32  etime(void);
int32  segy_check(char *Ce, char *Cb);
int32  read_headers(char *Ce, char *Cb, FILE *Fi);
int32  write_headers(char *Ce, char *Cb, FILE *Fo);
void   trace_2floatbuff(char *Cbuff,float *Fbuff,int32 numt);
void   float_2tracebuff(float *Fbuff,char *Cbuff,int32 numt);

int32  guhasgpu(void);
int32  gucores(void);
void   gufftfilt_32f_I(int32 nmin,int32 nmax,int32 nsampst,int32 nsamprt,float *Fdata,int32 nsize);

