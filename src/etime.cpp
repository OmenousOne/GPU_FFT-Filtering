#include "gpu_fft_filtering.h"
#include <sys/time.h>

int32 etime(void) 
{
   timeval tEnd;
   int32   t;
 
   gettimeofday(&tEnd, 0);
   t = (tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec;
   tStart = tEnd;
   return t;
}
