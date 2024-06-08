# GPU FFT Filtering 


## Background:
Seismic Processing relies on sound wave source being set off, (dynamite,
vibration, etc.), while geophones (on land) or hydrophones (on water) listen for
the sound refelecting off rock layers in the earth and returning to the surface
at various times/speeds. This information is "processed" to give us a picture of
what is underground. This information helps us to locate usually gas and oil. 

## Purpose:
The purpose of this project is to attempt to use GPU processing to 
use FFT transforms filter seismic data.I have designed this program to use an
FFT to transform the data to frequency and then only keep data between a user 
supplied minimum and maximum hertz range. Then transform the data back and 
write out the new data. 

	
## Data:
There are a multitude of formats that seismic data is recorded in, such
as SEGA, SEGB, SEGC, SEG2 and SEGY, possibly others. For ease of use I have 
chosen SEGY, which comes in a multitude of flavours. I will try and diagram it's
general formating below:

	EBCDIC Header	===
	                |  	- 3200 bytes   
			|	- EBCDIC or ASCII characters
			|       - Gives info about the data set usually
			===
	Binary Header   |	- 400 bytes
			|	- IBM ordered (Big Endian) or 
			|	  PC ordered (Little Endian)
			|	- Many different values in the header that gives  
			|	  info like number of traces, sample rate, 
			|         number of samples, sample format, etc
			|	- values in header maybe short, unsigned short,
			|	  long, unsigned long, etc.
			|
			===
	Trace1 Header   |	- 240 bytes
			|	- IBM ordered (Big Endian) or 
			|	  PC ordered (Little Endian)
			|	- Many different values stored like trace number,
			|	  trace count within file, type of trace, etc.
			|
			|
			|
			===
	Trace2 Data     |	- bytes = number of samples * sample format
			|	- sample formats include:
			|		4byte IEEE Float 
			|		8byte IEEE Float 
			|		4byte IBM  Float (IBM Native)
			|		4byte Integer
			|		3byte Integer
			|		2byte Short Integer
			|
			|	- also will include possible byte swapped 
			|	  versions of the above 
			|
			===
	Trace2 Header   |
			|
			|
			|
			|
			|
			===
	Trace2 Data     |
			|
			|
			|
			|
			|
			|
			|
			|
			|
			===
                         
	... this pattern of trace header/data continues until file is complete

- find more information about the SEGY standard at:

	https://en.wikipedia.org/wiki/SEG-Y

  I will not be dealing with multiple extended header samples nor 
  variable trace length data, I will assume all traces are the same
  length.

- for the purposes of this project I have chosen to support a smaller 
  subset of the standard. I have supplied data in both IEEE 4byte Float 
  formats native for  little endian for X86_64(Intel, AMD), aarch64(ARM,
  Apple M CPUs), PPC64LE(Power8/9) and big endian for PPC64BE(Power8/9),
  Power Mac(Motorola G3,4,5) also most ARM processors can run big endian
  as well. Find more info about endianess at:
 
	http://en.wikipedia.org/wiki/Endianess

Example 4byte integers as stored on a disk file in hexadecmal:

	Endian	Decimal Hexadecimal
        ------  ------- -----------
	Little  10	0A 00 00 00
	Big	10	00 00 00 0A

- it is faster to read and write floats that are native to your CPU,
  else you will have to byte swap every sample read in and written out.

				
## Running the Code:
- this code assumes the following:
	1) you have a working nvvv and g++ compiler
	2) you have the Nvidia CUDA SDK installed and working,
	   this includes npp libraries
	3) you have a supported GPU card 
	4) you have the correct matching drivers installed and 
	   working that match your CUDA SDK install

- in the src/ directory run:
 
	make clean build
	- this will remove the executable and build a new one
or      
	make all  
	- this will remove the executable and build a new one 
	  as well as install it in to the bin/ directory

- code maybe run from the src/ or bin/ directory as follows:

	program infile outfile multiplier traces

		infile     - input segy file IEEE 4byte Float
		outfile    - output segy file name 
		min        - minimum frequency in hertz to keep
		max        - maximum frequency in hertz to keep
		traces	   - number of simultaneous traces to read/write 
				    

	gpu_fft_filtering.exe ../data/sample_LE.sgy output.sgy 30 60 10000	

- for test purposes you may run the program with a scaler of 1.0, which 
  will not change the data at all and therefore the data will show no
  differences if compared using the diff command

- the program also prints out the first/last 8 samples of the first trace 
  of the input and output files to give a commandline QC that the scaler 
  is working correctly.  



## Results from runs tested:
- please find complete run logs in data/ directory
- as expected the run with a scaler value of 1 did not alter the data, a good check to 
  make sure things are working as expected -> run3.log
- runs that did more traces at once rather than one a a time ran faster -> run1.log vs run3.log
- scaler of 2 doubled that data samples and 0.5 halved the samples as expected -> run1.log vs run2.log
- identical runs where the only difference was that the data would have to be byte swapped 
  for one set but not for the other data set took longer as expected -> run1.log vs run4.log



