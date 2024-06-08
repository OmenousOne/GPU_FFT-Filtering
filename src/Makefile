IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-g  -I$(IDIR) -I/usr/local/cuda/include -lcuda -lcudart -lcufft  --std c++11

.PHONY: clean build run

build: gpu_fft_filtering.cpp endian.cpp bswaps.cpp bswapl.cpp bswapf.cpp etime.cpp read_headers.cpp write_headers.cpp read_trace.cpp write_trace.cpp segy_check.cpp float_2tracebuff.cpp trace_2floatbuff.cpp guhasgpu.cu gucores.cu gufftfilt_32f_i.cu 
	$(COMPILER) $(COMPILER_FLAGS) gpu_fft_filtering.cpp endian.cpp bswaps.cpp bswapl.cpp bswapf.cpp etime.cpp read_headers.cpp write_headers.cpp read_trace.cpp write_trace.cpp segy_check.cpp float_2tracebuff.cpp trace_2floatbuff.cpp guhasgpu.cu gucores.cu gufftfilt_32f_i.cu -o gpu_fft_filtering.exe -Wno-deprecated-gpu-targets

clean:
	rm -f gpu_fft_filtering.exe

install:
	cp -fv gpu_fft_filtering.exe ../bin

all: clean build install

