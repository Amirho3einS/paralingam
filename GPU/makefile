CFLAGS = -maxrregcount=64
CC = nvcc

ParaLiNGAM: main.cpp direct_lingam.cu gpu_kernels.o
	$(CC) $(CFLAGS) -o ParaLiNGAM gpu_kernels.o direct_lingam.cu main.cpp
gpu_kernels.o: gpu_kernels.cu
	$(CC) $(CFLAGS) --device-c gpu_kernels.cu

clean:
	rm -f ParaLiNGAM *.o 