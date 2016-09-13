all: gpu

gpu: gpu_main.o cuda_utils.h
	nvcc gpu_main.o -o gpu -O2

gpu_main.o: gpu_main.cu
	nvcc gpu_main.cu -c -std=c++11 -I /usr/local/cuda/samples/common/inc/ -O2 -DNDEBUG
