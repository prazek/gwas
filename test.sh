#!/usr/bin/env bash

# ~/llvm-build-release/bin/clang gpu_main.cu -o gpu --cuda-gpu-arch=sm_20 -L/usr/local/cuda/lib64/ -lcudart_static -ldl -lrt -pthread -std=c++11 -I /usr/local/cuda/samples/common/inc/ &&

make &&
time ./gpu < test.in > out.tmp &&
vimdiff out.tmp test.out
