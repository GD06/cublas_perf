COMMON_FLAGS=-O4

all:
	nvcc $(COMMON_FLAGS) -o cublas_gemm cublas_gemm.cpp -lcudart -lcublas
