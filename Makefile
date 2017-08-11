COMMON_FLAGS=-O2

all:
	nvcc $(COMMON_FLAGS) -o cublas_gemm cublas_gemm.cpp -lcudart -lcublas

clean:
	rm cublas_gemm
