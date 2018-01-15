PROJECT=gemm
CC=g++

CUDA_DIR=/usr/local/cuda
BLAS_DIR=/opt/local
INCLUDE=-I. -I$(CUDA_DIR)/include
LIB=-L$(CUDA_DIR)/lib -lcudart
FLAGS=-std=c++11 -framework Accelerate

TEST_BIN=test

.PHONY: test

test:
	$(CC) $(FLAGS) $(INCLUDE) $(TEST_BIN).cc sgemm.cc -o $(TEST_BIN) $(LIB)

clean:
	rm -rf $(TEST_BIN)
