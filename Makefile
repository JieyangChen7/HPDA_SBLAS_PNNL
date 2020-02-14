#compilers

#ENVIRONMENT_PARAMETERS
#CUDA_INSTALL_PATH ?= /usr/local/cuda
#CUDA_SAMPLES_PATH ?= /usr/local/cuda/samples


#CUDA_PARAMETERS
NVCC_FLAGS = -O3  -w -m64 -gencode=arch=compute_60,code=compute_60 --default-stream per-thread
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SAMPLES_PATH)/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse -Xcompiler -fopenmp 
INC = -I ../include -I ../include/detail/cuda

.PHONY: all lib test clean

.DEFAULT_GOAL := all

all: lib test

test: test_smt test_dgx1 test_dgx2

lib: dspmv_mgpu_v2.o dspmv_mgpu_v1.o dspmv_mgpu_v1_numa.o dspmv_mgpu_v1_numa_csc.o dspmv_mgpu_v1_numa_coo.o dspmv_mgpu_baseline.o dspmv_mgpu_baseline_csc.o dspmv_mgpu_baseline_coo.o csr5_kernel.o spmv_helper.o

test_smt: lib dspmv_test_smt.o 
	(cd test && nvcc -ccbin g++ $(NVCC_FLAGS) ../src/csr5_kernel.o ../src/spmv_helper.o dspmv_test_smt.o ../src/dspmv_mgpu_baseline.o ../src/dspmv_mgpu_baseline_csc.o ../src/dspmv_mgpu_baseline_coo.o ../src/dspmv_mgpu_v1.o ../src/dspmv_mgpu_v1_numa.o ../src/dspmv_mgpu_v1_numa_csc.o ../src/dspmv_mgpu_v1_numa_coo.o ../src/dspmv_mgpu_v2.o -o test_spmv_smt $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN))
	#(cd test && nvcc -ccbin g++ $(NVCC_FLAGS) ../src/csr5_kernel.o ../src/spmv_helper.o dspmspv_test.o ../src/dspmv_mgpu_baseline.o ../src/dspmv_mgpu_v1.o ../src/dspmv_mgpu_v2.o -o test_spmspv $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN))
test_dgx1: lib dspmv_test_dgx1.o 
	(cd test && nvcc -ccbin g++ $(NVCC_FLAGS) ../src/csr5_kernel.o ../src/spmv_helper.o dspmv_test_dgx1.o ../src/dspmv_mgpu_baseline.o ../src/dspmv_mgpu_baseline_csc.o ../src/dspmv_mgpu_baseline_coo.o ../src/dspmv_mgpu_v1.o ../src/dspmv_mgpu_v1_numa.o ../src/dspmv_mgpu_v1_numa_csc.o ../src/dspmv_mgpu_v1_numa_coo.o ../src/dspmv_mgpu_v2.o -o test_spmv_dgx1 $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN))
	
test_dgx2: lib dspmv_test_dgx2.o 
	(cd test && nvcc -ccbin g++ $(NVCC_FLAGS) ../src/csr5_kernel.o ../src/spmv_helper.o dspmv_test_dgx1.o ../src/dspmv_mgpu_baseline.o ../src/dspmv_mgpu_baseline_csc.o ../src/dspmv_mgpu_baseline_coo.o ../src/dspmv_mgpu_v1.o ../src/dspmv_mgpu_v1_numa.o ../src/dspmv_mgpu_v1_numa_csc.o ../src/dspmv_mgpu_v1_numa_coo.o ../src/dspmv_mgpu_v2.o -o test_spmv_dgx2 $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN))

dspmv_mgpu_v2.o: ./src/dspmv_mgpu_v2.cu 
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_mgpu_v2.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_mgpu_v1.o: ./src/dspmv_mgpu_v1.cu 
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_mgpu_v1.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_mgpu_v1_numa.o: ./src/dspmv_mgpu_v1_numa.cu
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_mgpu_v1_numa.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_mgpu_v1_numa_csc.o: ./src/dspmv_mgpu_v1_numa_csc.cu
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_mgpu_v1_numa_csc.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_mgpu_v1_numa_coo.o: ./src/dspmv_mgpu_v1_numa_coo.cu
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_mgpu_v1_numa_coo.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_mgpu_baseline.o: ./src/dspmv_mgpu_baseline.cu 
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_mgpu_baseline.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_mgpu_baseline_csc.o: ./src/dspmv_mgpu_baseline_csc.cu 
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_mgpu_baseline_csc.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_mgpu_baseline_coo.o: ./src/dspmv_mgpu_baseline_coo.cu 
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_mgpu_baseline_coo.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_test_smt.o: ./test/dspmv_test_smt.cu 
	(cd test && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_test_smt.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_test_dgx1.o: ./test/dspmv_test_dgx1.cu 
	(cd test && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_test_dgx1.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

dspmv_test_dgx2.o: ./test/dspmv_test_dgx2.cu 
	(cd test && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmv_test_dgx2.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

#dspmspv_test.o: ./test/dspmspv_test.cu
#	(cd test && nvcc -ccbin g++ -c $(NVCC_FLAGS) dspmspv_test.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

csr5_kernel.o: ./src/csr5_kernel.cu
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) csr5_kernel.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

spmv_helper.o: ./src/spmv_helper.cu 
	(cd src && nvcc -ccbin g++ -c $(NVCC_FLAGS) spmv_helper.cu $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS))

clean:
	(cd src && rm *.o)
	(cd test && rm *.o)
	(cd test && rm test_spmv)
