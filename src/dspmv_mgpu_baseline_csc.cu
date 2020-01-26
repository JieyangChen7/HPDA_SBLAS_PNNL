#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include <float.h>
//#include "anonymouslib_cuda.h"
#include "spmv_kernel.h"
using namespace std;

spmv_ret spMV_mgpu_baseline_csc(int m, int n, int nnz, double * alpha,
            double * cscVal, int * cscColPtr, int * cscRowIdx, 
            double * x, double * beta,
            double * y,
            int ngpu){

  double curr_time = 0.0;
  double numa_part_time = 0.0;
  double part_time = 0.0;
  double comp_time = 0.0;
  double comm_time = 0.0;
  double merg_time = 0.0;

  

  cudaStream_t * stream = new cudaStream_t [ngpu];
  cusparseHandle_t * handle = new cusparseHandle_t[ngpu];
  cusparseMatDescr_t * descr = new cusparseMatDescr_t[ngpu];

  int  * start_col  = new int[ngpu];
  int  * end_col    = new int[ngpu];
    
  int * dev_m            = new int      [ngpu];
  int * dev_n            = new int      [ngpu];
  int * dev_nnz          = new int      [ngpu];
  int ** host_cscColPtr  = new int    * [ngpu];
  int ** dev_cscColPtr   = new int    * [ngpu];
  int ** dev_cscRowIdx   = new int    * [ngpu];
  double ** dev_cscVal   = new double * [ngpu];

  int ** dev_csrRowPtr   = new int    * [ngpu];
  int ** dev_csrColIdx   = new int    * [ngpu];
  double ** dev_csrVal   = new double * [ngpu];

  double ** dev_x = new double * [ngpu];
  double ** dev_y = new double * [ngpu];

  double ** host_py = new double * [ngpu];

  double ** A = new double * [ngpu];
  int * lda = new int[ngpu];

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    cudaSetDevice(d);
    start_col[d] = floor((d)     * n / ngpu);
    end_col[d]   = floor((d + 1) * n / ngpu) - 1;
    dev_m[d]   = m;
    dev_n[d]   = end_col[d] - start_col[d] + 1;
    dev_nnz[d] = cscColPtr[end_col[d] + 1] - cscColPtr[start_col[d]];;
  }
  part_time += get_time() - curr_time;

  for (int d = 0; d < ngpu; d++) {
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaStreamCreate(&(stream[d])));
    checkCudaErrors(cusparseCreate(&(handle[d]))); 
    checkCudaErrors(cusparseSetStream(handle[d], stream[d]));
    checkCudaErrors(cusparseCreateMatDescr(&descr[d]));
    checkCudaErrors(cusparseSetMatType(descr[d],CUSPARSE_MATRIX_TYPE_GENERAL)); 
    checkCudaErrors(cusparseSetMatIndexBase(descr[d],CUSPARSE_INDEX_BASE_ZERO)); 

    cudaMallocHost((void**)&(A[d]), dev_m[d] * dev_n[d] * sizeof(double));
    lda[d] = m;
    checkCudaErrors(cudaMalloc((void**)&(dev_csrVal[d]),      dev_nnz[d]     * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&(dev_csrRowPtr[d]),   (dev_m[d] + 1) * sizeof(int)   ));
    checkCudaErrors(cudaMalloc((void**)&(dev_csrColIdx[d]),   dev_nnz[d]     * sizeof(int)   ));

    cudaMallocHost((void**)& host_py[d], dev_m[d] * sizeof(double));
    cudaMallocHost((void**)& host_cscColPtr[d], (dev_n[d]+1) * sizeof(int));

    checkCudaErrors(cudaMalloc((void**)&dev_cscColPtr[d], (dev_n[d] + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dev_cscRowIdx[d], dev_nnz[d] * sizeof(int))); 
    checkCudaErrors(cudaMalloc((void**)&dev_cscVal[d],    dev_nnz[d] * sizeof(double))); 

    checkCudaErrors(cudaMalloc((void**)&dev_x[d],           dev_n[d] * sizeof(double))); 
    checkCudaErrors(cudaMalloc((void**)&dev_y[d],           dev_m[d] * sizeof(double))); 

  }

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    for (int i = 0; i < dev_n[d] + 1; i++) {
      host_cscColPtr[d][i] = cscColPtr[start_col[d] + i] - cscColPtr[start_col[d]];
    }
  }
  part_time += get_time() - curr_time;

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    cudaSetDevice(d);
    checkCudaErrors(cudaMemcpyAsync(dev_cscColPtr[d], host_cscColPtr[d],                   (size_t)((dev_n[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice, stream[d]));
    checkCudaErrors(cudaMemcpyAsync(dev_cscRowIdx[d], &cscRowIdx[cscColPtr[start_col[d]]], (size_t)(dev_nnz[d] * sizeof(int)),     cudaMemcpyHostToDevice, stream[d])); 
    checkCudaErrors(cudaMemcpyAsync(dev_cscVal[d],    &cscVal[cscColPtr[start_col[d]]],    (size_t)(dev_nnz[d] * sizeof(double)),  cudaMemcpyHostToDevice, stream[d]));
    checkCudaErrors(cudaMemcpyAsync(dev_y[d],         y,                                   (size_t)(dev_m[d]*sizeof(double)),      cudaMemcpyHostToDevice, stream[d])); 
    checkCudaErrors(cudaMemcpyAsync(dev_x[d],         &x[start_col[d]],                    (size_t)(dev_n[d]*sizeof(double)),      cudaMemcpyHostToDevice, stream[d])); 
  }
  //time_comm = get_time() - curr_time;


  curr_time = get_time();
  for (int d = 0; d < ngpu; ++d) {
    cudaSetDevice(d);
    // csc2csrGPU(handle[d], m, n, nnz, A[d], lda[d],
    //             dev_cscVal[d], dev_cscColPtr[d], dev_cscRowIdx[d],
    //             dev_csrVal[d], dev_csrRowPtr[d], dev_csrColIdx[d]);
    checkCudaErrors(cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_TRANSPOSE, 
                               dev_n[d], dev_m[d], dev_nnz[d], 
                               alpha, descr[d], 
                               dev_cscVal[d], dev_cscColPtr[d], dev_cscRowIdx[d], 
                               dev_x[d], beta, dev_y[d]));       
    // checkCudaErrors(cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
    //                            dev_m[d], dev_n[d], dev_nnz[d], 
    //                            alpha, descr[d], dev_csrVal[d], 
    //                            dev_csrRowPtr[d], dev_csrColIdx[d], 
    //                            dev_x[d], beta, dev_y[d]));       
  }
  for (int d = 0; d < ngpu; ++d) {
    cudaSetDevice(d);
    cudaDeviceSynchronize();
  }
  comp_time = get_time() - curr_time;

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    cudaSetDevice(d);
    checkCudaErrors(cudaMemcpyAsync(host_py[d], dev_y[d], 
                    dev_m[d] * sizeof(double), cudaMemcpyDeviceToHost, stream[d])); 
  }
  for (int d = 0; d < ngpu; d++) {
    cudaSetDevice(d);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  for (int d = 0; d < ngpu; d++) {
    for (int i = 0; i < m; i++) {
      y[i] += host_py[d][i];
    }
  }
  merg_time = get_time() - curr_time;

  for (int d = 0; d < ngpu; d++) {
    cudaSetDevice(d);
    cudaFreeHost(A[d]);
    cudaFree(dev_csrVal[d]);
    cudaFree(dev_csrRowPtr[d]);
    cudaFree(dev_csrColIdx[d]);
    cudaFreeHost(host_py[d]);
    cudaFreeHost(host_cscColPtr[d]);

    cudaFree(dev_cscColPtr[d]);
    cudaFree(dev_cscRowIdx[d]);
    cudaFree(dev_cscVal[d]);

    cudaFree(dev_x[d]);
    cudaFree(dev_y[d]);

    cusparseDestroyMatDescr(descr[d]);
    cusparseDestroy(handle[d]);
    cudaStreamDestroy(stream[d]);
  }

  delete [] stream;
  delete [] handle;
  delete [] descr;
  delete [] start_col;
  delete [] end_col;
  delete [] dev_m;
  delete [] dev_n;
  delete [] dev_nnz;
  delete [] host_cscColPtr;
  delete [] dev_cscColPtr;
  delete [] dev_cscRowIdx;
  delete [] dev_cscVal;
  delete [] dev_csrRowPtr;
  delete [] dev_csrColIdx;
  delete [] dev_csrVal;
  delete [] dev_x;
  delete [] dev_y;
  delete [] host_py;
  delete [] A;
  delete [] lda;
    
  spmv_ret ret;
  ret.numa_part_time = numa_part_time;
  ret.part_time = part_time;
  ret.comp_time = comp_time;
  ret.comm_time = comm_time;
  ret.merg_time = merg_time;
  return ret;

}
