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

spmv_ret spMV_mgpu_baseline(int m, int n, int nnz, double * alpha,
            double * csrVal, int * csrRowPtr, int * csrColIndex, 
            double * x, double * beta,
            double * y,
            int ngpu){

  double curr_time = 0.0;
  double numa_part_time = 0.0;
  double part_time = 0.0;
  double comp_time = 0.0;
  double comm_time = 0.0;
  double merg_time = 0.0;

  cudaEvent_t * comp_start = new cudaEvent_t[ngpu];
  cudaEvent_t * comp_stop = new cudaEvent_t[ngpu];
  
  cudaEvent_t * comm_start = new cudaEvent_t[ngpu];
  cudaEvent_t * comm_stop = new cudaEvent_t[ngpu];

  curr_time = get_time();

  cudaStream_t * stream = new cudaStream_t [ngpu];

  cudaError_t * cudaStat1 = new cudaError_t[ngpu];
  cudaError_t * cudaStat2 = new cudaError_t[ngpu];
  cudaError_t * cudaStat3 = new cudaError_t[ngpu];
  cudaError_t * cudaStat4 = new cudaError_t[ngpu];
  cudaError_t * cudaStat5 = new cudaError_t[ngpu];
  cudaError_t * cudaStat6 = new cudaError_t[ngpu];

  cusparseStatus_t * status = new cusparseStatus_t[ngpu];
  cusparseHandle_t * handle = new cusparseHandle_t[ngpu];
  cusparseMatDescr_t * descr = new cusparseMatDescr_t[ngpu];

  int  * start_row  = new int[ngpu];
  int  * end_row    = new int[ngpu];
    
  int * dev_m            = new int      [ngpu];
  int * dev_n            = new int      [ngpu];
  int * dev_nnz          = new int      [ngpu];
  int ** host_csrRowPtr  = new int    * [ngpu];
  int ** dev_csrRowPtr   = new int    * [ngpu];
  int ** dev_csrColIndex = new int    * [ngpu];
  double ** dev_csrVal   = new double * [ngpu];


  double ** dev_x = new double * [ngpu];
  double ** dev_y = new double * [ngpu];

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++){
    start_row[d] = floor((d)     * m / ngpu);
    end_row[d]   = floor((d + 1) * m / ngpu) - 1;
    dev_m[d]   = end_row[d] - start_row[d] + 1;
    dev_n[d]   = n;
    dev_nnz[d] = (int)(csrRowPtr[end_row[d] + 1] - csrRowPtr[start_row[d]]);
  }
  part_time += get_time() - curr_time;
  
  for (int d = 0; d < ngpu; d++) {
    checkCudaErrors(cudaMallocHost((void**)& host_csrRowPtr[d], (dev_m[d]+1) * sizeof(int)));
  }

  curr_time = get_time();

  for (int d = 0; d < ngpu; d++){
    for (int i = 0; i < dev_m[d] + 1; i++) {
      host_csrRowPtr[d][i] = (int)(csrRowPtr[start_row[d] + i] - csrRowPtr[start_row[d]]);
    }

  }


  part_time += get_time() - curr_time;

  for (int d = 0; d < ngpu; d++){
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaStreamCreate(&(stream[d])));
    checkCudaErrors(cusparseCreate(&(handle[d]))); 
    checkCudaErrors(cusparseSetStream(handle[d], stream[d]));
    checkCudaErrors(cusparseCreateMatDescr(&descr[d]));
    checkCudaErrors(cusparseSetMatType(descr[d],CUSPARSE_MATRIX_TYPE_GENERAL)); 
    checkCudaErrors(cusparseSetMatIndexBase(descr[d],CUSPARSE_INDEX_BASE_ZERO)); 

    checkCudaErrors(cudaEventCreate(&(comp_start[d])));
    checkCudaErrors(cudaEventCreate(&(comp_stop[d])));
    checkCudaErrors(cudaEventCreate(&(comm_start[d])));
    checkCudaErrors(cudaEventCreate(&(comm_stop[d])));

    checkCudaErrors(cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dev_csrColIndex[d], dev_nnz[d] * sizeof(int))); 
    checkCudaErrors(cudaMalloc((void**)&dev_csrVal[d],      dev_nnz[d] * sizeof(double))); 
    checkCudaErrors(cudaMalloc((void**)&dev_x[d],           dev_n[d] * sizeof(double))); 
    checkCudaErrors(cudaMalloc((void**)&dev_y[d],           dev_m[d] * sizeof(double))); 
  }

  //curr_time = get_time();
  
  for (int d = 0; d < ngpu; d++){
    checkCudaErrors(cudaSetDevice(d));
    cudaEventRecord(comm_start[d], stream[d]);
    checkCudaErrors(cudaMemcpyAsync(dev_csrRowPtr[d],   host_csrRowPtr[d],                  (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice, stream[d]));
    checkCudaErrors(cudaMemcpyAsync(dev_csrColIndex[d], &csrColIndex[csrRowPtr[start_row[d]]], (size_t)(dev_nnz[d] * sizeof(int)),   cudaMemcpyHostToDevice, stream[d])); 
    checkCudaErrors(cudaMemcpyAsync(dev_csrVal[d],      &csrVal[csrRowPtr[start_row[d]]],      (size_t)(dev_nnz[d] * sizeof(double)), cudaMemcpyHostToDevice, stream[d]));
    checkCudaErrors(cudaMemcpyAsync(dev_y[d], &y[start_row[d]], (size_t)(dev_m[d]*sizeof(double)), cudaMemcpyHostToDevice, stream[d])); 
    checkCudaErrors(cudaMemcpyAsync(dev_x[d], x,                (size_t)(dev_n[d]*sizeof(double)), cudaMemcpyHostToDevice, stream[d])); 
    cudaEventRecord(comm_stop[d], stream[d]);
  }
  for (int d = 0; d < ngpu; ++d) {
    checkCudaErrors(cudaSetDevice(d));
    cudaEventRecord(comp_start[d], stream[d]);
    checkCudaErrors(cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
                   dev_m[d], dev_n[d], dev_nnz[d], 
                   alpha, descr[d], dev_csrVal[d], 
                   dev_csrRowPtr[d], dev_csrColIndex[d], 
                   dev_x[d], beta, dev_y[d]));
    cudaEventRecord(comp_stop[d], stream[d]);     
  }

  for (int d = 0; d < ngpu; ++d) {
    checkCudaErrors(cudaSetDevice(d));
    cudaEventSynchronize(comm_stop[d]);
    float elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, comm_start[d], comm_stop[d]);
    elapsedTime /= 1000.0;
    if (elapsedTime > comm_time) comm_time = elapsedTime;

    

    cudaEventSynchronize(comp_stop[d]);
    elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, comp_start[d], comp_stop[d]);
    elapsedTime /= 1000.0;
    if (elapsedTime > comp_time) comp_time = elapsedTime;

    printf("dev %d, elapsedTime1 %f comp_time %f\n", d, elapsedTime, comp_time);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  //comp_time = get_time() - curr_time;

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaMemcpyAsync( &y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]));
  }
  for (int d = 0; d < ngpu; d++) {
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  merg_time = get_time() - curr_time;

  for (int d = 0; d < ngpu; d++) {
    cudaSetDevice(d);
    cudaFree(dev_csrVal[d]);
    cudaFree(dev_csrRowPtr[d]);
    cudaFree(dev_csrColIndex[d]);
    cudaFree(dev_x[d]);
    cudaFree(dev_y[d]);
    cudaEventDestroy(comp_start[d]);
    cudaEventDestroy(comp_stop[d]);
    cudaEventDestroy(comm_start[d]);
    cudaEventDestroy(comm_stop[d]);
  }

  delete[] dev_csrVal;
  delete[] dev_csrRowPtr;
  delete[] dev_csrColIndex;
  delete[] dev_x;
  delete[] dev_y;
  delete[] host_csrRowPtr;
  delete[] start_row;
  delete[] end_row;
  delete[] comp_start;
  delete[] comp_stop;
  delete[] comm_start;
  delete[] comm_stop;
  
  spmv_ret ret;
  ret.numa_part_time = numa_part_time;
  ret.part_time = part_time;
  ret.comp_time = comp_time;
  ret.comm_time = comm_time;
  ret.merg_time = merg_time;
  return ret;

}
