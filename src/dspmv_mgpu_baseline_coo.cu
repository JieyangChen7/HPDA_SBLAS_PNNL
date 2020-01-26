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


spmv_ret spMV_mgpu_baseline_coo(int m, int n, int nnz, double * alpha,
            double * cooVal, int * cooRowIdx, int * cooColIdx, 
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

  int  * start_row  = new int[ngpu];
  int  * end_row    = new int[ngpu];

  int  * start_idx  = new int[ngpu];
  int  * end_idx    = new int[ngpu];
    
  int * dev_m            = new int      [ngpu];
  int * dev_n            = new int      [ngpu];
  int * dev_nnz          = new int      [ngpu];
  
  int ** host_cooRowIdx  = new int    * [ngpu];

  int ** dev_cooRowIdx   = new int    * [ngpu];
  int ** dev_cooColIdx   = new int    * [ngpu];
  double ** dev_cooVal   = new double * [ngpu];

  int ** dev_csrRowPtr   = new int    * [ngpu];
  int ** dev_csrColIdx   = new int    * [ngpu];
  double ** dev_csrVal   = new double * [ngpu];

  double ** dev_x = new double * [ngpu];
  double ** dev_y = new double * [ngpu];

  double ** host_py = new double * [ngpu];

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    checkCudaErrors(cudaSetDevice(d));
    start_row[d] = floor((d)     * m / ngpu);
    end_row[d]   = floor((d + 1) * m / ngpu) - 1;
    start_idx[d] = findFirstInSorted(cooRowIdx, nnz, start_row[d]);
    end_idx[d]   = findLastInSorted(cooRowIdx, nnz, end_row[d]);
    dev_m[d]   = end_row[d] - start_row[d] + 1;
    dev_n[d]   = n;
    dev_nnz[d] = end_idx[d] - start_idx[d] + 1;
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

    checkCudaErrors(cudaMalloc((void**)&(dev_csrVal[d]),      dev_nnz[d]     * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&(dev_csrRowPtr[d]),   (dev_m[d] + 1) * sizeof(int)   ));
    checkCudaErrors(cudaMalloc((void**)&(dev_csrColIdx[d]),   dev_nnz[d]     * sizeof(int)   ));

    checkCudaErrors(cudaMallocHost((void**)& (host_py[d]), dev_m[d] * sizeof(double)));
    checkCudaErrors(cudaMallocHost((void**)& (host_cooRowIdx[d]), dev_nnz[d] * sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&dev_cooVal[d],    dev_nnz[d] * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&dev_cooRowIdx[d], dev_nnz[d] * sizeof(int))); 
    checkCudaErrors(cudaMalloc((void**)&dev_cooColIdx[d], dev_nnz[d] * sizeof(int))); 

    checkCudaErrors(cudaMalloc((void**)&dev_x[d],           dev_n[d] * sizeof(double))); 
    checkCudaErrors(cudaMalloc((void**)&dev_y[d],           dev_m[d] * sizeof(double))); 

  }

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    for (int i = 0; i < dev_nnz[d]; i++) {
      host_cooRowIdx[d][i] = cooRowIdx[start_idx[d]] - start_row[d];
    }
  }
  part_time += get_time() - curr_time;

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaMemcpyAsync(dev_cooVal[d],    &(cooVal[start_idx[d]]),          dev_nnz[d] * sizeof(double), cudaMemcpyHostToDevice, stream[d]));
    checkCudaErrors(cudaMemcpyAsync(dev_cooRowIdx[d], &host_cooRowIdx[start_idx[d]], dev_nnz[d] * sizeof(int),    cudaMemcpyHostToDevice, stream[d])); 
    checkCudaErrors(cudaMemcpyAsync(dev_cooColIdx[d], &cooColIdx[start_idx[d]],      dev_nnz[d] * sizeof(int), cudaMemcpyHostToDevice, stream[d]));
    checkCudaErrors(cudaMemcpyAsync(dev_y[d],         &y[start_row[d]],                 dev_m[d]*sizeof(double),     cudaMemcpyHostToDevice, stream[d])); 
    checkCudaErrors(cudaMemcpyAsync(dev_x[d],         x,                             dev_n[d]*sizeof(double),     cudaMemcpyHostToDevice, stream[d])); 
  }
  //time_comm = get_time() - curr_time;


  //curr_time = get_time();
  for (int d = 0; d < ngpu; ++d) {
    checkCudaErrors(cudaSetDevice(d));
    coo2csr_gpu(handle[d], stream[d], dev_m[d], dev_n[d], dev_nnz[d],
                dev_cooVal[d], dev_cooRowIdx[d], dev_cooColIdx[d],
                dev_csrVal[d], dev_csrRowPtr[d], dev_csrColIdx[d]);
    // checkCudaErrors(cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
    //                            dev_m[d], dev_n[d], dev_nnz[d], 
    //                            alpha, descr[d], dev_csrVal[d], 
    //                            dev_csrRowPtr[d], dev_csrColIdx[d], 
    //                            dev_x[d], beta, dev_y[d]));       
  }
  for (int d = 0; d < ngpu; ++d) {
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  comp_time = get_time() - curr_time;

  curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaMemcpyAsync(&y[start_row[d]],   dev_y[d], dev_m[d]*sizeof(double),     cudaMemcpyDeviceToHost, stream[d])); 
    //checkCudaErrors(cudaMemcpyAsync(host_py[d], dev_y[d], 
                    //dev_m[d] * sizeof(double), cudaMemcpyDeviceToHost, stream[d])); 
  }
  for (int d = 0; d < ngpu; d++) {
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  // for (int d = 0; d < ngpu; d++) {
  //   for (int i = 0; i < dev_m[d]; i++) {
  //     y[start_row[d] + i] += host_py[d][i];
  //   }
  // }
  merg_time = get_time() - curr_time;

  for (int d = 0; d < ngpu; d++) {
    cudaFree(dev_csrVal[d]);
    cudaFree(dev_csrRowPtr[d]);
    cudaFree(dev_csrColIdx[d]);
    cudaFreeHost(host_py[d]);
    cudaFreeHost(host_cooRowIdx[d]);

    cudaFree(dev_cooColIdx[d]);
    cudaFree(dev_cooRowIdx[d]);
    cudaFree(dev_cooVal[d]);

    cudaFree(dev_x[d]);
    cudaFree(dev_y[d]);

    cusparseDestroyMatDescr(descr[d]);
    cusparseDestroy(handle[d]);
    cudaStreamDestroy(stream[d]);
  }

  delete [] stream;
  delete [] handle;
  delete [] descr;
  delete [] start_row;
  delete [] end_row;
  delete [] dev_m;
  delete [] dev_n;
  delete [] dev_nnz;
  delete [] host_cooRowIdx;
  delete [] dev_cooColIdx;
  delete [] dev_cooRowIdx;
  delete [] dev_cooVal;
  delete [] dev_csrRowPtr;
  delete [] dev_csrColIdx;
  delete [] dev_csrVal;
  delete [] dev_x;
  delete [] dev_y;
  delete [] host_py;
    
  spmv_ret ret;
  ret.numa_part_time = numa_part_time;
  ret.part_time = part_time;
  ret.comp_time = comp_time;
  ret.comm_time = comm_time;
  ret.merg_time = merg_time;
  return ret;

}
