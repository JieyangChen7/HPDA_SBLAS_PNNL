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
#include <string>
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

  cudaEvent_t * comp_start = new cudaEvent_t[ngpu];
  cudaEvent_t * comp_stop = new cudaEvent_t[ngpu];
  
  cudaEvent_t * comm_start = new cudaEvent_t[ngpu];
  cudaEvent_t * comm_stop = new cudaEvent_t[ngpu];

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

    checkCudaErrors(cudaEventCreate(&(comp_start[d])));
    checkCudaErrors(cudaEventCreate(&(comp_stop[d])));
    checkCudaErrors(cudaEventCreate(&(comm_start[d])));
    checkCudaErrors(cudaEventCreate(&(comm_stop[d])));

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
      host_cooRowIdx[d][i] = cooRowIdx[start_idx[d] + i] - start_row[d];
    }
  }
  part_time += get_time() - curr_time;

  //curr_time = get_time();
  for (int d = 0; d < ngpu; d++) {
    // print_vec(&(cooVal[start_idx[d]]), dev_nnz[d], "cooVal"+to_string(d));
    // print_vec(host_cooRowIdx[d], dev_nnz[d], "cooRowIdx"+to_string(d));
    // print_vec(&cooColIdx[start_idx[d]], dev_nnz[d], "cooColIdx"+to_string(d));
    // print_vec(&y[start_row[d]], dev_m[d], "y"+to_string(d));
    // print_vec(x, dev_n[d], "x"+to_string(d));
    // printf("dev_id %d, m=%d, n=%d, nnz=%d, start_idx=%d, end_idx=%d, start_row=%d, end_row=%d\n", 
    //         d, dev_m[d], dev_n[d], dev_nnz[d], start_idx[d], end_idx[d], start_row[d], end_row[d]);


    checkCudaErrors(cudaSetDevice(d));
    cudaEventRecord(comm_start[d], stream[d]);
    checkCudaErrors(cudaMemcpyAsync(dev_cooVal[d],    &(cooVal[start_idx[d]]),          dev_nnz[d] * sizeof(double), cudaMemcpyHostToDevice, stream[d]));
    checkCudaErrors(cudaMemcpyAsync(dev_cooRowIdx[d], host_cooRowIdx[d],             dev_nnz[d] * sizeof(int),    cudaMemcpyHostToDevice, stream[d])); 
    checkCudaErrors(cudaMemcpyAsync(dev_cooColIdx[d], &cooColIdx[start_idx[d]],      dev_nnz[d] * sizeof(int), cudaMemcpyHostToDevice, stream[d]));
    checkCudaErrors(cudaMemcpyAsync(dev_y[d],         &y[start_row[d]],                 dev_m[d]*sizeof(double),     cudaMemcpyHostToDevice, stream[d])); 
    checkCudaErrors(cudaMemcpyAsync(dev_x[d],         x,                             dev_n[d]*sizeof(double),     cudaMemcpyHostToDevice, stream[d])); 
    cudaEventRecord(comm_stop[d], stream[d]);
    // checkCudaErrors(cudaDeviceSynchronize());
    // print_vec_gpu(dev_cooVal[d], dev_nnz[d], "dev_cooVal"+to_string(d));
    // print_vec_gpu(dev_cooRowIdx[d], dev_nnz[d], "dev_cooRowIdx"+to_string(d));
    // print_vec_gpu(dev_cooColIdx[d], dev_nnz[d], "dev_cooColIdx"+to_string(d));
    // print_vec_gpu(dev_y[d], dev_m[d], "dev_y"+to_string(d));
    // print_vec_gpu(dev_x[d], dev_n[d], "dev_x"+to_string(d));
    

  }
  //time_comm = get_time() - curr_time;


  //curr_time = get_time();
  for (int d = 0; d < ngpu; ++d) {
    checkCudaErrors(cudaSetDevice(d));
    
    // coo2csrGPU(handle[d], stream[d], dev_m[d], dev_n[d], dev_nnz[d],
    //             dev_cooVal[d], dev_cooRowIdx[d], dev_cooColIdx[d],
    //             dev_csrVal[d], dev_csrRowPtr[d], dev_csrColIdx[d]);
    // checkCudaErrors(cudaDeviceSynchronize());
    // print_vec_gpu(dev_csrVal[d], dev_nnz[d], "dev_csrVal"+to_string(d));
    // print_vec_gpu(dev_csrRowPtr[d], dev_m[d]+1, "dev_csrRowPtr"+to_string(d));
    // print_vec_gpu(dev_csrColIdx[d], dev_nnz[d], "dev_csrColIdx"+to_string(d));

    checkCudaErrors(cudaEventRecord(comp_start[d], stream[d]));
    // checkCudaErrors(cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
    //                            dev_m[d], dev_n[d], dev_nnz[d], 
    //                            alpha, descr[d], dev_csrVal[d], 
    //                            dev_csrRowPtr[d], dev_csrColIdx[d], 
    //                            dev_x[d], beta, dev_y[d]));    
    checkCudaErrors(cudaEventRecord(comp_stop[d], stream[d]));

  } 

  for (int d = 0; d < ngpu; ++d) {
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaEventSynchronize(comm_stop[d]));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, comm_start[d], comm_stop[d]));
    elapsedTime /= 1000.0;
    if (elapsedTime > comm_time) comm_time = elapsedTime;

    // printf("dev %d, comm_time %f elapsedTime %f\n", d, comm_time, elapsedTime);

    checkCudaErrors(cudaEventSynchronize(comp_stop[d]));
    elapsedTime = 0;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, comp_start[d], comp_stop[d]));
    elapsedTime /= 1000.0;
    if (elapsedTime > comp_time) comp_time = elapsedTime;
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  //comp_time = get_time() - curr_time;

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
    checkCudaErrors(cudaSetDevice(d));
    checkCudaErrors(cudaFree(dev_csrVal[d]));
    checkCudaErrors(cudaFree(dev_csrRowPtr[d]));
    checkCudaErrors(cudaFree(dev_csrColIdx[d]));
    checkCudaErrors(cudaFreeHost(host_py[d]));
    checkCudaErrors(cudaFreeHost(host_cooRowIdx[d]));

    checkCudaErrors(cudaFree(dev_cooColIdx[d]));
    checkCudaErrors(cudaFree(dev_cooRowIdx[d]));
    checkCudaErrors(cudaFree(dev_cooVal[d]));

    checkCudaErrors(cudaFree(dev_x[d]));
    checkCudaErrors(cudaFree(dev_y[d]));

    checkCudaErrors(cusparseDestroyMatDescr(descr[d]));
    checkCudaErrors(cusparseDestroy(handle[d]));
    checkCudaErrors(cudaStreamDestroy(stream[d]));

    checkCudaErrors(cudaEventDestroy(comp_start[d]));
    checkCudaErrors(cudaEventDestroy(comp_stop[d]));
    checkCudaErrors(cudaEventDestroy(comm_start[d]));
    checkCudaErrors(cudaEventDestroy(comm_stop[d]));
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
