#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include <float.h>
#include "common_cuda.h"
//#include "anonymouslib_cuda.h"
#include "spmv_kernel.h"
#include <limits>
#include <sstream>
#include <string>
using namespace std;

int get_row_from_index(int n, int * a, int idx) {
  int l = 0;
  int r = n;
  int m = l + (r - l) / 2;
  while (l < r - 1) {
    //cout << "l = " << l <<endl;
    //cout << "r = " << r <<endl;
    int m = l + (r - l) / 2;
    //cout << "m = " << m <<endl;
    if (idx < a[m]) {
      r = m;
    } else if (idx >= a[m]) {
      l = m;
    } 
    //else {
    //  cout << "1st return: " << m << endl;
    //  return m;
    //}
  }
  //cout << "a[" << l << "] = " <<  a[l] << endl;
  //cout << " a[" << r << "] = " << a[r] << endl;
  //cout << " idx = " << idx << endl;
  if (idx == a[l]) return l;
  if (idx == a[r]) return r;
  return l;

}

double get_time()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  double ms = (double)tp.tv_sec * 1000 + (double)tp.tv_usec / 1000; //get current timestamp in milliseconds
  //return 0.00001;
  return ms / 1000;
}


double get_gpu_availble_mem(int ngpu) {
  size_t uCurAvailMemoryInBytes;
  size_t uTotalMemoryInBytes;
  

  double min_mem = numeric_limits<double>::max();
  int device;
  for (device = 0; device < ngpu; ++device) 
  {
    cudaSetDevice(device);
    cudaMemGetInfo(&uCurAvailMemoryInBytes, &uTotalMemoryInBytes);
    cout << uCurAvailMemoryInBytes << "/" << uTotalMemoryInBytes << endl;
    double aval_mem = (double)uCurAvailMemoryInBytes/1e9;
    cout << aval_mem << endl;
    if (aval_mem < min_mem) {
      min_mem = aval_mem;
    }
      // cudaDeviceProp deviceProp;
      // cudaGetDeviceProperties(&deviceProp, device);
      // printf("Device %d has compute capability %d.%d.\n",
      //        device, deviceProp.major, deviceProp.minor);
  }


  return min_mem;
}

void print_vec(double * a, int n, string s) {
  ostringstream ss;
  ss << s << ": ";
  for (int i = 0; i < n ; i++) {
    ss << a[i] << " ";
  }
  ss << endl;
  string res = ss.str();
  cout << res;
}

void print_vec(int * a, int n, string s) {
  ostringstream ss;
  ss << s << ": ";
  for (int i = 0; i < n ; i++) {
    ss << a[i] << " ";
  }
  ss << endl;
  string res = ss.str();
  cout << res;
}



void print_vec_gpu(double * a, int n, string s) {
  double * ha = new double[n]; 
  cudaMemcpy(ha, a, n*sizeof(double), cudaMemcpyDeviceToHost);
  print_vec(ha, n, s);
  delete [] ha;
}

void print_vec_gpu(int * a, int n, string s) {
  int * ha = new int[n];
  cudaMemcpy(ha, a, n*sizeof(int), cudaMemcpyDeviceToHost);
  print_vec(ha, n, s);
  delete [] ha;
}


void coo2csr(int m, int n, int nnz,
             double * cooVal, int * cooRowIdx, int * cooColIdx,
             double * csrVal, int * csrRowPtr, int * csrColIdx) {
  double * dcooVal;
  int * dcooRowIdx;
  int * dcooColIdx;
  double * dcsrVal;
  int * dcsrRowPtr;
  int * dcsrColIdx;

  checkCudaErrors(cudaMalloc((void**)&dcooVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcooRowIdx, nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcooColIdx, nnz * sizeof(int)));

  checkCudaErrors(cudaMalloc((void**)&dcsrVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcsrRowPtr, (m+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcsrColIdx, nnz * sizeof(int)));

  checkCudaErrors(cudaMemcpy(dcooVal, cooVal, nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooRowIdx, cooRowIdx, nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooColIdx, cooColIdx, nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcsrVal, csrVal, nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcsrRowPtr, csrRowPtr, (m+1) * sizeof(int),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcsrColIdx, csrColIdx, nnz * sizeof(int),
                              cudaMemcpyHostToDevice));

  coo2csr_gpu(m, n, nnz,
              dcooVal, dcooRowIdx, dcooColIdx,
              dcsrVal, dcsrRowPtr, dcsrColIdx);

  checkCudaErrors(cudaMemcpy(cooVal, dcooVal, nnz * sizeof(double),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cooRowIdx, dcooRowIdx, nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cooColIdx, dcooColIdx, nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csrVal, dcsrVal, nnz * sizeof(double),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csrRowPtr, dcsrRowPtr, (m+1) * sizeof(int),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csrColIdx, dcsrColIdx, nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(dcooVal));
  checkCudaErrors(cudaFree(dcooRowIdx));
  checkCudaErrors(cudaFree(dcooColIdx));
  checkCudaErrors(cudaFree(dcsrVal));
  checkCudaErrors(cudaFree(dcsrRowPtr));
  checkCudaErrors(cudaFree(dcsrColIdx));

}

void coo2csc(int m, int n, int nnz,
             double * cooVal, int * cooRowIdx, int * cooColIdx,
             double * cscVal, int * cscColPtr, int * cscRowIdx) {
  cudaStream_t stream;
  cusparseStatus_t status;
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  cudaStreamCreate(&stream);
  status = cusparseCreate(&handle); 
  if (status != CUSPARSE_STATUS_SUCCESS) 
  { 
    printf("CUSPARSE Library initialization failed");
    return; 
  } 
  status = cusparseSetStream(handle, stream);
  if (status != CUSPARSE_STATUS_SUCCESS) 
  { 
    printf("Stream bindind failed");
    return;
  } 
  status = cusparseCreateMatDescr(&descr);
  if (status != CUSPARSE_STATUS_SUCCESS) 
  { 
    printf("Matrix descriptor initialization failed");
    return;
  }   
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL); 
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);


  cusparseDestroyMatDescr(descr);
  cusparseDestroy(handle);
  cudaStreamDestroy(stream);

  
}

void coo2csr_gpu(int m, int n, int nnz,
                 double * cooVal, int * cooRowIdx, int * cooColIdx,
                 double * csrVal, int * csrRowPtr, int * csrColIdx) {
  cudaStream_t stream;
  cusparseStatus_t status;
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cusparseCreate(&handle)); 
  checkCudaErrors(cusparseSetStream(handle, stream));
  checkCudaErrors(cusparseCreateMatDescr(&descr));
  checkCudaErrors(cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL)); 
  checkCudaErrors(cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO));


  size_t buffer_size;
  checkCudaErrors(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz
                                                 cooRowIdx, cooColIdx,
                                                 &buffer_size));

  void * buffer;
  checkCudaErrors(cudaMalloc((void**)&buffer, buffer_size));

  int * P = new int[nnz];
  checkCudaErrors(cusparseXcoosortByRow(handle, m, n, nnz,
                                        cooRowIdx, cooColIdx,
                                        P, buffer));
  checkCudaErrors(cusparseXcoo2csr(handle,
                                   cooRowIdx, nnz, m, 
                                   CUSPARSE_INDEX_BASE_ZERO));
  checkCudaErrors(cudaMemcpy(csrVal, cooVal, nnz * sizeof(double), 
                              cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(csrColIdx, cooColIdx, nnz * sizeof(double), 
                              cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(buffer));
  delete [] P;

  checkCudaErrors(cusparseDestroyMatDescr(descr));
  checkCudaErrors(cusparseDestroy(handle));
  checkCudaErrors(cudaStreamDestroy(stream));

  
}

void csc2csr_gpu(int m, int n, int nnz,
                 double * cscVal, int * cscColPtr, int * cscRowIdx,
                 double * csrVal, int * csrRowPtr, int * csrColIdx) {
  cudaStream_t stream;
  cusparseStatus_t status;
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  cudaStreamCreate(&stream);
  status = cusparseCreate(&handle); 
  if (status != CUSPARSE_STATUS_SUCCESS) 
  { 
    printf("CUSPARSE Library initialization failed");
    return; 
  } 
  status = cusparseSetStream(handle, stream);
  if (status != CUSPARSE_STATUS_SUCCESS) 
  { 
    printf("Stream bindind failed");
    return;
  } 
  status = cusparseCreateMatDescr(&descr);
  if (status != CUSPARSE_STATUS_SUCCESS) 
  { 
    printf("Matrix descriptor initialization failed");
    return;
  }   
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL); 
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);


  cusparseDestroyMatDescr(descr);
  cusparseDestroy(handle);
  cudaStreamDestroy(stream);

  
}




