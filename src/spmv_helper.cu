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


void csr2csrNcsc(int m, int n, int nnz,
             double * cooVal, int * cooRowIdx, int * cooColIdx,
             double * csrVal, int * csrRowPtr, int * csrColIdx,
             double * cscVal, int * cscColPtr, int * cscRowIdx) {

  double * A = new double[m * n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A[i * m + j] = 0.0;
    }
  }

  for (int i = 0; i < nnz; i++) {
    A[cooRowIdx[i] * m + cooColIdx[i]] = cooVal[i];
  }

  int p = 0;
  csrRowPtr[0] = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (A[i * m + j] != 0) {
        csrVal[p] = A[i * m + j];
        csrColIdx[p] = j;
        printf("add %f, %d\n", csrVal[p], csrColIdx[p]);
        p++;
      }
    }
    printf("add to row %d\n", i+1);
    csrRowPtr[i + 1] = p;
    printf("row %d\n", p);
  }

  printf("")

  // p = 0;
  // cscColPtr[0] = 0;
  // for (int j = 0; j < n; j++) {
  //   for (int i = 0; i < m; i++) {
  //     if (A[i * m + j] != 0) {
  //       cscVal[p] = A[i * m + j];
  //       cscRowIdx[p] = i;
  //       p++;
  //     }
  //   }
  //   cscColPtr[j + 1] = p;
  // }



  // double * dcsrVal;
  // int * dcsrRowPtr;
  // int * dcsrColIdx;
  // double * dcscVal;
  // int * dcscColPtr;
  // int * dcscRowIdx;


  // checkCudaErrors(cudaMalloc((void**)&dcsrVal, nnz * sizeof(double)));
  // checkCudaErrors(cudaMalloc((void**)&dcsrRowPtr, (m+1) * sizeof(int)));
  // checkCudaErrors(cudaMalloc((void**)&dcsrColIdx, nnz * sizeof(int)));

  // checkCudaErrors(cudaMalloc((void**)&dcscVal, nnz * sizeof(double)));
  // checkCudaErrors(cudaMalloc((void**)&dcscColPtr, (n+1) * sizeof(int)));
  // checkCudaErrors(cudaMalloc((void**)&dcscRowIdx, nnz * sizeof(int)));

  // checkCudaErrors(cudaMemcpy(dcsrVal, csrVal, nnz * sizeof(double),
  //                             cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(dcsrRowPtr, csrRowPtr, (m+1) * sizeof(int),
  //                             cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(dcsrColIdx, csrColIdx, nnz * sizeof(int),
  //                             cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(dcscVal, cscVal, nnz * sizeof(double),
  //                             cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(dcscColPtr, cscColPtr, (n+1) * sizeof(int),
  //                             cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(dcscRowIdx, cscRowIdx, nnz * sizeof(int),
  //                             cudaMemcpyHostToDevice));

  // csr2csc_gpu(m, n, nnz,
  //             dcsrVal, dcsrRowPtr, dcsrColIdx,
  //             dcscVal, dcscColPtr, dcscRowIdx);

  
  // checkCudaErrors(cudaMemcpy(csrVal, dcsrVal, nnz * sizeof(double),
  //                             cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaMemcpy(csrRowPtr, dcsrRowPtr, (m+1) * sizeof(int),
  //                             cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaMemcpy(csrColIdx, dcsrColIdx, nnz * sizeof(int),
  //                             cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaMemcpy(dcscVal, cscVal, nnz * sizeof(double),
  //                             cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaMemcpy(dcscColPtr, cscColPtr, (n+1) * sizeof(int),
  //                             cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaMemcpy(dcscRowIdx, cscRowIdx, nnz * sizeof(int),
  //                             cudaMemcpyDeviceToHost));

  // checkCudaErrors(cudaFree(dcsrVal));
  // checkCudaErrors(cudaFree(dcsrRowPtr));
  // checkCudaErrors(cudaFree(dcsrColIdx));
  // checkCudaErrors(cudaFree(dcscVal));
  // checkCudaErrors(cudaFree(dcscColPtr));
  // checkCudaErrors(cudaFree(dcscRowIdx));

}


void csr2csc(int m, int n, int nnz,
             double * csrVal, int * csrRowPtr, int * csrColIdx,
             double * cscVal, int * cscColPtr, int * cscRowIdx) {

  



  double * dcsrVal;
  int * dcsrRowPtr;
  int * dcsrColIdx;
  double * dcscVal;
  int * dcscColPtr;
  int * dcscRowIdx;


  checkCudaErrors(cudaMalloc((void**)&dcsrVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcsrRowPtr, (m+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcsrColIdx, nnz * sizeof(int)));

  checkCudaErrors(cudaMalloc((void**)&dcscVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcscColPtr, (n+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcscRowIdx, nnz * sizeof(int)));

  checkCudaErrors(cudaMemcpy(dcsrVal, csrVal, nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcsrRowPtr, csrRowPtr, (m+1) * sizeof(int),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcsrColIdx, csrColIdx, nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcscVal, cscVal, nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcscColPtr, cscColPtr, (n+1) * sizeof(int),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcscRowIdx, cscRowIdx, nnz * sizeof(int),
                              cudaMemcpyHostToDevice));

  csr2csc_gpu(m, n, nnz,
              dcsrVal, dcsrRowPtr, dcsrColIdx,
              dcscVal, dcscColPtr, dcscRowIdx);

  
  checkCudaErrors(cudaMemcpy(csrVal, dcsrVal, nnz * sizeof(double),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csrRowPtr, dcsrRowPtr, (m+1) * sizeof(int),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csrColIdx, dcsrColIdx, nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(dcscVal, cscVal, nnz * sizeof(double),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(dcscColPtr, cscColPtr, (n+1) * sizeof(int),
                              cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(dcscRowIdx, cscRowIdx, nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(dcsrVal));
  checkCudaErrors(cudaFree(dcsrRowPtr));
  checkCudaErrors(cudaFree(dcsrColIdx));
  checkCudaErrors(cudaFree(dcscVal));
  checkCudaErrors(cudaFree(dcscColPtr));
  checkCudaErrors(cudaFree(dcscRowIdx));

}



void csr2csc_gpu(int m, int n, int nnz,
                 double * csrVal, int * csrRowPtr, int * csrColIdx,
                 double * cscVal, int * cscColPtr, int * cscRowIdx) {
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

  double * A;
  int lda = m;
  checkCudaErrors(cudaMalloc((void**)&A, lda * n * sizeof(double)));

  int * nnzPerCol = new int[n];
  int * nnzTotalDevHostPtr = new int[1];
  checkCudaErrors(cusparseDcsr2dense(handle, m, n, descr,
                                     csrVal, csrRowPtr, csrColIdx,
                                     A, lda));

  checkCudaErrors(cusparseDnnz(handle, CUSPARSE_DIRECTION_COLUMN,
                               m, n, descr, A, lda, nnzPerCol, nnzTotalDevHostPtr));

  checkCudaErrors(cusparseDdense2csc(handle, m, n, descr, 
                                     A, lda, nnzPerCol,
                                     cscVal, cscColPtr, cscRowIdx));

  // checkCudaErrors(cusparseDcsr2csc(handle, m, n, nnz,
  //                                   csrVal, csrRowPtr, csrColIdx,
  //                                   cscVal, cscColPtr, cscRowIdx,
  //                                   CUSPARSE_ACTION_NUMERIC,
  //                                   CUSPARSE_INDEX_BASE_ZERO));

  checkCudaErrors(cudaDeviceSynchronize());
  // checkCudaErrors(cudaFree(buffer));
  // delete [] P;

  checkCudaErrors(cusparseDestroyMatDescr(descr));
  checkCudaErrors(cusparseDestroy(handle));
  checkCudaErrors(cudaStreamDestroy(stream));

  
}




