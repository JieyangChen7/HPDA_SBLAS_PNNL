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

  printf("converting: %d, %d\n", m, n);

  sortCOORow(m, n, nnz, cooVal, cooRowIdx, cooColIdx);


  // double * A = new double[m * n];
  // for (int i = 0; i < m; i++) {
  //   for (int j = 0; j < n; j++) {
  //     A[i * n + j] = 0.0;
  //   }
  // }

  // for (int i = 0; i < nnz; i++) {
  //   A[cooRowIdx[i] * n + cooColIdx[i]] = cooVal[i];
  // }

  // for (int i = 0; i < m; i++) {
  //   print_vec(&(A[i * n]), n, "A");
  // }

  // int p = 0;
  // csrRowPtr[0] = 0;
  // for (int i = 0; i < m; i++) {
  //   for (int j = 0; j < n; j++) {
  //     if (A[i * n + j] != 0) {
  //       csrVal[p] = A[i * n + j];
  //       csrColIdx[p] = j;
  //       //printf("add %f, %d\n", csrVal[p], csrColIdx[p]);
  //       p++;
  //     }
  //   }
  //   //printf("add to row %d\n", i+1);
  //   csrRowPtr[i + 1] = p;
  //   //printf("row %d\n", p);
  // }

  

  // p = 0;
  // cscColPtr[0] = 0;
  // for (int j = 0; j < n; j++) {
  //   for (int i = 0; i < m; i++) {
  //     if (A[i * n + j] != 0) {
  //       cscVal[p] = A[i * n + j];
  //       cscRowIdx[p] = i;
  //       p++;
  //     }
  //   }
  //   cscColPtr[j + 1] = p;
  // }

  // printf("done converting\n");
  //delete [] A;

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


void csc2csrGPU(cusparseHandle_t handle, int m, int n, int nnz, double * A, int lda, 
                 double * cscVal, int * cscColPtr, int * cscRowIdx,
                 double * csrVal, int * csrRowPtr, int * csrColIdx) {
  
  cusparseMatDescr_t descr;
  checkCudaErrors(cusparseCreateMatDescr(&descr));
  checkCudaErrors(cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL)); 
  checkCudaErrors(cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO));

  int * nnzPerRow = new int[m];
  int * nnzTotalDevHostPtr = new int[1];
  checkCudaErrors(cusparseDcsc2dense(handle, m, n, descr,
                                     cscVal, cscRowIdx, cscColPtr,
                                     A, lda));

  checkCudaErrors(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW,
                               m, n, descr, A, lda, nnzPerRow, nnzTotalDevHostPtr));

  checkCudaErrors(cusparseDdense2csr(handle, m, n, descr, 
                                     A, lda, nnzPerRow,
                                     csrVal, csrRowPtr, csrColIdx));

  // checkCudaErrors(cusparseDcsr2csc(handle, m, n, nnz,
  //                                   csrVal, csrRowPtr, csrColIdx,
  //                                   cscVal, cscColPtr, cscRowIdx,
  //                                   CUSPARSE_ACTION_NUMERIC,
  //                                   CUSPARSE_INDEX_BASE_ZERO));

  // checkCudaErrors(cudaDeviceSynchronize());
  // checkCudaErrors(cudaFree(buffer));
  // delete [] P;

  // checkCudaErrors(cusparseDestroyMatDescr(descr));
  // checkCudaErrors(cusparseDestroy(handle));
  // checkCudaErrors(cudaStreamDestroy(stream));

  
}

void sortCOORow(int m, int n, int nnz,
                double * cooVal, int * cooRowIdx, int * cooColIdx) {
  cudaStream_t stream;
  cusparseStatus_t status;
  cusparseHandle_t handle;

  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cusparseCreate(&handle)); 
  checkCudaErrors(cusparseSetStream(handle, stream));

  double * dcooVal;
  int * dcooRowIdx;
  int * dcooColIdx;
  double * dcooValSorted;
  void * buffer;
  size_t bufferSize = 0;
  int * dP;

  checkCudaErrors(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, 
                                                  dcooRowIdx, dcooColIdx,
                                                  &bufferSize));

  checkCudaErrors(cudaMalloc((void**)&dcooVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcooRowIdx, nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcooColIdx, nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcooValSorted, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&buffer, bufferSize ));
  checkCudaErrors(cudaMalloc((void**)&dP, nnz * sizeof(int)));

  checkCudaErrors(cudaMemcpy(dcooVal, cooVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooRowIdx, cooRowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooColIdx, cooColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));

  checkCudaErrors(cusparseCreateIdentityPermutation(handle, nnz, dP));
  
  checkCudaErrors(cusparseXcoosortByRow(handle, m, n, nnz, 
                                        dcooRowIdx, dcooColIdx,
                                        dP, buffer));
  checkCudaErrors(cusparseDgthr(handle, nnz, dcooVal, dcooValSorted, dP,
                                CUSPARSE_INDEX_BASE_ZERO));
  checkCudaErrors(cudaDeviceSynchronize());


  checkCudaErrors(cudaMemcpy(cooVal, dcooValSorted, nnz * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cooRowIdx, dcooRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cooColIdx, dcooColIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(dcooVal));
  checkCudaErrors(cudaFree(dcooRowIdx));
  checkCudaErrors(cudaFree(dcooColIdx));
  checkCudaErrors(cudaFree(dcooValSorted));
  checkCudaErrors(cudaFree(buffer));
  checkCudaErrors(cudaFree(dP));

}

void sortCOOCol(int m, int n, int nnz,
                double * cooVal, int * cooRowIdx, int * cooColIdx) {
  cudaStream_t stream;
  cusparseStatus_t status;
  cusparseHandle_t handle;

  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cusparseCreate(&handle)); 
  checkCudaErrors(cusparseSetStream(handle, stream));

  double * dcooVal;
  int * dcooRowIdx;
  int * dcooColIdx;
  double * dcooValSorted;
  void * buffer;
  size_t bufferSize = 0;
  int * dP;

  checkCudaErrors(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, 
                                                  dcooRowIdx, dcooColIdx,
                                                  &bufferSize));

  checkCudaErrors(cudaMalloc((void**)&dcooVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcooRowIdx, nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcooColIdx, nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcooValSorted, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&buffer, bufferSize ));
  checkCudaErrors(cudaMalloc((void**)&dP, nnz * sizeof(int)));

  checkCudaErrors(cudaMemcpy(dcooVal, cooVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooRowIdx, cooRowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooColIdx, cooColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));

  checkCudaErrors(cusparseCreateIdentityPermutation(handle, nnz, dP));
  
  checkCudaErrors(cusparseXcoosortByColumn(handle, m, n, nnz, 
                                        dcooRowIdx, dcooColIdx,
                                        dP, buffer));
  checkCudaErrors(cusparseDgthr(handle, nnz, dcooVal, dcooValSorted, dP,
                                CUSPARSE_INDEX_BASE_ZERO));
  checkCudaErrors(cudaDeviceSynchronize());


  checkCudaErrors(cudaMemcpy(cooVal, dcooValSorted, nnz * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cooRowIdx, dcooRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cooColIdx, dcooColIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(dcooVal));
  checkCudaErrors(cudaFree(dcooRowIdx));
  checkCudaErrors(cudaFree(dcooColIdx));
  checkCudaErrors(cudaFree(dcooValSorted));
  checkCudaErrors(cudaFree(buffer));
  checkCudaErrors(cudaFree(dP));

}

void sortCOOGPUEx (cusparseHandle_t handle, int m, int n, int nnz,
                      int * cooRowIdx, int * cooColIdx,
                      double ** cooValSorted,  void ** buffer, int ** P) {
  size_t bufferSize = 0;
  checkCudaErrors(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, 
                                                 cooRowIdx, cooColIdx,
                                                 &bufferSize));


  checkCudaErrors(cudaMalloc((void**)cooValSorted, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)buffer, bufferSize ));
  checkCudaErrors(cudaMalloc((void**)P, nnz * sizeof(int)));
}




void sortCOORowGPU(cusparseHandle_t handle, cudaStream_t stream,
                    int m, int n, int nnz,
                    double * cooVal, int * cooRowIdx, int * cooColIdx,
                    double * cooValSorted,  void * buffer, int * P) {
  

  // double * cooValSorted;
  // void * buffer;
  // size_t bufferSize = 0;
  // int * P;

  // checkCudaErrors(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, 
  //                                                 dcooRowIdx, dcooColIdx,
  //                                                 &bufferSize));

  // checkCudaErrors(cudaMalloc((void**)&cooValSorted, nnz * sizeof(double)));
  // checkCudaErrors(cudaMalloc((void**)&buffer, bufferSize ));
  // checkCudaErrors(cudaMalloc((void**)&P, nnz * sizeof(int)));

  checkCudaErrors(cusparseCreateIdentityPermutation(handle, nnz, P));
  
  checkCudaErrors(cusparseXcoosortByRow(handle, m, n, nnz, 
                                        cooRowIdx, cooColIdx,
                                        P, buffer));
  checkCudaErrors(cusparseDgthr(handle, nnz, cooVal, cooValSorted, P,
                                CUSPARSE_INDEX_BASE_ZERO));

  checkCudaErrors(cudaMemcpyAsync(cooVal, cooValSorted, nnz * sizeof(double), 
                                  cudaMemcpyDeviceToDevice, stream));

  // checkCudaErrors(cudaFree(dcooValSorted));
  // checkCudaErrors(cudaFree(buffer));
  // checkCudaErrors(cudaFree(dP));

}

void sortCOOColGPU(cusparseHandle_t handle, cudaStream_t stream,
                int m, int n, int nnz,
                double * cooVal, int * cooRowIdx, int * cooColIdx,
                double * cooValSorted, void * buffer, int * P) {
 
  // double * cooValSorted;
  // void * buffer;
  // size_t bufferSize = 0;
  // int * P;

  // checkCudaErrors(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, 
  //                                                 cooRowIdx, cooColIdx,
  //                                                 &bufferSize));

  // checkCudaErrors(cudaMalloc((void**)&dcooValSorted, nnz * sizeof(double)));
  // checkCudaErrors(cudaMalloc((void**)&buffer, bufferSize ));
  // checkCudaErrors(cudaMalloc((void**)&P, nnz * sizeof(int)));

  checkCudaErrors(cusparseCreateIdentityPermutation(handle, nnz, P));
  
  checkCudaErrors(cusparseXcoosortByColumn(handle, m, n, nnz, 
                                        cooRowIdx, cooColIdx,
                                        P, buffer));
  checkCudaErrors(cusparseDgthr(handle, nnz, cooVal, cooValSorted, P,
                                CUSPARSE_INDEX_BASE_ZERO));

  checkCudaErrors(cudaMemcpyAsync(cooVal, cooValSorted, nnz * sizeof(double), 
                                  cudaMemcpyDeviceToDevice, stream));

  // checkCudaErrors(cudaFree(dcooValSorted));
  // checkCudaErrors(cudaFree(buffer));
  // checkCudaErrors(cudaFree(dP));

}

//coo sorted by row
void coo2csrGPU(cusparseHandle_t handle, cudaStream_t stream, int m, int n, int nnz,
                double * cooVal, int * cooRowIdx, int * cooColIdx,
                double * csrVal, int * csrRowPtr, int * csrColIdx) {

  // checkCudaErrors(cudaDeviceSynchronize());
  // print_vec_gpu(cooRowIdx, nnz, "cooRowIdx");
  checkCudaErrors(cusparseXcoo2csr(handle, cooRowIdx, nnz, m, csrRowPtr,
                  CUSPARSE_INDEX_BASE_ZERO));
  // checkCudaErrors(cudaDeviceSynchronize());
  // print_vec_gpu(csrRowPtr, m+1, "csrRowPtr");
  checkCudaErrors(cudaMemcpyAsync(csrVal, cooVal, nnz * sizeof(double), cudaMemcpyDeviceToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(csrColIdx, cooColIdx, nnz * sizeof(int), cudaMemcpyDeviceToDevice, stream));
}


void coo2csr(int m, int n, int nnz,
             double * cooVal, int * cooRowIdx, int * cooColIdx,
             double * csrVal, int * csrRowPtr, int * csrColIdx) {
  cudaStream_t stream;
  cusparseStatus_t status;
  cusparseHandle_t handle;

  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cusparseCreate(&handle)); 
  checkCudaErrors(cusparseSetStream(handle, stream));

  double * dcooVal;
  int * dcooRowIdx;
  int * dcooColIdx;
  checkCudaErrors(cudaMalloc((void**)&dcooVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcooRowIdx, nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcooColIdx, nnz * sizeof(int)));

  double * dcsrVal;
  int * dcsrRowPtr;
  int * dcsrColIdx;
  checkCudaErrors(cudaMalloc((void**)&dcsrVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcsrRowPtr, (m+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcsrColIdx, nnz * sizeof(int)));


  checkCudaErrors(cudaMemcpy(dcooVal, cooVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooRowIdx, cooRowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooColIdx, cooColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));

  double * cooValSorted;
  void * buffer;
  int * P;

  sortCOOGPUEx(handle, m, n, nnz,
               dcooRowIdx, dcooColIdx,
               &cooValSorted, &buffer, &P);
  checkCudaErrors(cudaDeviceSynchronize());

  sortCOORowGPU(handle, stream,
                 m, n, nnz,
                 dcooVal, dcooRowIdx, dcooColIdx,
                 cooValSorted, buffer, P);
  checkCudaErrors(cudaDeviceSynchronize());

  // print_vec_gpu(dcooVal, 5, "dcooVal");
  // print_vec_gpu(dcooRowIdx, 5, "dcooRowIdx");
  // print_vec_gpu(dcooColIdx, 5, "dcooColIdx");

  coo2csrGPU(handle, stream, m, n, nnz,
             dcooVal, dcooRowIdx, dcooColIdx,
             dcsrVal, dcsrRowPtr, dcsrColIdx);

  checkCudaErrors(cudaDeviceSynchronize());

  // print_vec_gpu(dcsrVal, 5, "dcsrVal");
  // print_vec_gpu(dcsrRowPtr+m, 1, "dcsrRowPtr");
  // print_vec_gpu(dcsrColIdx, 5, "dcsrColIdx");

  checkCudaErrors(cudaMemcpy(csrVal, dcsrVal,       nnz * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csrRowPtr, dcsrRowPtr, (m+1) * sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csrColIdx, dcsrColIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaDeviceSynchronize());

  // print_vec(csrRowPtr+m, 1, "csrRowPtr");

  checkCudaErrors(cudaFree(dcooVal));
  checkCudaErrors(cudaFree(dcooRowIdx));
  checkCudaErrors(cudaFree(dcooColIdx));

  checkCudaErrors(cudaFree(dcsrVal));
  checkCudaErrors(cudaFree(dcsrRowPtr));
  checkCudaErrors(cudaFree(dcsrColIdx));

  checkCudaErrors(cudaFree(cooValSorted));
  checkCudaErrors(cudaFree(buffer));
  checkCudaErrors(cudaFree(P));

}


void coo2csc(int m, int n, int nnz,
             double * cooVal, int * cooRowIdx, int * cooColIdx,
             double * cscVal, int * cscColPtr, int * cscRowIdx) {
  cudaStream_t stream;
  cusparseStatus_t status;
  cusparseHandle_t handle;

  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cusparseCreate(&handle)); 
  checkCudaErrors(cusparseSetStream(handle, stream));

  double * dcooVal;
  int * dcooRowIdx;
  int * dcooColIdx;
  checkCudaErrors(cudaMalloc((void**)&dcooVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcooRowIdx, nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcooColIdx, nnz * sizeof(int)));

  double * dcscVal;
  int * dcscColPtr;
  int * dcscRowIdx;
  checkCudaErrors(cudaMalloc((void**)&dcscVal, nnz * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&dcscColPtr, (n+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dcscRowIdx, nnz * sizeof(int)));

  checkCudaErrors(cudaMemcpy(dcooVal, cooVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooRowIdx, cooRowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dcooColIdx, cooColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));

  double * cooValSorted;
  void * buffer;
  int * P;

  sortCOOGPUEx(handle, m, n, nnz,
               dcooRowIdx, dcooColIdx,
               &cooValSorted, &buffer, &P);
  checkCudaErrors(cudaDeviceSynchronize());

  sortCOOColGPU(handle, stream,
                 m, n, nnz,
                 dcooVal, dcooRowIdx, dcooColIdx,
                 cooValSorted, buffer, P);
  checkCudaErrors(cudaDeviceSynchronize());

  coo2csrGPU(handle, stream, n, m, nnz,
             dcooVal, dcooColIdx, dcooRowIdx,
             dcscVal, dcscColPtr, dcscRowIdx);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(cscVal, dcscVal,       nnz * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cscColPtr, dcscColPtr, (n+1) * sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cscRowIdx, dcscRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(dcooVal));
  checkCudaErrors(cudaFree(dcooRowIdx));
  checkCudaErrors(cudaFree(dcooColIdx));

  checkCudaErrors(cudaFree(dcscVal));
  checkCudaErrors(cudaFree(dcscColPtr));
  checkCudaErrors(cudaFree(dcscRowIdx));

  checkCudaErrors(cudaFree(cooValSorted));
  checkCudaErrors(cudaFree(buffer));
  checkCudaErrors(cudaFree(P));

}



int findFirstInSorted(int * a, int n, int key) {
  int l = 0;
  int r = n;
  int m = l + (r - l) / 2;
  while (l < r - 1) {
    int m = l + (r - l) / 2;
    if (key <= a[m]) {
      r = m;
    } else if (key > a[m]) {
      l = m;
    } 
  }
  //cout << "a[" << l << "] = " <<  a[l] << endl;
  //cout << " a[" << r << "] = " << a[r] << endl;
  //cout << " idx = " << idx << endl;
  if (key == a[l]) return l;
  if (key == a[r]) return r;
  return l;
}


int findLastInSorted(int * a, int n, int key) {
  int l = 0;
  int r = n;
  int m = l + (r - l) / 2;
  while (l < r - 1) {
    int m = l + (r - l) / 2;
    if (key < a[m]) {
      r = m;
    } else if (key >= a[m]) {
      l = m;
    } 
  }
  //cout << "a[" << l << "] = " <<  a[l] << endl;
  //cout << " a[" << r << "] = " << a[r] << endl;
  //cout << " idx = " << idx << endl;
  if (key == a[l]) return l;
  if (key == a[r]) return r;
  return l;
}

void report_mem_usage(int d) {
  size_t free_byte ;
  size_t total_byte ;
  checkCudaErrors(cudaSetDevice(d));
  checkCudaErrors(cudaMemGetInfo( &free_byte, &total_byte ));
  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  printf("GPU %d memory usage: used = %f, free = %f MB, total = %f MB\n", d,
  used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
void report_all_mem_usage() {
  int deviceCount;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  for (int d = 0; d < deviceCount; d++) {
    report_mem_usage(d);
  }

