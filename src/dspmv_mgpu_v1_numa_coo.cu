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
#include <vector>
#include <cuda_profiler_api.h>
#include <omp.h>
#include <sched.h>
#include <library_types.h>
#include <string>
using namespace std;

__global__ void
_calcCooRowIdx(int * cooRowIdx, int nnz, int offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < nnz; i += blockDim.x * gridDim.x) {
    cooRowIdx[i] -= offset; 
  }
}

void calcCooRowIdx(int * cooRowIdx, int nnz, int offset, cudaStream_t stream) {
  int thread_per_block = 256;
  int block_per_grid = ceil((float)nnz / thread_per_block); 
  _calcCooRowIdx<<<block_per_grid, thread_per_block, 0, stream>>>(cooRowIdx, nnz, offset);
}


spmv_ret spMV_mgpu_v1_numa_coo(int m, int n, int nnz, double * alpha,
          double * cooVal, int * cooRowIdx, int * cooColIdx, 
          double * x, double * beta,
          double * y,
          int ngpu, 
          int kernel,
          int * numa_mapping,
          int part_opt,
          int merg_opt){

  double numa_part_time = 0.0;
  double part_time = 0.0;
  double comp_time = 0.0;
  double comm_time = 0.0;
  double merg_time = 0.0;
  
  struct NumaContext numaContext(numa_mapping, ngpu);
  struct pCOO * pcooNuma = new struct pCOO[numaContext.numNumaNodes];

  omp_set_num_threads(ngpu);
  #pragma omp parallel default (shared) reduction(max:numa_part_time)
  {
    string s;
    unsigned int dev_id = omp_get_thread_num();
    unsigned int hwthread = sched_getcpu();

    int numa_id = numaContext.numaMapping[dev_id];
  
    numa_part_time = 0;
  
    if (numaContext.representiveThreads[dev_id]) {
      printf("represent thread %d hw thread %d\n", dev_id, hwthread);

      double tmp_time = get_time();

      int tmp1 = numaContext.workload[numa_id] * nnz;
      int tmp2 = numaContext.workload[numa_id + 1] * nnz;

      pcooNuma[numa_id].startIdx = floor((double)tmp1 / ngpu);
      pcooNuma[numa_id].endIdx = floor((double)tmp2 / ngpu) - 1;
      pcooNuma[numa_id].startRow = cooRowIdx[pcooNuma[numa_id].startIdx];
      pcooNuma[numa_id].endRow = cooRowIdx[pcooNuma[numa_id].endIdx];

      printf("thread %d hw done1 %d\n", dev_id);

      pcooNuma[numa_id].m = pcooNuma[numa_id].endRow - pcooNuma[numa_id].startRow + 1;
      pcooNuma[numa_id].n = n;
      pcooNuma[numa_id].nnz  = pcooNuma[numa_id].endIdx - pcooNuma[numa_id].startIdx + 1;

      printf("thread %d hw done2 %d\n", dev_id);
      // Mark imcomplete rows
      // True: imcomplete
      if (pcooNuma[numa_id].startIdx > findFirstInSorted(cooRowIdx, nnz, pcooNuma[numa_id].startRow)) {
        pcooNuma[numa_id].startFlag = true;
        pcooNuma[numa_id].org_y = y[pcooNuma[numa_id].startRow];
      } else {
        pcooNuma[numa_id].startFlag = false;
      }
      printf("thread %d hw done3 %d\n", dev_id);

      // Mark imcomplete rows
      // True: imcomplete
      if (pcooNuma[numa_id].endIdx < findLastInSorted(cooRowIdx, nnz, pcooNuma[numa_id].endRow))  {
        pcooNuma[numa_id].endFlag = true;
      } else {
        pcooNuma[numa_id].endFlag = false;
      }
      printf("thread %d hw done4 %d\n", dev_id);
    
      printf("numa_id %d, numa_start_idx %d, numa_end_idx %d\n",numa_id, pcooNuma[numa_id].startIdx, pcooNuma[numa_id].endIdx);
      printf("numa_id %d, numa_start_row %d, numa_end_row %d\n",numa_id, pcooNuma[numa_id].startRow, pcooNuma[numa_id].endRow);
    

      numa_part_time += get_time() - tmp_time;

      // preparing data on host 
      // cudaMallocHost((void**)&(pcooNuma[numa_id].val),    pcooNuma[numa_id].nnz * sizeof(double));
      // cudaMallocHost((void**)&(pcooNuma[numa_id].rowIdx), pcooNuma[numa_id].nnz * sizeof(int));
      // cudaMallocHost((void**)&(pcooNuma[numa_id].colIdx), pcooNuma[numa_id].nnz * sizeof(int));
      // cudaMallocHost((void**)&(pcooNuma[numa_id].x), pcooNuma[numa_id].n * sizeof(double));
      // cudaMallocHost((void**)&(pcooNuma[numa_id].y), pcooNuma[numa_id].m * sizeof(double));

      printf("done pin allocation\n"); 
      tmp_time = get_time();

      for (int i = pcooNuma[numa_id].startIdx; i <= pcooNuma[numa_id].endIdx; i++) {
        pcooNuma[numa_id].val[i - pcooNuma[numa_id].startIdx] = cooVal[i];
        pcooNuma[numa_id].rowIdx[i - pcooNuma[numa_id].startIdx] = cooRowIdx[i] - pcooNuma[numa_id].startRow;
        pcooNuma[numa_id].colIdx[i - pcooNuma[numa_id].startIdx] = cooColIdx[i];
      }

      for (int i = 0; i < pcooNuma[numa_id].n; i++) {
        pcooNuma[numa_id].x[i] = x[i];
      }
    
      for (int i = 0; i < pcooNuma[numa_id].m; i++) {
        pcooNuma[numa_id].y[i] = y[pcooNuma[numa_id].startRow + i];
      }

      numa_part_time += get_time() - tmp_time;

    }
  }
  





  
  // struct pCOO * pcooGPU = new struct pCOO[ngpu];

  // double * start_element = new double[ngpu];
  // double * end_element = new double[ngpu];
  // bool * start_flags = new bool[ngpu];
  // bool * end_flags = new bool[ngpu];
  // double * org_y = new double[ngpu];
  // int * start_rows = new int[ngpu];




  // omp_set_num_threads(ngpu);
  // #pragma omp parallel default (shared) reduction(max:comp_time) reduction(max:comm_time) reduction(max:part_time) reduction(max:merg_time)
  // {
  //   unsigned int dev_id = omp_get_thread_num();
  //   checkCudaErrors(cudaSetDevice(dev_id));
  //   unsigned int hwthread = sched_getcpu();
  //   float elapsedTime;

  //   comm_time = 0.0;
  //   comp_time = 0.0;
  //   merg_time = 0.0;

  //   cudaStream_t stream;
  //   cusparseStatus_t status;
  //   cusparseHandle_t handle;
  //   cusparseMatDescr_t descr;

  //   checkCudaErrors(cudaStreamCreate(&stream));
  //   checkCudaErrors(cusparseCreate(&handle)); 
  //   checkCudaErrors(cusparseSetStream(handle, stream));
  //   checkCudaErrors(cusparseCreateMatDescr(&descr));
  //   checkCudaErrors(cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL)); 
  //   checkCudaErrors(cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO));

  //   cudaEvent_t comp_start, comp_stop;
  //   cudaEvent_t comm_start, comm_stop;

  //   checkCudaErrors(cudaEventCreate(&comp_start));
  //   checkCudaErrors(cudaEventCreate(&comp_stop));
  //   checkCudaErrors(cudaEventCreate(&comm_start));
  //   checkCudaErrors(cudaEventCreate(&comm_stop));



  //   int numa_id = numaContext.numaMapping[dev_id];
  //   int local_dev_id = 0;
  //   for (int i = 0; i < ngpu; i++) {
  //     if (i == dev_id) break;
  //     if (numa_id == numaContext.numaMapping[i]) local_dev_id++;
  //   }

  //   printf("omp thread %d, hw thread %d, numa_id %d, local_id %d\n", dev_id, hwthread, numa_id, local_dev_id);  
  //   double tmp_time = get_time();

  //   // Calculate the start and end index
  //   int tmp1 = local_dev_id * pcooNuma[numa_id].nnz;
  //   int tmp2 = (local_dev_id + 1) * pcooNuma[numa_id].nnz;

  //   pcooGPU[dev_id].startIdx = floor((double)tmp1 / numaContext.numGPUs[numa_id]);
  //   pcooGPU[dev_id].endIdx   = floor((double)tmp2 / numaContext.numGPUs[numa_id]) - 1;
  //   pcooGPU[dev_id].startRow = pcooNuma[numa_id].rowIdx[pcooGPU[dev_id].startIdx];
  //   pcooGPU[dev_id].endRow = pcooNuma[numa_id].rowIdx[pcooGPU[dev_id].endIdx];

  //   pcooGPU[dev_id].m = pcooGPU[dev_id].endRow - pcooGPU[dev_id].startRow + 1;
  //   pcooGPU[dev_id].n = n;
  //   pcooGPU[dev_id].nnz  = pcooGPU[dev_id].endIdx - pcooGPU[dev_id].startIdx + 1;
  
  //   // Mark imcomplete rows
  //   // True: imcomplete
  //   if (pcooGPU[dev_id].startIdx > findFirstInSorted(pcooNuma[numa_id].rowIdx, 
  //                                                    pcooGPU[dev_id].nnz, 
  //                                                    pcooGPU[dev_id].startRow)) {
  //     pcooGPU[dev_id].startFlag = true;
  //     pcooGPU[dev_id].org_y = pcooNuma[numa_id].y[pcooGPU[dev_id].startRow];
  //   } else {
  //     pcooGPU[dev_id].startFlag = false;
  //   }

  //   if (local_dev_id == 0) {
  //     pcooGPU[dev_id].startFlag = pcooNuma[numa_id].startFlag; // see if numa block is complete
  //   }

  //   // Mark imcomplete rows
  //   // True: imcomplete
  //   if (pcooGPU[dev_id].endIdx < findLastInSorted(pcooNuma[numa_id].rowIdx, 
  //                                                 pcooGPU[dev_id].nnz, 
  //                                                 pcooGPU[dev_id].endRow))  {
  //     pcooGPU[dev_id].endFlag = true;
  //   } else {
  //     pcooGPU[dev_id].endFlag = false;
  //   }

  //   if (local_dev_id + 1 == numaContext.numGPUs[numa_id]) {
  //     pcooGPU[dev_id].endFlag = pcooNuma[numa_id].endFlag;
  //   }
    


  //   printf("omp thread %d, dev_m %d, dev_n %d, dev_nnz %d, start_idx %d, end_idx %d, start_row %d, end_row %d\n", 
  //          dev_id, pcooGPU[dev_id].m, pcooGPU[dev_id].n, pcooGPU[dev_id].nnz, pcooGPU[dev_id].startIdx, pcooGPU[dev_id].endIdx, 
  //          pcooGPU[dev_id].startRow, pcooGPU[dev_id].endRow);


  //   // preparing data on host 

  //   //tmp_time = get_time();
  //   pcooGPU[dev_id].val    = &(pcooNuma[numa_id].val[pcooGPU[dev_id].startIdx]);
  //   pcooGPU[dev_id].rowIdx = &(pcooNuma[numa_id].rowIdx[pcooGPU[dev_id].startIdx]);
  //   pcooGPU[dev_id].colIdx = &(pcooNuma[numa_id].colIdx[pcooGPU[dev_id].startIdx]);
  //   pcooGPU[dev_id].x      = pcooNuma[numa_id].x;
  //   pcooGPU[dev_id].y      = &(pcooNuma[numa_id].y[pcooGPU[dev_id].startRow]);


  //   // host_csrVal = &numa_csrVal[numa_id][start_idx];
  //   // host_csrRowPtr = &numa_csrRowPtr[numa_id][start_row];
  //   // host_csrColIndex = &numa_csrColIndex[numa_id][start_idx];
  //   // host_x = numa_x[numa_id];
  //   // host_y = &numa_y[numa_id][start_row];
  //   part_time = get_time() - tmp_time;  


  //   checkCudaErrors(cudaMalloc((void**)&(pcooGPU[dev_id].dval),    pcooGPU[dev_id].nnz * sizeof(double)));
  //   checkCudaErrors(cudaMalloc((void**)&(pcooGPU[dev_id].drowIdx), pcooGPU[dev_id].nnz * sizeof(int)   ));
  //   checkCudaErrors(cudaMalloc((void**)&(pcooGPU[dev_id].dcolIdx), pcooGPU[dev_id].nnz * sizeof(int)   ));
  //   checkCudaErrors(cudaMalloc((void**)&(pcooGPU[dev_id].dx),      pcooGPU[dev_id].n   * sizeof(double))); 
  //   checkCudaErrors(cudaMalloc((void**)&(pcooGPU[dev_id].dy),      pcooGPU[dev_id].m   * sizeof(double))); 
  //   checkCudaErrors(cudaMallocHost((void**)&(pcooGPU[dev_id].py),  pcooGPU[dev_id].m   * sizeof(double)));

  //   double * dev_csrVal;
  //   int * dev_csrRowPtr;
  //   int * dev_csrColIdx;
  //   checkCudaErrors(cudaMalloc((void**)&dev_csrVal,    pcooGPU[dev_id].nnz     * sizeof(double)));
  //   checkCudaErrors(cudaMalloc((void**)&dev_csrRowPtr, (pcooGPU[dev_id].m + 1) * sizeof(int)   ));
  //   checkCudaErrors(cudaMalloc((void**)&dev_csrColIdx, pcooGPU[dev_id].nnz     * sizeof(int) ));


  //   if (part_opt == 0) {
      
  //     checkCudaErrors(cudaMallocHost((void**)&pcooGPU[dev_id].host_rowIdx, pcooGPU[dev_id].nnz * sizeof(int)));
      

  //     tmp_time = get_time();
  //     for (int i = 0; i < pcooGPU[dev_id].nnz; i ++) {
  //       pcooGPU[dev_id].host_rowIdx[i] = pcooGPU[dev_id].rowIdx[i] - pcooGPU[dev_id].startRow;
  //     }
  //     part_time += get_time() - tmp_time; 
  //     checkCudaErrors(cudaEventRecord(comm_start, stream));
  //     checkCudaErrors(cudaMemcpyAsync(pcooGPU[dev_id].drowIdx, pcooGPU[dev_id].host_rowIdx, pcooGPU[dev_id].nnz * sizeof(int), cudaMemcpyHostToDevice, stream)); 
  //     checkCudaErrors(cudaEventRecord(comm_stop, stream));
  //     checkCudaErrors(cudaDeviceSynchronize());
  //     checkCudaErrors(cudaFreeHost(pcooGPU[dev_id].host_rowIdx));
  //   }
 
  //   if (part_opt == 1) {
  //     checkCudaErrors(cudaEventRecord(comm_start, stream));
  //     checkCudaErrors(cudaMemcpyAsync(pcooGPU[dev_id].drowIdx, pcooGPU[dev_id].rowIdx, pcooGPU[dev_id].nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
  //     checkCudaErrors(cudaEventRecord(comm_stop, stream));
  //     checkCudaErrors(cudaDeviceSynchronize());
      
  //     tmp_time = get_time();
  //     calcCooRowIdx(pcooGPU[dev_id].drowIdx, pcooGPU[dev_id].nnz, pcooGPU[dev_id].startRow, stream);
  //     checkCudaErrors(cudaDeviceSynchronize());
  //     part_time += get_time() - tmp_time;  
  //   }

  //   checkCudaErrors(cudaEventSynchronize(comm_stop));
  //   elapsedTime = 0.0;
  //   checkCudaErrors(cudaEventElapsedTime(&elapsedTime, comm_start, comm_stop));
  //   elapsedTime /= 1000.0;
  //   comm_time += elapsedTime;


    

  //   // original partition*******************************************
  //   int tmp1 = dev_id * nnz;
  //   int tmp2 = (dev_id + 1) * nnz;

  //   start_idx = floor((double)tmp1 / ngpu);
  //   end_idx   = floor((double)tmp2 / ngpu) - 1;

  //   // Calculate the start and end row
  //   start_row = get_row_from_index(m, csrRowPtr, start_idx);
  //   // Mark imcomplete rows
  //   // True: imcomplete
  //   if (start_idx > csrRowPtr[start_row]) {
  //     start_flag = true;
  //     y2 = y[start_row];
  //     org_y[dev_id] = y[start_row]; //use dev_id for global merge
  //     start_rows[dev_id] = start_row;
  //   } else {
  //     start_flag = false;
  //   }
  //   start_flags[dev_id] = start_flag;

  //   end_row = get_row_from_index(m, csrRowPtr, end_idx);
  //   // Mark imcomplete rows
  //   // True: imcomplete
  //   if (end_idx < csrRowPtr[end_row + 1] - 1)  {
  //     end_flag = true;
  //   } else {
  //     end_flag = false;
  //   }
    
  //   // Cacluclate dimensions
  //   dev_m = end_row - start_row + 1;
  //   dev_n = n;
  //   dev_nnz   = (int)(end_idx - start_idx + 1);


    

  //   part_time = get_time() - tmp_time;  

  //   printf("omp thread %d, dev_m %d, dev_n %d, dev_nnz %d, start_idx %d, end_idx %d, start_row %d, end_row %d\n", dev_id, dev_m, dev_n, dev_nnz, start_idx, end_idx, start_row, end_row);

  //   // preparing data on host 
  //   cudaMallocHost((void**)&host_csrVal, dev_nnz * sizeof(double));
  //   cudaMallocHost((void**)&host_csrRowPtr, (dev_m + 1)*sizeof(int));
  //   cudaMallocHost((void**)&host_csrColIndex, dev_nnz * sizeof(int));
  //   cudaMallocHost((void**)&host_x, dev_n * sizeof(double));
  //   cudaMallocHost((void**)&host_y, dev_m * sizeof(double));

  //   tmp_time = get_time();
  //   for (int i = start_idx; i <= end_idx; i++) {
  //     host_csrVal[i - start_idx] = csrVal[i];
  //   }
  //   //host_csrVal = &numa_csrVal[numa_id][start_idx];

  //   host_csrRowPtr[0] = 0;
  //   host_csrRowPtr[dev_m] = dev_nnz;
  //   for (int j = 1; j < dev_m; j++) {
  //     host_csrRowPtr[j] = (int)(csrRowPtr[start_row + j] - start_idx);
  //   }
  //   //host_csrRowPtr = &numa_csrRowPtr[numa_id][start_row];

  //   printf("dev %d: %d %d %d %d %d\n", host_csrRowPtr[0],host_csrRowPtr[1],host_csrRowPtr[2],host_csrRowPtr[3],host_csrRowPtr[4]);

  //   for (int i = start_idx; i <= end_idx; i++) {
  //     host_csrColIndex[i - start_idx] = csrColIndex[i];
  //   }
  //   //host_csrColIndex = &numa_csrColIndex[numa_id][start_idx];

  //   for (int i = 0; i < dev_n; i++) {
  //     host_x[i] = x[i];
  //   }
  //   //host_x = numa_x[numa_id];

  //   for (int i = 0; i < dev_m; i++) {
  //     host_y[i] = y[start_row + i];
  //   }
  //   //host_y = &numa_y[numa_id][start_row];

  //   part_time += get_time() - tmp_time;

  //   // end of original partition*********************************

    

    



  //   checkCudaErrors(cudaEventRecord(comm_start, stream));
  //   checkCudaErrors(cudaMemcpyAsync(pcooGPU[dev_id].dcolIdx, pcooGPU[dev_id].colIdx, pcooGPU[dev_id].nnz * sizeof(int), cudaMemcpyHostToDevice, stream)); 
  //   checkCudaErrors(cudaMemcpyAsync(pcooGPU[dev_id].dval,    pcooGPU[dev_id].val,    pcooGPU[dev_id].nnz * sizeof(double), cudaMemcpyHostToDevice, stream));
  //   checkCudaErrors(cudaMemcpyAsync(pcooGPU[dev_id].dy,      pcooGPU[dev_id].y,      pcooGPU[dev_id].m*sizeof(double),  cudaMemcpyHostToDevice, stream)); 
  //   checkCudaErrors(cudaMemcpyAsync(pcooGPU[dev_id].dx,      pcooGPU[dev_id].x,      pcooGPU[dev_id].n*sizeof(double), cudaMemcpyHostToDevice, stream)); 
  //   checkCudaErrors(cudaEventRecord(comm_stop, stream));

  //   // #pragma omp barrier
  //   // #pragma omp critical 
  //   // {
  //   // cudaDeviceSynchronize();
  //   // print_vec_gpu(pcooGPU[dev_id].dval, pcooGPU[dev_id].nnz, "dval"+to_string(dev_id));
  //   // print_vec_gpu(pcooGPU[dev_id].drowIdx, pcooGPU[dev_id].nnz, "drowIdx"+to_string(dev_id));
  //   // print_vec_gpu(pcooGPU[dev_id].dcolIdx, pcooGPU[dev_id].nnz, "dcolIdx"+to_string(dev_id));
  //   // print_vec_gpu(pcooGPU[dev_id].dy, pcooGPU[dev_id].m, "y"+to_string(dev_id));
  //   // print_vec_gpu(pcooGPU[dev_id].dx, pcooGPU[dev_id].n, "y_before"+to_string(dev_id));
  //   // printf("dev_id %d, alpha %f, beta %f\n", dev_id, *alpha, *beta);
  //   // }


  //   //calcCsrRowPtr(dev_csrRowPtr, dev_m, start_idx, dev_nnz, stream);
  //   //cudaDeviceSynchronize();
  
  //   //print_vec_gpu(dev_x, dev_n, "x"+to_string(dev_id));
  //   checkCudaErrors(cudaEventRecord(comp_start, stream));
  //   coo2csrGPU(handle, stream, pcooGPU[dev_id].m, pcooGPU[dev_id].n, pcooGPU[dev_id].nnz,
  //               pcooGPU[dev_id].dval, pcooGPU[dev_id].drowIdx, pcooGPU[dev_id].dcolIdx,
  //               dev_csrVal, dev_csrRowPtr, dev_csrColIdx);
  //   // #pragma omp barrier
  //   // #pragma omp critical 
  //   // {
  //   // checkCudaErrors(cudaDeviceSynchronize());
  //   // print_vec_gpu(dev_csrVal, pcooGPU[dev_id].nnz, "dev_csrVal"+to_string(dev_id));
  //   // print_vec_gpu(dev_csrRowPtr, pcooGPU[dev_id].m+1, "dev_csrRowPtr"+to_string(dev_id));
  //   // print_vec_gpu(dev_csrColIdx, pcooGPU[dev_id].nnz, "dev_csrColIdx"+to_string(dev_id));
  //   // }
  
  //   checkCudaErrors(cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
  //                             pcooGPU[dev_id].m, pcooGPU[dev_id].n, pcooGPU[dev_id].nnz, 
  //                             alpha, descr, dev_csrVal, dev_csrRowPtr, dev_csrColIdx, 
  //                             pcooGPU[dev_id].dx, beta, pcooGPU[dev_id].dy));
  //   checkCudaErrors(cudaEventRecord(comp_stop, stream));

  //   checkCudaErrors(cudaEventSynchronize(comm_stop));
  //   elapsedTime = 0.0;
  //   checkCudaErrors(cudaEventElapsedTime(&elapsedTime, comm_start, comm_stop));
  //   elapsedTime /= 1000.0;
  //   comm_time += elapsedTime;

  //   checkCudaErrors(cudaEventSynchronize(comp_stop));
  //   elapsedTime = 0.0;
  //   checkCudaErrors(cudaEventElapsedTime(&elapsedTime, comp_start, comp_stop));
  //   elapsedTime /= 1000.0;
  //   comp_time += elapsedTime;


    
  //   checkCudaErrors(cudaDeviceSynchronize());
  //   // if (status != CUSPARSE_STATUS_SUCCESS) printf("dev_id %d: exec error\n", dev_id);
  //   //print_vec_gpu(pcooGPU[dev_id].dy, pcooGPU[dev_id].m, "y_after"+to_string(dev_id));
  //   // printf("omp thread %d, time %f\n", dev_id, get_time() - tmp_time);
  //   //comp_time = get_time() - tmp_time;
  //   // GPU based merge
  //   #pragma omp barrier
  //   tmp_time = get_time();

  //   if (merg_opt == 1) {

  //     double * dev_y_no_overlap = pcooGPU[dev_id].dy;
  //     int dev_m_no_overlap = pcooGPU[dev_id].m;
  //     int start_row_no_overlap = pcooNuma[numa_id].startRow + pcooGPU[dev_id].startRow;

  //     // double * dev_y_no_overlap = dev_y;
  //     // int dev_m_no_overlap = dev_m;
  //     // int start_row_no_overlap = numa_start_row[numa_id] + start_row;
  //     //int start_row_no_overlap = start_row;
  //     if (pcooGPU[dev_id].startFlag) {
  //       dev_y_no_overlap += 1;
  //       start_row_no_overlap += 1;
  //       dev_m_no_overlap -= 1;
  //       checkCudaErrors(cudaMemcpyAsync(start_element+dev_id, pcooGPU[dev_id].dy, sizeof(double), cudaMemcpyDeviceToHost, stream));
  //       //cudaMemcpyAsync(start_element+dev_id, dev_y, sizeof(double), cudaMemcpyDeviceToHost, stream);
  //     }
  //     checkCudaErrors(cudaMemcpyAsync(y+start_row_no_overlap, dev_y_no_overlap, dev_m_no_overlap*sizeof(double),  cudaMemcpyDeviceToHost, stream));
  //     checkCudaErrors(cudaDeviceSynchronize());
  //     #pragma omp barrier
  //     if (dev_id == 0) {
  //       for (int i = 0; i < ngpu; i++) {
  //         if (pcooGPU[i].startFlag) {
  //           y[pcooNuma[numaContext.numaMapping[i]].startRow + pcooGPU[i].startRow] += (start_element[i] - (*beta) * pcooGPU[i].org_y); 
  //           //y[start_rows[i]] += (start_element[i] - (*beta) * org_y[i]);
  //         } 

  //         // if (start_flags[i]) {
  //         //   y[numa_start_row[numa_mapping[i]] + start_rows[i]] += (start_element[i] - (*beta) * org_y[i]); 
  //         //   //y[start_rows[i]] += (start_element[i] - (*beta) * org_y[i]);
  //         // } 
  //       }
  //     }
  //   }

  //   if (merg_opt == 0) {
  //     //  CPU based merge
  //     checkCudaErrors(cudaMemcpyAsync(pcooGPU[dev_id].py, pcooGPU[dev_id].dy, pcooGPU[dev_id].m * sizeof(double),  cudaMemcpyDeviceToHost, stream));
  //     checkCudaErrors(cudaDeviceSynchronize());
  //     //printf("thread %d time: %f\n", dev_id,  get_time() - tmp_time);
  //     #pragma omp critical
  //     {
  //       double tmp = 0.0;
  //       if (pcooGPU[dev_id].startFlag) {
  //         tmp = y[pcooNuma[numa_id].startRow + pcooGPU[dev_id].startRow];
  //       }
  //       for (int i = 0; i < pcooGPU[dev_id].m; i++) {
  //         y[pcooNuma[numa_id].startRow + pcooGPU[dev_id].startRow + i] += pcooGPU[dev_id].py[i];
  //       }
  //       if (pcooGPU[dev_id].startFlag) {
  //         //y[pcsrGPU[dev_id].startRow] += tmp;
  //         y[pcooNuma[numa_id].startRow + pcooGPU[dev_id].startRow] -= tmp * (*beta);
  //       }
  //     }
  //   }



  //   //  CPU based merge
  //   // cudaMemcpyAsync(host_y, dev_y, dev_m*sizeof(double),  cudaMemcpyDeviceToHost, stream);

  //   // cudaDeviceSynchronize();
    
    
  //   // //printf("thread %d time: %f\n", dev_id,  get_time() - tmp_time);
  //   // #pragma omp critical
  //   // {
  //   //   double tmp = 0.0;
      
  //   //   if (start_flag) {
  //   //     tmp = y[start_row];
  //   //   }

  //   //   for (int i = 0; i < dev_m; i++) y[start_row + i] = host_y[i];

  //   //   if (start_flag) {
  //   //     y[start_row] += tmp;
  //   //     y[start_row] -= y2 * (*beta);
  //   //   }
  //   // }
    
  
  //   merg_time = get_time() - tmp_time;

  //   //cudaProfilerStop();

  //   checkCudaErrors(cudaFree(dev_csrVal));
  //   checkCudaErrors(cudaFree(dev_csrRowPtr));
  //   checkCudaErrors(cudaFree(dev_csrColIdx));

  //   checkCudaErrors(cudaFree(pcooGPU[dev_id].dval));
  //   checkCudaErrors(cudaFree(pcooGPU[dev_id].drowIdx));
  //   checkCudaErrors(cudaFree(pcooGPU[dev_id].dcolIdx));
  //   checkCudaErrors(cudaFree(pcooGPU[dev_id].dx));
  //   checkCudaErrors(cudaFree(pcooGPU[dev_id].dy));
  //   checkCudaErrors(cudaFreeHost(pcooGPU[dev_id].py));
          
  //   //cudaFreeHost(host_csrRowPtr);
  //   //cudaFreeHost(host_csrVal);
  //   //cudaFreeHost(host_csrColIndex);
  //   //cudaFreeHost(host_x);
  //   //cudaFreeHost(host_y);

  //   checkCudaErrors(cudaEventDestroy(comp_start));
  //   checkCudaErrors(cudaEventDestroy(comp_stop));
  //   checkCudaErrors(cudaEventDestroy(comm_start));
  //   checkCudaErrors(cudaEventDestroy(comm_stop));

  //   checkCudaErrors(cusparseDestroyMatDescr(descr));
  //   checkCudaErrors(cusparseDestroy(handle));
  //   checkCudaErrors(cudaStreamDestroy(stream));

  // }

  // for (int numa_id = 0; numa_id < numaContext.numNumaNodes; numa_id++) {
  //   checkCudaErrors(cudaFreeHost(pcooNuma[numa_id].val));
  //   checkCudaErrors(cudaFreeHost(pcooNuma[numa_id].rowIdx));
  //   checkCudaErrors(cudaFreeHost(pcooNuma[numa_id].colIdx));
  //   checkCudaErrors(cudaFreeHost(pcooNuma[numa_id].x));
  //   checkCudaErrors(cudaFreeHost(pcooNuma[numa_id].y));
  // }


  // // printf("end part time: %f\n", part_time);
  // //cout << "time_parse = " << time_parse << ", time_comm = " << time_comm << ", time_comp = "<< time_comp <<", time_post = " << time_post << endl;
  spmv_ret ret;
  // ret.comp_time = comp_time;
  // ret.comm_time = comm_time;
  // ret.part_time = part_time;
  // ret.merg_time = merg_time;
  // ret.numa_part_time = numa_part_time;
  return ret;
}

