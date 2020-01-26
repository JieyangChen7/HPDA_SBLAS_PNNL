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
_calcCscColPtr(int * cscColPrt, int m, int offset, int nnz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < m; i += blockDim.x * gridDim.x) {
    cscColPrt[i] -= offset; 
    //printf("thread %d: %d - %d\n", idx, csrRowPrt[i], offset);
  }
  if (idx == 0) {
    cscColPrt[0] = 0;
    cscColPrt[m] = nnz;
  }
}

void calcCscColPtr(int * cscColPrt, int m, int offset, int nnz, cudaStream_t stream) {
  int thread_per_block = 256;
  int block_per_grid = ceil((float)m / thread_per_block); 
  _calcCscColPtr<<<block_per_grid, thread_per_block, 0, stream>>>(cscColPrt, m, offset, nnz);
}





spmv_ret spMV_mgpu_v1_numa_csc(int m, int n, long long nnz, double * alpha,
          double * cscVal, int * cscColPtr, int * cscRowIndex, 
          double * x, double * beta,
          double * y,
          int ngpu, 
          int kernel,
          int * numa_mapping){

  double curr_time = 0.0;
  double time_comm = 0.0;
  double time_comp = 0.0;


  struct NumaContext numaContext(numa_mapping, ngpu);
  struct pCSC * pcscNuma = new struct pCSC[numaContext.numNumaNodes];

  double numa_part_time;
  
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

      pcscNuma[numa_id].startIdx = floor((double)tmp1 / ngpu);
      pcscNuma[numa_id].endIdx = floor((double)tmp2 / ngpu) - 1;


      // Calculate the start and end row
      pcscNuma[numa_id].startCol = get_row_from_index(n+1, cscColPtr, pcscNuma[numa_id].startIdx);
      // Mark imcomplete rows
      // True: imcomplete
      if (pcscNuma[numa_id].startIdx > cscColPtr[pcscNuma[numa_id].startCol]) {
        pcscNuma[numa_id].startFlag = true;
      } else {
        pcscNuma[numa_id].startFlag = false;
      }

      pcscNuma[numa_id].endCol = get_row_from_index(n+1, cscColPtr, pcscNuma[numa_id].endIdx);
      // Mark imcomplete rows
      // True: imcomplete
      if (pcscNuma[numa_id].endIdx < cscColPtr[pcscNuma[numa_id].endCol + 1] - 1)  {
        pcscNuma[numa_id].endFlag = true;

      } else {
        pcscNuma[numa_id].endFlag = false;

      }

      // Cacluclate dimensions
      pcscNuma[numa_id].m = m;
      pcscNuma[numa_id].n = pcscNuma[numa_id].endCol - pcscNuma[numa_id].startCol + 1;
      pcscNuma[numa_id].nnz  = pcscNuma[numa_id].endIdx - pcscNuma[numa_id].startIdx + 1;

      printf("numa_id %d, numa_start_idx %d, numa_end_idx %d\n",
              numa_id, pcscNuma[numa_id].startIdx, pcscNuma[numa_id].endIdx);
      printf("numa_id %d, numa_start_col %d, numa_end_col %d\n",
              numa_id, pcscNuma[numa_id].startCol, pcscNuma[numa_id].endCol);
    
      numa_part_time += get_time() - tmp_time;

      // preparing data on host 
      cudaMallocHost((void**)&(pcscNuma[numa_id].val), pcscNuma[numa_id].nnz * sizeof(double));
      cudaMallocHost((void**)&(pcscNuma[numa_id].colPtr), (pcscNuma[numa_id].n + 1)*sizeof(int));
      cudaMallocHost((void**)&(pcscNuma[numa_id].rowIdx), pcscNuma[numa_id].nnz * sizeof(int));
      cudaMallocHost((void**)&(pcscNuma[numa_id].x), pcscNuma[numa_id].n * sizeof(double));
      cudaMallocHost((void**)&(pcscNuma[numa_id].y), pcscNuma[numa_id].m * sizeof(double));

      tmp_time = get_time();

      for (int i = pcscNuma[numa_id].startIdx; i <= pcscNuma[numa_id].endIdx; i++) {
        pcscNuma[numa_id].val[i - pcscNuma[numa_id].startIdx] = cscVal[i];
      }

      // for (int i = numa_start_idx[numa_id]; i <= numa_end_idx[numa_id]; i++) {
      //   numa_csrVal[numa_id][i - numa_start_idx[numa_id]] = csrVal[i];
      // }


      pcscNuma[numa_id].colPtr[0] = 0;
      pcscNuma[numa_id].colPtr[pcscNuma[numa_id].n] = pcscNuma[numa_id].nnz;
      for (int j = 1; j < pcscNuma[numa_id].n; j++) {
        pcscNuma[numa_id].colPtr[j] = cscColPtr[pcscNuma[numa_id].startCol + j] - pcscNuma[numa_id].startIdx;
      }

      for (int i = pcscNuma[numa_id].startIdx; i <= pcscNuma[numa_id].endIdx; i++) {
        pcscNuma[numa_id].rowIdx[i - pcscNuma[numa_id].startIdx] = cscRowIndex[i];
      }

      for (int i = 0; i < pcscNuma[numa_id].n; i++) {
        pcscNuma[numa_id].x[i] = x[pcscNuma[numa_id].startCol + i];
      }
    
      for (int i = 0; i < pcscNuma[numa_id].m; i++) {
        pcscNuma[numa_id].y[i] = y[i];
      }

      numa_part_time += get_time() - tmp_time;

      // print_vec(pcscNuma[numa_id].val, pcscNuma[numa_id].nnz, "cscVal"+to_string(dev_id));
      // print_vec(pcscNuma[numa_id].colPtr, pcscNuma[numa_id].n + 1, "colPtr"+to_string(dev_id));
      // print_vec(pcscNuma[numa_id].rowIdx, pcscNuma[numa_id].nnz, "rowIdx"+to_string(dev_id));
      // print_vec(pcscNuma[numa_id].x, pcscNuma[numa_id].n, "x"+to_string(dev_id));
      // print_vec(pcscNuma[numa_id].y, pcscNuma[numa_id].m, "y_before"+to_string(dev_id));
      // printf("dev_id %d, alpha %f, beta %f\n", dev_id, *alpha, *beta);

    }
  }


  struct pCSC * pcscGPU = new struct pCSC[ngpu];

  double * start_element = new double[ngpu];
  double * end_element = new double[ngpu];
  bool * start_flags = new bool[ngpu];
  bool * end_flags = new bool[ngpu];
  double * org_y = new double[ngpu];
  int * start_rows = new int[ngpu];


  omp_set_num_threads(ngpu);

  double core_time;
  double part_time;
  double merg_time;
  #pragma omp parallel default (shared) reduction(max:core_time) reduction(max:part_time) reduction(max:merg_time)
  {
    unsigned int dev_id = omp_get_thread_num();
    cudaSetDevice(dev_id);
    unsigned int hwthread = sched_getcpu();

    printf("omp thread %d, hw thread %d\n", dev_id, hwthread);  

    int numa_id = numaContext.numaMapping[dev_id];
    int local_dev_id = 0;
    for (int i = 0; i < ngpu; i++) {
      if (i == dev_id) break;
      if (numa_id == numaContext.numaMapping[i]) local_dev_id++;
    }

    cudaStream_t stream;
    cusparseStatus_t status;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    int err;
  
    double tmp_time = get_time();

    // Calculate the start and end index
    long long tmp1 = local_dev_id * pcscNuma[numa_id].nnz;
    long long tmp2 = (local_dev_id + 1) * pcscNuma[numa_id].nnz;

    pcscGPU[dev_id].startIdx = floor((double)tmp1 / numaContext.numGPUs[numa_id]);
    pcscGPU[dev_id].endIdx   = floor((double)tmp2 / numaContext.numGPUs[numa_id]) - 1;
  
    // Calculate the start and end col
    pcscGPU[dev_id].startCol = get_row_from_index(pcscNuma[numa_id].m, pcscNuma[numa_id].colPtr, pcscGPU[dev_id].startIdx);
    // Mark imcomplete rows
    // True: imcomplete
    if (pcscGPU[dev_id].startIdx > pcscNuma[numa_id].colPtr[pcscGPU[dev_id].startCol]) {
      pcscGPU[dev_id].startFlag = true;
      //start_rows[dev_id] = start_row;
    } else {
      pcscGPU[dev_id].startFlag = false;
    }
    //start_flags[dev_id] = start_flag;   

    pcscGPU[dev_id].endCol = get_row_from_index(pcscNuma[numa_id].m, pcscNuma[numa_id].colPtr, pcscGPU[dev_id].endIdx);
    // Mark imcomplete rows
    // True: imcomplete
    if (pcscGPU[dev_id].endIdx < pcscNuma[numa_id].colPtr[pcscGPU[dev_id].endCol + 1] - 1)  {
      pcscGPU[dev_id].endFlag = true;
    } else {
      pcscGPU[dev_id].endFlag = false;
    }
    
    // Cacluclate dimensions
    pcscGPU[dev_id].m = m;
    pcscGPU[dev_id].n = pcscGPU[dev_id].endCol - pcscGPU[dev_id].startCol + 1;
    pcscGPU[dev_id].nnz = pcscGPU[dev_id].endIdx - pcscGPU[dev_id].startIdx + 1;

    part_time = get_time() - tmp_time;  



    // preparing data on host 
    //cudaMallocHost((void**)&host_csrVal, dev_nnz * sizeof(double));
    //for (int i = start_idx; i <= end_idx; i++) {
    //  host_csrVal[i - start_idx] = csrVal[i];
    //}

    //cudaMallocHost((void**)&host_cscColPtr, (dev_n + 1)*sizeof(int));
    //host_csrRowPtr[0] = 0;
    //host_csrRowPtr[dev_m] = dev_nnz;
    //for (int j = 1; j < dev_m; j++) {
    //  host_csrRowPtr[j] = (int)(csrRowPtr[start_row + j] - start_idx);
    //}

    //cudaMallocHost((void**)&host_csrColIndex, dev_nnz * sizeof(int));
    //for (int i = start_idx; i <= end_idx; i++) {
    //  host_csrColIndex[i - start_idx] = csrColIndex[i];
    //}

    //cudaMallocHost((void**)&host_x, dev_n * sizeof(double));
    //for (int i = 0; i < dev_n; i++) {
    //  host_x[i] = x[i];
    //}

    //cudaMallocHost((void**)&host_y, dev_m * sizeof(double));
    //for (int i = 0; i < dev_m; i++) {
    //  host_y[i] = y[start_row + i];
    //}
    
    tmp_time = get_time();

    pcscGPU[dev_id].val = &(pcscNuma[numa_id].val[pcscGPU[dev_id].startIdx]);
    pcscGPU[dev_id].colPtr = &(pcscNuma[numa_id].colPtr[pcscGPU[dev_id].startCol]);
    pcscGPU[dev_id].rowIdx = &(pcscNuma[numa_id].rowIdx[pcscGPU[dev_id].startIdx]);
    pcscGPU[dev_id].x = &(pcscNuma[numa_id].x[pcscGPU[dev_id].startCol]);
    pcscGPU[dev_id].y = pcscNuma[numa_id].y;


    // print_vec(pcscGPU[dev_id].val, pcscGPU[dev_id].nnz, "cscVal"+to_string(dev_id));
    // print_vec(pcscGPU[dev_id].colPtr, pcscGPU[dev_id].n + 1, "colPtr"+to_string(dev_id));
    // print_vec(pcscGPU[dev_id].rowIdx, pcscGPU[dev_id].nnz, "rowIdx"+to_string(dev_id));
    // print_vec(pcscGPU[dev_id].x, pcscGPU[dev_id].n, "x"+to_string(dev_id));
    // print_vec(pcscGPU[dev_id].y, pcscGPU[dev_id].m, "y_before"+to_string(dev_id));
    // printf("dev_id %d, alpha %f, beta %f\n", dev_id, *alpha, *beta);
    // host_cscColPtr[0] = 0;
    // host_cscColPtr[dev_n] = dev_nnz;
    // for (int j = 1; j < dev_m; j++) {
    //   host_cscColPtr[j] = (int)(cscColPtr[start_col + j] - start_idx);
    // }
    //host_cscRowIndex = cscRowIndex[start_idx];
    // host_x = &x[start_col];
    // host_y = y;
  
    part_time += get_time() - tmp_time;

  
    // preparing GPU env
    cudaStreamCreate(&stream);

    status = cusparseCreate(&handle); 
    if (status != CUSPARSE_STATUS_SUCCESS) 
    { 
      printf("CUSPARSE Library initialization failed");
      //return 1; 
    } 
    status = cusparseSetStream(handle, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) 
    { 
      printf("Stream bindind failed");
      //return 1;
    } 
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) 
    { 
      printf("Matrix descriptor initialization failed");
      //return 1;
    }   
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL); 
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    
    cudaMalloc((void**)&pcscGPU[dev_id].dval,    pcscGPU[dev_id].nnz     * sizeof(double));
    cudaMalloc((void**)&pcscGPU[dev_id].dcolPtr, (pcscGPU[dev_id].n + 1) * sizeof(int)   );
    cudaMalloc((void**)&pcscGPU[dev_id].drowIdx, pcscGPU[dev_id].nnz     * sizeof(int)   );
    cudaMalloc((void**)&pcscGPU[dev_id].dx,      pcscGPU[dev_id].n       * sizeof(double)); 
    cudaMalloc((void**)&pcscGPU[dev_id].dy,      pcscGPU[dev_id].m       * sizeof(double)); 


    // tmp_time = get_time();
    // cudaMemcpyAsync(pcscGPU[dev_id].dcolPtr, pcscGPU[dev_id].colPtr, (pcscGPU[dev_id].n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream); 
    // cudaDeviceSynchronize();
    // calcCscColPtr(pcscGPU[dev_id].dcolPtr, pcscGPU[dev_id].n, pcscGPU[dev_id].startIdx, pcscGPU[dev_id].nnz, stream);
    // cudaDeviceSynchronize();
    // printf("dev_id %d, part_kernel_time = %f\n", dev_id, get_time() - tmp_time);
    // part_time += get_time() - tmp_time;  
    // printf("dev_id %d, part_time = %f\n", dev_id, part_time); 


    #pragma omp barrier
    tmp_time = get_time();

    cudaMemcpyAsync(pcscGPU[dev_id].dval,    pcscGPU[dev_id].val,    pcscGPU[dev_id].nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(pcscGPU[dev_id].dcolPtr, pcscGPU[dev_id].colPtr, (pcscGPU[dev_id].n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(pcscGPU[dev_id].drowIdx, pcscGPU[dev_id].rowIdx, pcscGPU[dev_id].nnz * sizeof(double), cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(pcscGPU[dev_id].dx,      pcscGPU[dev_id].x,      pcscGPU[dev_id].n * sizeof(double),  cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(pcscGPU[dev_id].dy,      pcscGPU[dev_id].y,      pcscGPU[dev_id].m * sizeof(double), cudaMemcpyHostToDevice, stream); 
    
    checkCudaErrors(cudaDeviceSynchronize());
    print_vec_gpu(pcscGPU[dev_id].dval, pcscGPU[dev_id].nnz, "cscVal"+to_string(dev_id));
    print_vec_gpu(pcscGPU[dev_id].dcolPtr, pcscGPU[dev_id].n + 1, "colPtr"+to_string(dev_id));
    print_vec_gpu(pcscGPU[dev_id].drowIdx, pcscGPU[dev_id].nnz, "rowIdx"+to_string(dev_id));
    print_vec_gpu(pcscGPU[dev_id].dx, pcscGPU[dev_id].n, "x"+to_string(dev_id));
    print_vec_gpu(pcscGPU[dev_id].dy, pcscGPU[dev_id].m, "y_before"+to_string(dev_id));
    printf("dev_id %d, alpha %f, beta %f\n", dev_id, *alpha, *beta);

    
  //   time_comm = get_time() - curr_time;
  //   curr_time = get_time();

  //   err = 0;
    // if (kernel == 1) {

    double * dev_csrVal;
    int * dev_csrRowPtr;
    int * dev_csrColIndex;
    checkCudaErrors(cudaMalloc((void**)&dev_csrVal,    pcscGPU[dev_id].nnz     * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&dev_csrRowPtr, (pcscGPU[dev_id].m + 1) * sizeof(int)   ));
    checkCudaErrors(cudaMalloc((void**)&dev_csrColIndex, pcscGPU[dev_id].nnz     * sizeof(int) ));

    csc2csr_gpu(m, n, nnz,
                 pcscGPU[dev_id].dval, pcscGPU[dev_id].dcolPtr, pcscGPU[dev_id].drowIdx,
                 dev_csrVal, dev_csrRowPtr, dev_csrColIndex);

    checkCudaErrors(cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            pcscGPU[dev_id].m, pcscGPU[dev_id].n, pcscGPU[dev_id].nnz, 
                            alpha, descr, dev_csrVal, 
                            dev_csrRowPtr, dev_csrColIndex, 
                            pcscGPU[dev_id].dx, beta, pcscGPU[dev_id].dy));
      
  
    checkCudaErrors(cudaDeviceSynchronize());
    printf("omp thread %d, time %f\n", dev_id, get_time() - tmp_time);
    core_time = get_time() - tmp_time;

    checkCudaErrors(cudaMemcpyAsync(pcscGPU[dev_id].y, pcscGPU[dev_id].dy, 
                    pcscGPU[dev_id].m * sizeof(double), cudaMemcpyHostToDevice, stream)); 

    checkCudaErrors(cudaDeviceSynchronize());
    print_vec_gpu(pcscGPU[dev_id].y, m, "y"+to_string(dev_id));
    #pragma omp barrier
    if (dev_id == 0) {
      for (int i = 0; i < m; i++) {
        for (int d = 0; d < ngpu; d++) {
          y[i] += pcscGPU[d].y[i];
        }
      }
    }
    merg_time = get_time() - tmp_time;

    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    cudaStreamDestroy(stream);

  }

  spmv_ret ret;
  ret.comp_time = core_time;
  ret.comm_time = 0.0;
  ret.part_time = part_time;
  ret.merg_time = merg_time;
  return ret;
}

