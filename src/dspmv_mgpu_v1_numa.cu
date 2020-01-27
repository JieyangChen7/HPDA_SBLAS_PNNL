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
_calcCsrRowPtr(int * csrRowPrt, int m, int offset, int nnz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < m; i += blockDim.x * gridDim.x) {
    csrRowPrt[i] -= offset; 
    //printf("thread %d: %d - %d\n", idx, csrRowPrt[i], offset);
  }
  if (idx == 0) {
    csrRowPrt[0] = 0;
    csrRowPrt[m] = nnz;
  }
}

void calcCsrRowPtr(int * csrRowPrt, int m, int offset, int nnz, cudaStream_t stream) {
  int thread_per_block = 256;
  int block_per_grid = ceil((float)m / thread_per_block); 
  _calcCsrRowPtr<<<block_per_grid, thread_per_block, 0, stream>>>(csrRowPrt, m, offset, nnz);
}

// spmv_ret spMspV_mgpu_v1_numa(int m, int n, int nnz, double * alpha,
//                                   double * csrVal, int * csrRowPtr, int * csrColIndex,
//                                   double * x, double * beta,
//                                   double * y,
//                                   int ngpu,
//                                   int kernel) {
//   double start = get_time();   
//   int nnz_reduced = 0;
//   vector<double> * csrVal_reduced = new vector<double>();
//   vector<int> * csrRowPtr_reduced = new vector<int>();
//   vector<int> * csrColIndex_reduced = new vector<int>();

//   double * csrVal_reduced_pin;
//   int * csrRowPtr_reduced_pin;
//   int * csrColIndex_reduced_pin;

//   csrRowPtr_reduced->push_back(0);
//   for (int i = 0; i < m; i++) {
//     for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
//       if (x[csrColIndex[j]] != 0.0) {
//   csrVal_reduced->push_back(csrVal[j]);
//         csrColIndex_reduced->push_back(csrColIndex[j]);
//   nnz_reduced ++;
//       }
//     }
//     csrRowPtr_reduced->push_back(nnz_reduced);
//   }


//   double convert_time = get_time() - start;

//   cudaMallocHost((void **)&csrVal_reduced_pin, nnz_reduced * sizeof(double));
//   cudaMallocHost((void **)&csrRowPtr_reduced_pin, (m+1) * sizeof(int));
//   cudaMallocHost((void **)&csrColIndex_reduced_pin, nnz_reduced * sizeof(int));

//   cudaMemcpy(csrVal_reduced_pin, csrVal_reduced->data(), nnz_reduced * sizeof(double), cudaMemcpyHostToHost);
//   cudaMemcpy(csrRowPtr_reduced_pin, csrRowPtr_reduced->data(), (m+1) * sizeof(int), cudaMemcpyHostToHost);
//   cudaMemcpy(csrColIndex_reduced_pin, csrColIndex_reduced->data(), nnz_reduced * sizeof(int), cudaMemcpyHostToHost);

//   delete csrVal_reduced;
//   delete csrRowPtr_reduced;
//   delete csrColIndex_reduced;

//   cout << "spMspV_mgpu_v1: nnz reduced from " << nnz << " to " << nnz_reduced << std::endl;

//   int numa_mapping[6] = {0,0,0,1,1,1};

//   spmv_ret ret =  spMV_mgpu_v1_numa(m, n, nnz_reduced, alpha,
//                csrVal_reduced_pin,
//                csrRowPtr_reduced_pin, 
//                csrColIndex_reduced_pin,
//                x, beta, y, ngpu, kernel, numa_mapping);
//   cudaFreeHost(csrVal_reduced_pin);
//   cudaFreeHost(csrRowPtr_reduced_pin);
//   cudaFreeHost(csrColIndex_reduced_pin);

//   return ret;
// }

spmv_ret spMV_mgpu_v1_numa(int m, int n, int nnz, double * alpha,
          double * csrVal, int * csrRowPtr, int * csrColIndex, 
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
  struct pCSR * pcsrNuma = new struct pCSR[numaContext.numNumaNodes];

  // figure out the number of numa nodes
 //  int num_numa_nodes = 0;
 //  for (int i = 0; i < ngpu; i++) {
 //    if (numa_mapping[i] > num_numa_nodes) {
 //      num_numa_nodes = numa_mapping[i];
 //    }
 //  }
 //  num_numa_nodes += 1;
 //  printf("# of NUMA nodes: %d\n", num_numa_nodes);
  // printf("Representive threads: ");
  // int * representive_threads = new int [num_numa_nodes];
  // for (int i = 0; i < num_numa_nodes; i++) {
 //    for (int j = 0; j < ngpu; j++) {
 //      if (numa_mapping[j] == i) {
 //        representive_threads[i] = j;
 //        break;
 //      }
 //    }
  //  printf("%d ", representive_threads[i]);
 //  }
  // printf("\n");

  // printf("# of GPU distribution: ");
  // int * num_gpus = new int [num_numa_nodes];
  // for (int j = 0; j < num_numa_nodes; j++) {
  //  num_gpus[j] = 0;
  // }
  // for (int j = 0; j < ngpu; j++) {
  //  num_gpus[numa_mapping[j]]++;
  // }
  // for (int i = 0; i < num_numa_nodes; i++) {
  //  printf("%d ", num_gpus[i]);
  // }
  // printf("\n");

  // int * workload = new int [num_numa_nodes+1];
 //  workload[0] = 0;
  // workload[1] = num_gpus[0];     
  // for (int i = 2; i < num_numa_nodes+1; i++) {
  //  workload[i] = workload[i-1] + num_gpus[i-1];
  // }
  // print_vec(workload, num_numa_nodes+1, "workload: "); 

  // double ** numa_csrVal = new double*[num_numa_nodes];
  // int ** numa_csrRowPtr = new int*[num_numa_nodes]; 
  // int ** numa_csrColIndex = new int*[num_numa_nodes];
  // int * numa_m = new int[num_numa_nodes];
  // int * numa_n = new int[num_numa_nodes];
  // int * numa_nnz = new int[num_numa_nodes];

 //   int * numa_start_idx = new int[num_numa_nodes];
  // int * numa_end_idx = new int[num_numa_nodes];
    
  // int * numa_start_row = new int[num_numa_nodes];
  // int * numa_end_row = new int[num_numa_nodes];

  // bool * numa_start_flag = new bool[num_numa_nodes];
  // bool * numa_end_flag = new bool[num_numa_nodes];

  // double ** numa_x = new double*[num_numa_nodes];
  // double ** numa_y = new double*[num_numa_nodes];
  
  // double * numa_org_y = new double[num_numa_nodes];


  
  omp_set_num_threads(ngpu);
  #pragma omp parallel default (shared) reduction(max:numa_part_time)
  {
    string s;
    unsigned int dev_id = omp_get_thread_num();
    unsigned int hwthread = sched_getcpu();
    //bool is_represetative = false;
    // int numa_id;
    // for (int i = 0; i < num_numa_nodes; i++) {
    //   if (representive_threads[i] == dev_id)  {
    //     is_represetative = true;
    //     numa_id = i;
    //   }
    // }

    int numa_id = numaContext.numaMapping[dev_id];
  
    numa_part_time = 0;
  
    

    //if (is_represetative) {
    if (numaContext.representiveThreads[dev_id]) {
      // printf("represent thread %d hw thread %d\n", dev_id, hwthread);

      double tmp_time = get_time();

      int tmp1 = numaContext.workload[numa_id] * nnz;
      int tmp2 = numaContext.workload[numa_id + 1] * nnz;

      pcsrNuma[numa_id].startIdx = floor((double)tmp1 / ngpu);
      pcsrNuma[numa_id].endIdx = floor((double)tmp2 / ngpu) - 1;
      // numa_start_idx[numa_id] = floor((double)tmp1 / ngpu);
      // numa_end_idx[numa_id]   = floor((double)tmp2 / ngpu) - 1;



      // Calculate the start and end row
      pcsrNuma[numa_id].startRow = get_row_from_index(m+1, csrRowPtr, pcsrNuma[numa_id].startIdx);
      //numa_start_row[numa_id] = get_row_from_index(m+1, csrRowPtr, numa_start_idx[numa_id]);
      // Mark imcomplete rows
      // True: imcomplete
      if (pcsrNuma[numa_id].startIdx > csrRowPtr[pcsrNuma[numa_id].startRow]) {
      //if (numa_start_idx[numa_id] > csrRowPtr[numa_start_row[numa_id]]) {
        pcsrNuma[numa_id].startFlag = true;
        //numa_start_flag[numa_id] = true;
        pcsrNuma[numa_id].org_y = y[pcsrNuma[numa_id].startRow];
        //numa_org_y[numa_id] = y[numa_start_row[numa_id]];
      } else {
        pcsrNuma[numa_id].startFlag = false;
        //numa_start_flag[numa_id] = false;
      }

      pcsrNuma[numa_id].endRow = get_row_from_index(m+1, csrRowPtr, pcsrNuma[numa_id].endIdx);
      //numa_end_row[numa_id] = get_row_from_index(m+1, csrRowPtr, numa_end_idx[numa_id]);
      // Mark imcomplete rows
      // True: imcomplete
      if (pcsrNuma[numa_id].endIdx < csrRowPtr[pcsrNuma[numa_id].endRow + 1] - 1)  {
      //if (numa_end_idx[numa_id] < csrRowPtr[numa_end_row[numa_id] + 1] - 1)  {
        pcsrNuma[numa_id].endFlag = true;
        //numa_end_flag[numa_id] = true;
      } else {
        pcsrNuma[numa_id].endFlag = false;
        //numa_end_flag[numa_id] = false;
      }

      //print_vec(csrRowPtr+m+1-5, 5, "last" + to_string(numa_id));

      // Cacluclate dimensions

      pcsrNuma[numa_id].m = pcsrNuma[numa_id].endRow - pcsrNuma[numa_id].startRow + 1;
      pcsrNuma[numa_id].n = n;
      pcsrNuma[numa_id].nnz  = pcsrNuma[numa_id].endIdx - pcsrNuma[numa_id].startIdx + 1;

      // numa_m[numa_id] = numa_end_row[numa_id] - numa_start_row[numa_id] + 1;
      // numa_n[numa_id] = n;
      // numa_nnz[numa_id]   = (int)(numa_end_idx[numa_id] - numa_start_idx[numa_id] + 1);

      // printf("numa_id %d, numa_start_idx %d, numa_end_idx %d\n",numa_id, pcsrNuma[numa_id].startIdx, pcsrNuma[numa_id].endIdx);
      // printf("numa_id %d, numa_start_row %d, numa_end_row %d\n",numa_id, pcsrNuma[numa_id].startRow, pcsrNuma[numa_id].endRow);
    

      numa_part_time += get_time() - tmp_time;

      // preparing data on host 
      checkCudaErrors(cudaMallocHost((void**)&(pcsrNuma[numa_id].val), pcsrNuma[numa_id].nnz * sizeof(double)));
      checkCudaErrors(cudaMallocHost((void**)&(pcsrNuma[numa_id].rowPtr), (pcsrNuma[numa_id].m + 1)*sizeof(int)));
      checkCudaErrors(cudaMallocHost((void**)&(pcsrNuma[numa_id].colIdx), pcsrNuma[numa_id].nnz * sizeof(int)));
      checkCudaErrors(cudaMallocHost((void**)&(pcsrNuma[numa_id].x), pcsrNuma[numa_id].n * sizeof(double)));
      checkCudaErrors(cudaMallocHost((void**)&(pcsrNuma[numa_id].y), pcsrNuma[numa_id].m * sizeof(double)));


      // cudaMallocHost((void**)&numa_csrVal[numa_id], numa_nnz[numa_id] * sizeof(double));
      // cudaMallocHost((void**)&numa_csrRowPtr[numa_id], (numa_m[numa_id] + 1)*sizeof(int));
      // cudaMallocHost((void**)&numa_csrColIndex[numa_id], numa_nnz[numa_id] * sizeof(int));
      // cudaMallocHost((void**)&numa_x[numa_id], numa_n[numa_id] * sizeof(double));
      // cudaMallocHost((void**)&numa_y[numa_id], numa_m[numa_id] * sizeof(double));

      tmp_time = get_time();


      for (int i = pcsrNuma[numa_id].startIdx; i <= pcsrNuma[numa_id].endIdx; i++) {
        pcsrNuma[numa_id].val[i - pcsrNuma[numa_id].startIdx] = csrVal[i];
      }

      // for (int i = numa_start_idx[numa_id]; i <= numa_end_idx[numa_id]; i++) {
      //   numa_csrVal[numa_id][i - numa_start_idx[numa_id]] = csrVal[i];
      // }


      pcsrNuma[numa_id].rowPtr[0] = 0;
      pcsrNuma[numa_id].rowPtr[pcsrNuma[numa_id].m] = pcsrNuma[numa_id].nnz;
      for (int j = 1; j < pcsrNuma[numa_id].m; j++) {
        pcsrNuma[numa_id].rowPtr[j] = csrRowPtr[pcsrNuma[numa_id].startRow + j] - pcsrNuma[numa_id].startIdx;
      }

      // numa_csrRowPtr[numa_id][0] = 0;
      // numa_csrRowPtr[numa_id][numa_m[numa_id]] = numa_nnz[numa_id];
      // for (int j = 1; j < numa_m[numa_id]; j++) {
      //   numa_csrRowPtr[numa_id][j] = (int)(csrRowPtr[numa_start_row[numa_id] + j] - numa_start_idx[numa_id]);
      // }

      //print_vec(csrRowPtr, nnz+1, "org rowptr");
      //print_vec(numa_csrRowPtr[numa_id], numa_nnz[numa_id], "numa rowptr " + to_string(numa_id)); 
  
      for (int i = pcsrNuma[numa_id].startIdx; i <= pcsrNuma[numa_id].endIdx; i++) {
        pcsrNuma[numa_id].colIdx[i - pcsrNuma[numa_id].startIdx] = csrColIndex[i];
      }

      // for (int i = numa_start_idx[numa_id]; i <= numa_end_idx[numa_id]; i++) {
      //   numa_csrColIndex[numa_id][i - numa_start_idx[numa_id]] = csrColIndex[i];
      // }

      for (int i = 0; i < pcsrNuma[numa_id].n; i++) {
        pcsrNuma[numa_id].x[i] = x[i];
      }


      // for (int i = 0; i < numa_n[numa_id]; i++) {
      //   numa_x[numa_id][i] = x[i];
      // }
    
      for (int i = 0; i < pcsrNuma[numa_id].m; i++) {
        pcsrNuma[numa_id].y[i] = y[pcsrNuma[numa_id].startRow + i];
      }

      // for (int i = 0; i < numa_m[numa_id]; i++) {
      //   numa_y[numa_id][i] = y[numa_start_row[numa_id] + i];
      // }
      numa_part_time += get_time() - tmp_time;

    }
  }
  





  
  struct pCSR * pcsrGPU = new struct pCSR[ngpu];

  double * start_element = new double[ngpu];
  double * end_element = new double[ngpu];
  bool * start_flags = new bool[ngpu];
  bool * end_flags = new bool[ngpu];
  double * org_y = new double[ngpu];
  int * start_rows = new int[ngpu];




  omp_set_num_threads(ngpu);
  #pragma omp parallel default (shared) reduction(max:comp_time) reduction(max:part_time) reduction(max:merg_time)
  {
    unsigned int dev_id = omp_get_thread_num();
    checkCudaErrors(cudaSetDevice(dev_id));
    unsigned int hwthread = sched_getcpu();

    float elapsedTime;

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

    cudaEvent_t comp_start, comp_stop;
    cudaEvent_t comm_start, comm_stop;

    checkCudaErrors(cudaEventCreate(&comp_start));
    checkCudaErrors(cudaEventCreate(&comp_stop));
    checkCudaErrors(cudaEventCreate(&comm_start));
    checkCudaErrors(cudaEventCreate(&comm_stop));

    int numa_id = numaContext.numaMapping[dev_id];
    int local_dev_id = 0;
    for (int i = 0; i < ngpu; i++) {
      if (i == dev_id) break;
      if (numa_id == numaContext.numaMapping[i]) local_dev_id++;
    }

    // int numa_id = numa_mapping[dev_id];
    // int local_dev_id = 0;
    // for (int i = 0; i < ngpu; i++) {
    //   if (i == dev_id) break;
    //   if (numa_id == numa_mapping[i]) local_dev_id++;
    // }


    // printf("omp thread %d, hw thread %d, numa_id %d, local_id %d\n", dev_id, hwthread, numa_id, local_dev_id);  

    // int start_idx, end_idx;
    // int start_row, end_row;
    // bool start_flag, end_flag; 


    // double * host_csrVal;
    // int    * host_csrRowPtr;
    // int    * host_csrColIndex;
    // double * host_x;
    // double * host_y;

    // double * dev_csrVal;
    // int    * dev_csrRowPtr;
    // int    * dev_csrColIndex;

    // int    dev_nnz, dev_m, dev_n;

    
    // double * dev_x;
    // double * dev_y;
    // double y2;

    double tmp_time = get_time();

    // Calculate the start and end index
    int tmp1 = local_dev_id * pcsrNuma[numa_id].nnz;
    int tmp2 = (local_dev_id + 1) * pcsrNuma[numa_id].nnz;

    pcsrGPU[dev_id].startIdx = floor((double)tmp1 / numaContext.numGPUs[numa_id]);
    pcsrGPU[dev_id].endIdx   = floor((double)tmp2 / numaContext.numGPUs[numa_id]) - 1;


    // int tmp1 = local_dev_id * numa_nnz[numa_id];
    // int tmp2 = (local_dev_id + 1) * numa_nnz[numa_id];

    // start_idx = floor((double)tmp1 / num_gpus[numa_id]);
    // end_idx   = floor((double)tmp2 / num_gpus[numa_id]) - 1;
  
    // Calculate the start and end row
    pcsrGPU[dev_id].startRow = get_row_from_index(pcsrNuma[numa_id].m, pcsrNuma[numa_id].rowPtr, pcsrGPU[dev_id].startIdx);
    //start_row = get_row_from_index(numa_m[numa_id], numa_csrRowPtr[numa_id], start_idx);
    // Mark imcomplete rows
    // True: imcomplete
    if (pcsrGPU[dev_id].startIdx > pcsrNuma[numa_id].rowPtr[pcsrGPU[dev_id].startRow]) {
    //if (start_idx > numa_csrRowPtr[numa_id][start_row]) {
      pcsrGPU[dev_id].startFlag = true;
      //start_flag = true;
      //y2 = y[start_row];
      pcsrGPU[dev_id].org_y = pcsrNuma[numa_id].y[pcsrGPU[dev_id].startRow];
      //org_y[dev_id] = y[start_row]; //use dev_id for global merge
      //start_rows[dev_id] = start_row;
    } else {
      pcsrGPU[dev_id].startFlag = false;
      //start_flag = false;
    }

    if (local_dev_id == 0) {
      pcsrGPU[dev_id].startFlag = pcsrNuma[numa_id].startFlag; // see if numa block is complete
    }
  
    // if (local_dev_id == 0) {
    //   start_flag = numa_start_flag[numa_id]; // see if numa block is complete
    // }
    // start_flags[dev_id] = start_flag;   

    pcsrGPU[dev_id].endRow = get_row_from_index(pcsrNuma[numa_id].m, pcsrNuma[numa_id].rowPtr, pcsrGPU[dev_id].endIdx);
    //end_row = get_row_from_index(numa_m[numa_id], numa_csrRowPtr[numa_id], end_idx);
    // Mark imcomplete rows
    // True: imcomplete

    if (pcsrGPU[dev_id].endIdx < pcsrNuma[numa_id].rowPtr[pcsrGPU[dev_id].endRow + 1] - 1)  {
    //if (end_idx < numa_csrRowPtr[numa_id][end_row + 1] - 1)  {
      pcsrGPU[dev_id].endFlag = true;
      //end_flag = true;
    } else {
      pcsrGPU[dev_id].endFlag = false;
      //end_flag = false;
    }

    if (local_dev_id + 1 == numaContext.numGPUs[numa_id]) {
      pcsrGPU[dev_id].endFlag = pcsrNuma[numa_id].endFlag;
    }
    

    // if (local_dev_id+1 == num_gpus[numa_id]) {
    //   end_flag = numa_end_flag[numa_id];
    // }
    
    // // Cacluclate dimensions
    pcsrGPU[dev_id].m = pcsrGPU[dev_id].endRow - pcsrGPU[dev_id].startRow + 1;
    pcsrGPU[dev_id].n = n;
    pcsrGPU[dev_id].nnz  = pcsrGPU[dev_id].endIdx - pcsrGPU[dev_id].startIdx + 1;
    // dev_m = end_row - start_row + 1;
    // dev_n = n;
    // dev_nnz   = (int)(end_idx - start_idx + 1);

    // printf("omp thread %d, dev_m %d, dev_n %d, dev_nnz %d, start_idx %d, end_idx %d, start_row %d, end_row %d\n", 
    //         dev_id, pcsrGPU[dev_id].m, pcsrGPU[dev_id].n, pcsrGPU[dev_id].nnz, pcsrGPU[dev_id].startIdx, pcsrGPU[dev_id].endIdx, 
    //         pcsrGPU[dev_id].startRow, pcsrGPU[dev_id].endRow);


    // preparing data on host 

    //tmp_time = get_time();
    pcsrGPU[dev_id].val = &(pcsrNuma[numa_id].val[pcsrGPU[dev_id].startIdx]);
    pcsrGPU[dev_id].rowPtr = &(pcsrNuma[numa_id].rowPtr[pcsrGPU[dev_id].startRow]);
    pcsrGPU[dev_id].colIdx = &(pcsrNuma[numa_id].colIdx[pcsrGPU[dev_id].startIdx]);
    pcsrGPU[dev_id].x = pcsrNuma[numa_id].x;
    pcsrGPU[dev_id].y = &(pcsrNuma[numa_id].y[pcsrGPU[dev_id].startRow]);





    // host_csrVal = &numa_csrVal[numa_id][start_idx];
    // host_csrRowPtr = &numa_csrRowPtr[numa_id][start_row];
    // host_csrColIndex = &numa_csrColIndex[numa_id][start_idx];
    // host_x = numa_x[numa_id];
    // host_y = &numa_y[numa_id][start_row];
    part_time = get_time() - tmp_time;  


    checkCudaErrors(cudaMalloc((void**)&(pcsrGPU[dev_id].dval),    pcsrGPU[dev_id].nnz     * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&(pcsrGPU[dev_id].drowPtr), (pcsrGPU[dev_id].m + 1) * sizeof(int)   ));
    checkCudaErrors(cudaMalloc((void**)&(pcsrGPU[dev_id].dcolIdx), pcsrGPU[dev_id].nnz     * sizeof(int)   ));
    checkCudaErrors(cudaMalloc((void**)&(pcsrGPU[dev_id].dx),      pcsrGPU[dev_id].n       * sizeof(double))); 
    checkCudaErrors(cudaMalloc((void**)&(pcsrGPU[dev_id].dy),      pcsrGPU[dev_id].m       * sizeof(double))); 
    checkCudaErrors(cudaMallocHost((void**)&(pcsrGPU[dev_id].py),  pcsrGPU[dev_id].m       * sizeof(double)));


    if (part_opt == 0) {

      checkCudaErrors(cudaMallocHost((void**)&(pcsrGPU[dev_id].host_csrRowPtr), (pcsrGPU[dev_id].m + 1)*sizeof(int)));

      tmp_time = get_time();
      pcsrGPU[dev_id].host_csrRowPtr[0] = 0;
      pcsrGPU[dev_id].host_csrRowPtr[pcsrGPU[dev_id].m] = pcsrGPU[dev_id].nnz;
      for (int j = 1; j < pcsrGPU[dev_id].m; j++) {
        pcsrGPU[dev_id].host_csrRowPtr[j] = pcsrGPU[dev_id].rowPtr[j] - pcsrGPU[dev_id].startIdx;
      }
      part_time += get_time() - tmp_time;
      checkCudaErrors(cudaEventRecord(comm_start, stream));
      checkCudaErrors(cudaMemcpyAsync(pcsrGPU[dev_id].drowPtr, pcsrGPU[dev_id].host_csrRowPtr, (pcsrGPU[dev_id].m + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
      checkCudaErrors(cudaEventRecord(comm_stop, stream));
      checkCudaErrors(cudaFreeHost(pcsrGPU[dev_id].host_csrRowPtr));
    }

    if (part_opt == 1) {
      checkCudaErrors(cudaEventRecord(comm_start, stream));
      checkCudaErrors(cudaMemcpyAsync(pcsrGPU[dev_id].drowPtr, pcsrGPU[dev_id].rowPtr, (pcsrGPU[dev_id].m + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
      checkCudaErrors(cudaEventRecord(comm_stop, stream));
      checkCudaErrors(cudaDeviceSynchronize());
      tmp_time = get_time();
      calcCsrRowPtr(pcsrGPU[dev_id].drowPtr, pcsrGPU[dev_id].m, pcsrGPU[dev_id].startIdx, pcsrGPU[dev_id].nnz, stream);
      checkCudaErrors(cudaDeviceSynchronize());
      part_time += get_time() - tmp_time;
    }

    checkCudaErrors(cudaEventSynchronize(comm_stop));
    elapsedTime = 0.0;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, comm_start, comm_stop));
    elapsedTime /= 1000.0;
    comm_time += elapsedTime;

    #pragma omp barrier
    tmp_time = get_time();


    // original partition*******************************************
/*    int tmp1 = dev_id * nnz;
    int tmp2 = (dev_id + 1) * nnz;

    start_idx = floor((double)tmp1 / ngpu);
    end_idx   = floor((double)tmp2 / ngpu) - 1;

    // Calculate the start and end row
    start_row = get_row_from_index(m, csrRowPtr, start_idx);
    // Mark imcomplete rows
    // True: imcomplete
    if (start_idx > csrRowPtr[start_row]) {
      start_flag = true;
      y2 = y[start_row];
      org_y[dev_id] = y[start_row]; //use dev_id for global merge
      start_rows[dev_id] = start_row;
    } else {
      start_flag = false;
    }
    start_flags[dev_id] = start_flag;

    end_row = get_row_from_index(m, csrRowPtr, end_idx);
    // Mark imcomplete rows
    // True: imcomplete
    if (end_idx < csrRowPtr[end_row + 1] - 1)  {
      end_flag = true;
    } else {
      end_flag = false;
    }
    
    // Cacluclate dimensions
    dev_m = end_row - start_row + 1;
    dev_n = n;
    dev_nnz   = (int)(end_idx - start_idx + 1);


    

    part_time = get_time() - tmp_time;  

    printf("omp thread %d, dev_m %d, dev_n %d, dev_nnz %d, start_idx %d, end_idx %d, start_row %d, end_row %d\n", dev_id, dev_m, dev_n, dev_nnz, start_idx, end_idx, start_row, end_row);

    // preparing data on host 
    cudaMallocHost((void**)&host_csrVal, dev_nnz * sizeof(double));
    cudaMallocHost((void**)&host_csrRowPtr, (dev_m + 1)*sizeof(int));
    cudaMallocHost((void**)&host_csrColIndex, dev_nnz * sizeof(int));
    cudaMallocHost((void**)&host_x, dev_n * sizeof(double));
    cudaMallocHost((void**)&host_y, dev_m * sizeof(double));

    tmp_time = get_time();
    for (int i = start_idx; i <= end_idx; i++) {
      host_csrVal[i - start_idx] = csrVal[i];
    }
    //host_csrVal = &numa_csrVal[numa_id][start_idx];

    host_csrRowPtr[0] = 0;
    host_csrRowPtr[dev_m] = dev_nnz;
    for (int j = 1; j < dev_m; j++) {
      host_csrRowPtr[j] = (int)(csrRowPtr[start_row + j] - start_idx);
    }
    //host_csrRowPtr = &numa_csrRowPtr[numa_id][start_row];

    printf("dev %d: %d %d %d %d %d\n", host_csrRowPtr[0],host_csrRowPtr[1],host_csrRowPtr[2],host_csrRowPtr[3],host_csrRowPtr[4]);

    for (int i = start_idx; i <= end_idx; i++) {
      host_csrColIndex[i - start_idx] = csrColIndex[i];
    }
    //host_csrColIndex = &numa_csrColIndex[numa_id][start_idx];

    for (int i = 0; i < dev_n; i++) {
      host_x[i] = x[i];
    }
    //host_x = numa_x[numa_id];

    for (int i = 0; i < dev_m; i++) {
      host_y[i] = y[start_row + i];
    }
    //host_y = &numa_y[numa_id][start_row];

    part_time += get_time() - tmp_time;
*/
    // end of original partition*********************************

    

    
    
  //   // cudaMalloc((void**)&dev_csrVal,      dev_nnz     * sizeof(double));
  //   // cudaMalloc((void**)&dev_csrRowPtr,   (dev_m + 1) * sizeof(int)   );
  //   // cudaMalloc((void**)&dev_csrColIndex, dev_nnz     * sizeof(int)   );
  //   // cudaMalloc((void**)&dev_x,           dev_n       * sizeof(double)); 
  //   // cudaMalloc((void**)&dev_y,           dev_m       * sizeof(double)); 

    checkCudaErrors(cudaEventRecord(comm_start, stream));
    checkCudaErrors(cudaMemcpyAsync(pcsrGPU[dev_id].dcolIdx, pcsrGPU[dev_id].colIdx, pcsrGPU[dev_id].nnz * sizeof(int), cudaMemcpyHostToDevice, stream)); 
    checkCudaErrors(cudaMemcpyAsync(pcsrGPU[dev_id].dval, pcsrGPU[dev_id].val, pcsrGPU[dev_id].nnz * sizeof(double), cudaMemcpyHostToDevice, stream)); 
    checkCudaErrors(cudaMemcpyAsync(pcsrGPU[dev_id].dy, pcsrGPU[dev_id].y, pcsrGPU[dev_id].m*sizeof(double),  cudaMemcpyHostToDevice, stream)); 
    checkCudaErrors(cudaMemcpyAsync(pcsrGPU[dev_id].dx, pcsrGPU[dev_id].x, pcsrGPU[dev_id].n*sizeof(double), cudaMemcpyHostToDevice, stream)); 
    checkCudaErrors(cudaEventRecord(comm_stop, stream));

    // cudaMemcpyAsync(dev_csrColIndex, host_csrColIndex, dev_nnz * sizeof(int), cudaMemcpyHostToDevice, stream); 
    // cudaMemcpyAsync(dev_csrVal, host_csrVal, dev_nnz * sizeof(double), cudaMemcpyHostToDevice, stream); 
    // cudaMemcpyAsync(dev_y, host_y, dev_m*sizeof(double),  cudaMemcpyHostToDevice, stream); 
    // cudaMemcpyAsync(dev_x, host_x, dev_n*sizeof(double), cudaMemcpyHostToDevice, stream); 
    
  
    checkCudaErrors(cudaDeviceSynchronize());
    //print_vec_gpu(dev_csrRowPtr, 5, "csrRowPtr"+to_string(dev_id));
    //print_vec_gpu(dev_csrVal, 5, "csrVal"+to_string(dev_id));
    //print_vec_gpu(dev_csrColIndex, 5, "csrColIndex"+to_string(dev_id));
    //print_vec_gpu(dev_x, 5, "x"+to_string(dev_id));
    //print_vec_gpu(dev_y, 5, "y_before"+to_string(dev_id));
    //printf("dev_id %d, alpha %f, beta %f\n", dev_id, *alpha, *beta);


    //calcCsrRowPtr(dev_csrRowPtr, dev_m, start_idx, dev_nnz, stream);
    //cudaDeviceSynchronize();
  
      
    //print_vec_gpu(dev_x, dev_n, "x"+to_string(dev_id));
  
    checkCudaErrors(cudaEventRecord(comp_start, stream));
    checkCudaErrors(cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              pcsrGPU[dev_id].m, pcsrGPU[dev_id].n, pcsrGPU[dev_id].nnz, 
                              alpha, descr, pcsrGPU[dev_id].dval, 
                              pcsrGPU[dev_id].drowPtr, pcsrGPU[dev_id].dcolIdx, 
                              pcsrGPU[dev_id].dx, beta, pcsrGPU[dev_id].dy));
    checkCudaErrors(cudaEventRecord(comp_stop, stream));

    checkCudaErrors(cudaEventSynchronize(comm_stop));
    elapsedTime = 0.0;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, comm_start, comm_stop));
    elapsedTime /= 1000.0;
    comm_time += elapsedTime;

    checkCudaErrors(cudaEventSynchronize(comp_stop));
    elapsedTime = 0.0;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, comp_start, comp_stop));
    elapsedTime /= 1000.0;
    comp_time += elapsedTime;

    checkCudaErrors(cudaDeviceSynchronize());
    //print_vec_gpu(dev_y, 5, "y"+to_string(dev_id));
    // printf("omp thread %d, time %f\n", dev_id, get_time() - tmp_time);
    //comp_time = get_time() - tmp_time;
    // GPU based merge
    tmp_time = get_time();
    
    if (merg_opt == 1) {
      double * dev_y_no_overlap = pcsrGPU[dev_id].dy;
      int dev_m_no_overlap = pcsrGPU[dev_id].m;
      int start_row_no_overlap = pcsrNuma[numa_id].startRow + pcsrGPU[dev_id].startRow;

      // double * dev_y_no_overlap = dev_y;
      // int dev_m_no_overlap = dev_m;
      // int start_row_no_overlap = numa_start_row[numa_id] + start_row;
      //int start_row_no_overlap = start_row;
      if (pcsrGPU[dev_id].startFlag) {
        dev_y_no_overlap += 1;
        start_row_no_overlap += 1;
        dev_m_no_overlap -= 1;
        checkCudaErrors(cudaMemcpyAsync(start_element+dev_id, pcsrGPU[dev_id].dy, sizeof(double), cudaMemcpyDeviceToHost, stream));
        //cudaMemcpyAsync(start_element+dev_id, dev_y, sizeof(double), cudaMemcpyDeviceToHost, stream);
      }
      checkCudaErrors(cudaMemcpyAsync(y+start_row_no_overlap, dev_y_no_overlap, dev_m_no_overlap*sizeof(double),  cudaMemcpyDeviceToHost, stream));
      checkCudaErrors(cudaDeviceSynchronize());
      #pragma omp barrier
      if (dev_id == 0) {
        for (int i = 0; i < ngpu; i++) {
          if (pcsrGPU[i].startFlag) {
            y[pcsrNuma[numaContext.numaMapping[i]].startRow + pcsrGPU[i].startRow] += (start_element[i] - (*beta) * pcsrGPU[i].org_y); 
            //y[start_rows[i]] += (start_element[i] - (*beta) * org_y[i]);
          } 

          // if (start_flags[i]) {
          //   y[numa_start_row[numa_mapping[i]] + start_rows[i]] += (start_element[i] - (*beta) * org_y[i]); 
          //   //y[start_rows[i]] += (start_element[i] - (*beta) * org_y[i]);
          // } 
        }
      }
    }

    if (merg_opt == 0) {
      //  CPU based merge
      checkCudaErrors(cudaMemcpyAsync(pcsrGPU[dev_id].py, pcsrGPU[dev_id].dy, pcsrGPU[dev_id].m * sizeof(double),  cudaMemcpyDeviceToHost, stream));
      checkCudaErrors(cudaDeviceSynchronize());
      //printf("thread %d time: %f\n", dev_id,  get_time() - tmp_time);
      #pragma omp critical
      {
        double tmp = 0.0;
        if (pcsrGPU[dev_id].startFlag) {
          tmp = y[pcsrGPU[dev_id].startRow];
        }
        for (int i = 0; i < pcsrNuma[numa_id].m; i++) {
          y[pcsrGPU[dev_id].startRow + i] = pcsrGPU[dev_id].py[i];
        }
        if (pcsrGPU[dev_id].startFlag) {
          //y[pcsrGPU[dev_id].startRow] += tmp;
          y[pcsrGPU[dev_id].startRow] -= tmp * (*beta);
        }
      }
    }
    
  
    merg_time = get_time() - tmp_time;

    //cudaProfilerStop();

    checkCudaErrors(cudaFree(pcsrGPU[dev_id].dval));
    checkCudaErrors(cudaFree(pcsrGPU[dev_id].drowPtr));
    checkCudaErrors(cudaFree(pcsrGPU[dev_id].dcolIdx));
    checkCudaErrors(cudaFree(pcsrGPU[dev_id].dx));
    checkCudaErrors(cudaFree(pcsrGPU[dev_id].dy));
    checkCudaErrors(cudaFreeHost(pcsrGPU[dev_id].py));


          
    //cudaFreeHost(host_csrRowPtr);
    //cudaFreeHost(host_csrVal);
    //cudaFreeHost(host_csrColIndex);
    //cudaFreeHost(host_x);
    //cudaFreeHost(host_y);

    checkCudaErrors(cudaEventDestroy(comp_start));
    checkCudaErrors(cudaEventDestroy(comp_stop));
    checkCudaErrors(cudaEventDestroy(comm_start));
    checkCudaErrors(cudaEventDestroy(comm_stop));

    checkCudaErrors(cusparseDestroyMatDescr(descr));
    checkCudaErrors(cusparseDestroy(handle));
    checkCudaErrors(cudaStreamDestroy(stream));

  }

  printf("end part time: %f\n", part_time);
  //cout << "time_parse = " << time_parse << ", time_comm = " << time_comm << ", time_comp = "<< time_comp <<", time_post = " << time_post << endl;
  for (int numa_id = 0; numa_id < numaContext.numNumaNodes; numa_id++) {
    checkCudaErrors(cudaFreeHost(pcsrNuma[numa_id].val));
    checkCudaErrors(cudaFreeHost(pcsrNuma[numa_id].rowPtr));
    checkCudaErrors(cudaFreeHost(pcsrNuma[numa_id].colIdx));
    checkCudaErrors(cudaFreeHost(pcsrNuma[numa_id].x));
    checkCudaErrors(cudaFreeHost(pcsrNuma[numa_id].y));
  }

  spmv_ret ret;
  ret.comp_time = comp_time;
  ret.comm_time = 0.0;
  ret.part_time = part_time;
  ret.merg_time = merg_time;
  ret.numa_part_time = numa_part_time;
  return ret;
}

