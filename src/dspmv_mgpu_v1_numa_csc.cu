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
using namespace std;


/*
spmv_ret spMspV_mgpu_v1_numa(int m, int n, long long nnz, double * alpha,
                                  double * csrVal, long long * csrRowPtr, int * csrColIndex,
                                  double * x, double * beta,
                                  double * y,
                                  int ngpu,
                                  int kernel) {
  double start = get_time();	 
  long long nnz_reduced = 0;
  vector<double> * csrVal_reduced = new vector<double>();
  vector<long long> * csrRowPtr_reduced = new vector<long long>();
  vector<int> * csrColIndex_reduced = new vector<int>();

  double * csrVal_reduced_pin;
  long long * csrRowPtr_reduced_pin;
  int * csrColIndex_reduced_pin;

  csrRowPtr_reduced->push_back(0);
  for (int i = 0; i < m; i++) {
    for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
      if (x[csrColIndex[j]] != 0.0) {
	csrVal_reduced->push_back(csrVal[j]);
        csrColIndex_reduced->push_back(csrColIndex[j]);
	nnz_reduced ++;
      }
    }
    csrRowPtr_reduced->push_back(nnz_reduced);
  }


  double convert_time = get_time() - start;

  cudaMallocHost((void **)&csrVal_reduced_pin, nnz_reduced * sizeof(double));
  cudaMallocHost((void **)&csrRowPtr_reduced_pin, (m+1) * sizeof(long long));
  cudaMallocHost((void **)&csrColIndex_reduced_pin, nnz_reduced * sizeof(int));

  cudaMemcpy(csrVal_reduced_pin, csrVal_reduced->data(), nnz_reduced * sizeof(double), cudaMemcpyHostToHost);
  cudaMemcpy(csrRowPtr_reduced_pin, csrRowPtr_reduced->data(), (m+1) * sizeof(long long), cudaMemcpyHostToHost);
  cudaMemcpy(csrColIndex_reduced_pin, csrColIndex_reduced->data(), nnz_reduced * sizeof(int), cudaMemcpyHostToHost);

  delete csrVal_reduced;
  delete csrRowPtr_reduced;
  delete csrColIndex_reduced;

  cout << "spMspV_mgpu_v1: nnz reduced from " << nnz << " to " << nnz_reduced << std::endl;

  spmv_ret ret =  spMV_mgpu_v1_numa(m, n, nnz_reduced, alpha,
               csrVal_reduced_pin,
               csrRowPtr_reduced_pin, 
               csrColIndex_reduced_pin,
               x, beta, y, ngpu, kernel);
  cudaFreeHost(csrVal_reduced_pin);
  cudaFreeHost(csrRowPtr_reduced_pin);
  cudaFreeHost(csrColIndex_reduced_pin);

  return ret;
}
*/
spmv_ret spMV_mgpu_v1_numa_csc(int m, int n, long long nnz, double * alpha,
				  double * cscVal, long long * cscColPtr, int * cscRowIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  int kernel){

  double curr_time = 0.0;
  double time_comm = 0.0;
  double time_comp = 0.0;
	//printf("test0\n");
	double * start_element = new double[ngpu];
	double * end_element = new double[ngpu];
	bool * start_flags = new bool[ngpu];
	bool * end_flags = new bool[ngpu];
	double * org_y = new double[ngpu];
	int * start_rows = new int[ngpu];
	//printf("test01\n");

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

		long long  start_idx, end_idx;
		int start_col, end_col;
		bool start_flag, end_flag; 


		double * host_cscVal;
		int    * host_cscColPtr;
		int    * host_cscRowIndex;
	  double * host_x;
	  double * host_y;

		double * dev_cscVal;
		int    * dev_cscColPtr;
		int    * dev_cscRowIndex;

		int    dev_nnz, dev_m, dev_n;

	  
		double * dev_x;
		double * dev_y;
		double y2;

		cudaStream_t stream;
		cusparseStatus_t status;
		cusparseHandle_t handle;
		cusparseMatDescr_t descr;
		int err;


    double tmp_time = get_time();

		// Calculate the start and end index
		long long tmp1 = dev_id * nnz;
		long long tmp2 = (dev_id + 1) * nnz;

		//double tmp3 = (double)(tmp1 / ngpu);
		//double tmp4 = (double)(tmp2 / ngpu);

		start_idx = floor((double)tmp1 / ngpu);
		end_idx   = floor((double)tmp2 / ngpu) - 1;
	
		// Calculate the start and end col
	  start_col = get_row_from_index(m, cscColPtr, start_idx);
		// Mark imcomplete rows
		// True: imcomplete
		if (start_idx > cscColPtr[start_col]) {
			start_flag = true;
			start_rows[dev_id] = start_row;
		} else {
			start_flag = false;
		}
		start_flags[dev_id] = start_flag;		

		end_col = get_row_from_index(m, cscColPtr, end_idx);
		// Mark imcomplete rows
		// True: imcomplete
		if (end_idx < cscColPtr[end_row + 1] - 1)  {
			end_flag = true;
		} else {
			end_flag = false;
		}
		
		// Cacluclate dimensions
		dev_m = m;
		dev_n = end_col - start_col + 1;
    dev_nnz   = (int)(end_idx - start_idx + 1);

		part_time = get_time() - tmp_time;	



		// preparing data on host	
		//cudaMallocHost((void**)&host_csrVal, dev_nnz * sizeof(double));
		//for (int i = start_idx; i <= end_idx; i++) {
		//	host_csrVal[i - start_idx] = csrVal[i];
		//}

		cudaMallocHost((void**)&host_cscColPtr, (dev_n + 1)*sizeof(int));
		//host_csrRowPtr[0] = 0;
		//host_csrRowPtr[dev_m] = dev_nnz;
		//for (int j = 1; j < dev_m; j++) {
		//	host_csrRowPtr[j] = (int)(csrRowPtr[start_row + j] - start_idx);
		//}

		//cudaMallocHost((void**)&host_csrColIndex, dev_nnz * sizeof(int));
		//for (int i = start_idx; i <= end_idx; i++) {
    //  host_csrColIndex[i - start_idx] = csrColIndex[i];
    //}

		//cudaMallocHost((void**)&host_x, dev_n * sizeof(double));
		//for (int i = 0; i < dev_n; i++) {
		//	host_x[i] = x[i];
		//}

		//cudaMallocHost((void**)&host_y, dev_m * sizeof(double));
    //for (int i = 0; i < dev_m; i++) {
    //  host_y[i] = y[start_row + i];
    //}
		
		tmp_time = get_time();

		host_cscVal = cscVal[start_idx];
		host_cscColPtr[0] = 0;
    host_cscColPtr[dev_n] = dev_nnz;
    for (int j = 1; j < dev_m; j++) {
      host_cscColPtr[j] = (int)(cscColPtr[start_col + j] - start_idx);
    }
		host_cscRowIndex = cscRowIndex[start_idx];
		host_x = &x[start_col];
		host_y = y;
	
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
		
		cudaMalloc((void**)&dev_cscVal,      dev_nnz     * sizeof(double));
		cudaMalloc((void**)&dev_cscColPtr,   (dev_n + 1) * sizeof(int)   );
		cudaMalloc((void**)&dev_cscRowIndex, dev_nnz     * sizeof(int)   );
		cudaMalloc((void**)&dev_x,           dev_n       * sizeof(double)); 
		cudaMalloc((void**)&dev_y,           dev_m       * sizeof(double)); 

		double * dense_A;
    double lda = dev_m;
		cudaMalloc((void**)&dense_A, dev_n * dev_m * sizeof(double));

	        //cudaProfilerStart();
		#pragma omp barrier
		tmp_time = get_time();

		cudaMemcpyAsync(dev_cscColPtr, host_cscColPtr, (dev_n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_cscRowIndex, host_cscRowIndex, dev_nnz * sizeof(int), cudaMemcpyHostToDevice, stream); 
		cudaMemcpyAsync(dev_cscVal, host_cscVal, dev_nnz * sizeof(double), cudaMemcpyHostToDevice, stream); 
		cudaMemcpyAsync(dev_y, host_y, dev_m*sizeof(double),  cudaMemcpyHostToDevice, stream); 
		cudaMemcpyAsync(dev_x, host_x, dev_n*sizeof(double), cudaMemcpyHostToDevice, stream); 
		
		
		time_comm = get_time() - curr_time;
		curr_time = get_time();

		err = 0;
		if (kernel == 1) {

			cusparseDcsc2dense(handle,
												 dev_m, dev_n,
											   descr,
												 dev_cscVal,
												 dev_cscRowIndex,
												 dev_cscColPtr,
												 dense_A,
												 lda);

			


			status = cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
															dev_m, dev_n, dev_nnz, 
															alpha, descr, dev_csrVal, 
															dev_csrRowPtr, dev_csrColIndex, 
															dev_x, beta, dev_y);
		  
			/*
		  cusparseSpMatDescr_t A_desc;
			cusparseCreateCsr(&A_desc, dev_m, dev_n, dev_nnz, 
												dev_csrVal, dev_csrRowPtr, dev_csrColIndex,
											  CUSPARSE_INDEX_32I, 
												CUSPARSE_INDEX_32I, 
												CUSPARSE_INDEX_BASE_ZERO, 
												CUDA_R_64F);
			cusparseDnVecDescr_t x_desc;
			cusparseCreateDnVec(&x_desc, dev_n, dev_x, CUDA_R_64F);
	
			cusparseDnVecDescr_t y_desc;
      cusparseCreateDnVec(&y_desc, dev_m, dev_y, CUDA_R_64F);

			size_t buffer_size;
			cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
															&alpha, A_desc, x_desc, &beta, y_desc,
															CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, 
															&buffer_size);
			void * buffer;
			cudaMalloc(&buffer, buffer_size);

			cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   &alpha, A_desc, x_desc, &beta, y_desc,
                   CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, 
                   buffer);

			
			*/
	
 
		} else if (kernel == 2) {
		    status = cusparseDcsrmv_mp(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
																	 dev_m, dev_n, dev_nnz, 
																	 alpha, descr, dev_csrVal, 
																	 dev_csrRowPtr, dev_csrColIndex, 
																	 dev_x,  beta, dev_y); 
		} else if (kernel == 3) {
		    err = csr5_kernel(dev_m, dev_n, dev_nnz, 
													alpha, dev_csrVal, 
													dev_csrRowPtr, dev_csrColIndex, 
													dev_x, beta, dev_y, stream); 
		}

		core_time = get_time() - tmp_time;
		// GPU based merge
		tmp_time = get_time();
		double * dev_y_no_overlap = dev_y;
		int dev_m_no_overlap = dev_m;
		int start_row_no_overlap = start_row;
		if (start_flag) {
			dev_y_no_overlap += 1;
			start_row_no_overlap += 1;
			dev_m_no_overlap -= 1;
			cudaMemcpyAsync(start_element+dev_id, dev_y, sizeof(double), cudaMemcpyDeviceToHost, stream);
		}
		cudaMemcpyAsync(y+start_row_no_overlap, dev_y_no_overlap, dev_m_no_overlap*sizeof(double),  cudaMemcpyDeviceToHost, stream);
		cudaDeviceSynchronize();
		#pragma omp barrier
		if (dev_id == 0) {
			for (int i = 0; i < ngpu; i++) {
				if (start_flags[i]) {
					y[start_rows[i]] += (start_element[i] - (*beta) * org_y[i]); 
				}	
			}
		}

		/* CPU based merge
		cudaMemcpyAsync(host_y, dev_y, dev_m*sizeof(double),  cudaMemcpyDeviceToHost, stream);

		cudaDeviceSynchronize();
		
		
		//printf("thread %d time: %f\n", dev_id,  get_time() - tmp_time);
		#pragma omp critical
		{
		  double tmp = 0.0;
			
		  if (start_flag) {
		    tmp = y[start_row];
		  }

      for (int i = 0; i < dev_m; i++) y[start_row + i] = host_y[i];

		  if (start_flag) {
		    y[start_row] += tmp;
	      y[start_row] -= y2 * (*beta);
		  }
		}
		*/
		merg_time = get_time() - tmp_time;

		//cudaProfilerStop();

		cudaFree(dev_csrVal);
		cudaFree(dev_csrRowPtr);
		cudaFree(dev_csrColIndex);
		cudaFree(dev_x);
		cudaFree(dev_y);
					
    cudaFreeHost(host_csrRowPtr);
		cudaFreeHost(host_csrVal);
		cudaFreeHost(host_csrColIndex);
		cudaFreeHost(host_x);
		cudaFreeHost(host_y);

		cusparseDestroyMatDescr(descr);
	  cusparseDestroy(handle);
		cudaStreamDestroy(stream);

		}

		//cout << "time_parse = " << time_parse << ", time_comm = " << time_comm << ", time_comp = "<< time_comp <<", time_post = " << time_post << endl;
                spmv_ret ret;
                ret.comp_time = core_time;
                ret.comm_time = 0.0;
								ret.part_time = part_time;
								ret.merg_time = merg_time;
		return ret;
	}

