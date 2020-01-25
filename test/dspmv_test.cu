#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "mmio.h"
#include <float.h>
#include <omp.h>
//#include "anonymouslib_cuda.h"
#include <cuda_profiler_api.h>
#include "spmv_kernel.h"
#include <limits>
#include <fstream>
#include <sstream>
using namespace std;



void print_error(cusparseStatus_t status) {
	if (status == CUSPARSE_STATUS_NOT_INITIALIZED)
		cout << "CUSPARSE_STATUS_NOT_INITIALIZED" << endl;
	else if (status == CUSPARSE_STATUS_ALLOC_FAILED)
		cout << "CUSPARSE_STATUS_ALLOC_FAILED" << endl;
	else if (status == CUSPARSE_STATUS_INVALID_VALUE)
		cout << "CUSPARSE_STATUS_INVALID_VALUE" << endl;
	else if (status == CUSPARSE_STATUS_ARCH_MISMATCH)
		cout << "CUSPARSE_STATUS_ARCH_MISMATCH" << endl;
	else if (status == CUSPARSE_STATUS_INTERNAL_ERROR)
		cout << "CUSPARSE_STATUS_INTERNAL_ERROR" << endl;
	else if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
		cout << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << endl;
}


/* CPU version spmv for correctness verification */
void csr_spmv_cpu(int m, int n, int nnz, 
		   double * alpha,
                   double * cooVal, int * csrRowPtr, int * cooColIndex,
                   double * x, 
	           double * beta,
                   double * y) { 
  for (int i = 0; i < m; i++) {
    double sum = 0.0;
    for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++ ){
      sum += cooVal[j] * x[cooColIndex[j]];
    }
    y[i] = (*alpha) * sum + (*beta) * y[i]; 
  }
}


int main(int argc, char *argv[]) {


	if (argc < 5) {
		cout << "Incorrect number of arguments!" << endl;
		cout << "Usage ./spmv [input matrix file] [number of GPU(s)] [number of test(s)] [kernel version (1-3)] [data type ('f' or 'b')]"  << endl;
		return -1;
	}

	char input_type = argv[1][0];

	char * filename = argv[2];

	int NGPU = atoi(argv[3]);
	int repeat_test = atoi(argv[4]);
	//int kernel_version = atoi(argv[5]);
	
	//int divide = atoi(argv[7]);
	//int copy_of_workspace = atoi(argv[8]);

	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int m, n;
    int nnz;   
    int * cooRowIndex;
    int * cooColIndex;
    double * cooVal;
    int * csrRowPtr;

    char * csv_output = argv[5];
    //cout << "csv_output" << csv_output << endl;
    int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount < NGPU) {
		cout << "Error: Not enough number of GPUs. Only " << deviceCount << "available." << endl;
		return -1;
	}
	if (NGPU <= 0) {
		cout << "Error: Number of GPU(s) needs to be greater than 0." << endl;
		return -1;
	}

	/*if (kernel_version != 1 && kernel_version != 2 && kernel_version != 3) {
		cout << "Error: The kernel version can only be: 1, 2, or 3." << endl;
		return -1;
	}
	*/
	// if (divide <= 0) {
	// 	cout << "Error: Number of tasks needs to be greater than 0." << endl;
	// 	return -1;
	// }

	// if (copy_of_workspace <= 0) {
	// 	cout << "Error: Number of Hyper-Q needs to be greater than 0." << endl;
	// 	return -1;
	// }



	cout << "Using total " << NGPU << " GPU(s)." << endl; 
	//cout << "Kernel #" << kernel_version << " is selected." << endl;
	//cout << divide <<  "total task(s) will be generated for version 2 with "<< copy_of_workspace << " Hyper-Q(s) on each GPU." << endl;

    if (input_type == 'f') {

	    cout << "Loading input matrix from " << filename << endl;
	    if ((f = fopen(filename, "r")) == NULL) {
	        printf("File open error. \n");
					exit(1);
	    }
	    if (mm_read_banner(f, &matcode) != 0) {
	        printf("Could not process Matrix Market banner.\n");
	        exit(1);
	    }
	    cout << mm_typecode_to_str(matcode) << endl;
	    int nnz_int;
	    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz_int)) !=0) {
	      printf("header reading error. \n");  
				exit(1);
	    }
	    nnz = nnz_int;
				
		//nnz = 10;
		//	m = 11;
		//	n = 10;

			cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;
	    cudaMallocHost((void **)&cooRowIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooColIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooVal, nnz * sizeof(double));;
	    // Read matrix from file into COO format
	    cout << "Start reading data from file" << endl;
      if (mm_is_pattern(matcode)) { // binary input
				cout << "binary input\n";
				for (int i = 0; i < nnz; i++) {
					fscanf(f, "%d %d\n", &cooRowIndex[i], &cooColIndex[i]);
          cooVal[i] = 1;
					cooRowIndex[i]--;
					cooColIndex[i]--;
				}
			} else if (mm_is_real(matcode)){ // float input
				cout << "float input\n";
				for (int i = 0; i < nnz; i++) {
					fscanf(f, "%d %d %lg\n", &cooRowIndex[i], &cooColIndex[i], &cooVal[i]);
					cooRowIndex[i]--;
					cooColIndex[i]--;
				}
			} else if (mm_is_integer(matcode)){ // integer input
				cout << "integer input\n";
				for (int i = 0; i < nnz; i++) {
					int tmp;
          fscanf(f, "%d %d %d\n", &cooRowIndex[i], &cooColIndex[i], &tmp);
          cooVal[i] = tmp;
					cooRowIndex[i]--;
					cooColIndex[i]--;
				}
			}
			/*
			// testing data
		  for (int i = 0; i < nnz; i++) {
				cooVal[i] = i;
				cooRowIndex[i] = i+1;
				cooColIndex[i] = i;
			}
			*/
			

	
/*
			for (int i = 0; i < nnz; i++) {
	    	//cout << i << endl;
				if (mm_is_pattern(matcode)) { // binary input
	    		//cout << i << endl;
					fscanf(f, "%d %d\n", &cooRowIndex[i], &cooColIndex[i]);
	    		cooVal[i] = 1;
	    	} else if (mm_is_real(matcode)){ // float input
	        //cout << i << endl;	
					fscanf(f, "%d %d %lg\n", &cooRowIndex[i], &cooColIndex[i], &cooVal[i]);
	      } else if (mm_is_integer(matcode)){ // integer input
					//cout << i << endl;
					int tmp;
					fscanf(f, "%d %d %d\n", &cooRowIndex[i], &cooColIndex[i], &tmp);
    			cooVal[i] = tmp;
				}
	      cooRowIndex[i]--;  
	      cooColIndex[i]--;
			}
*/
	  	cout << "Done loading data from file" << endl;
	} else if(input_type == 'g') { // generate data
                //int n = 10000;
		n = atoi(filename);

		m = n;
		int nb = m / 8;
		double r;
		double r1 = 0.9;
		double r2 = 0.01;

		int p = 0;

		for (int i = 0; i < m; i += nb) {
			if (i == 0) {
				r = r1;
			} else {
				r = r2;
			}
			for (int ii = i; ii < i + nb; ii++) {
				for (int j = 0; j < n * r; j++) {
					p++;
				}
			}
		}


		nnz = p;

		cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;
		

		cudaMallocHost((void **)&cooRowIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooColIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooVal, nnz * sizeof(double));

	    p = 0;
		

		cout << "Start generating data " << std::flush;
		for (int i = 0; i < m; i += nb) {
			cout << "." << std::flush;
			if (i == 0) {
				r = r1;
			} else {
				r = r2;
			}
			//cout << "Matrix:" << endl;
			for (int ii = i; ii < i + nb; ii++) {
				for (int j = 0; j < n * r; j++) {
					//if (p > nnz) { cout << "error" << endl; break;}
					//else {

					cooRowIndex[p] = ii;
					cooColIndex[p] = j;
					cooVal[p] = 1.0;//(double) rand() / (RAND_MAX);
					p++;

					//cout << 1 << " ";
					//}
				}
				//cout << endl;
			}
		}
		cout << endl;

		//cout << "m: " << m << " n: " << n << " nnz: " << p << endl;


		cout << "Done generating data." << endl;


	}



    




	// Convert COO to CSR
    //csrRowPtr = (int *) malloc((m+1) * sizeof(int));
    cudaMallocHost((void **)&csrRowPtr, (m+1) * sizeof(int));

    //cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;
    long long matrix_data_space = nnz * sizeof(double) + nnz * sizeof(int) + (m+1) * sizeof(int);
    //cout << matrix_data_space << endl;

    double matrix_size_in_gb = (double)matrix_data_space / 1e9;
    cout << "Matrix space size: " << matrix_size_in_gb << " GB." << endl;

    int * counter = new int[m];
    for (int i = 0; i < m; i++) {
    	counter[i] = 0;
    }
    for (int i = 0; i < nnz; i++) {
	counter[cooRowIndex[i]]++;
    }
    //cout << "nnz: " << nnz << endl;
    //cout << "counter: ";
    int t = 0;
    for (int i = 0; i < m; i++) {
	//cout << counter[i] << ", ";
	t += counter[i];
    }
    //cout << t << endl;
    //cout << endl;

    //cout << "csrRowPtr: ";
    csrRowPtr[0] = 0;
    for (int i = 1; i <= m; i++) {
	csrRowPtr[i] = csrRowPtr[i - 1] + counter[i - 1];
	//cout << "csrRowPtr[" << i <<"] = "<<csrRowPtr[i] << endl;
    }

	double * x;
	double * y_baseline;

	double * y_v1k1;
        double * y_v1k2;
        double * y_v1k3;
        double * y_v2k1;
        double * y_v2k2;
        double * y_v2k3;

	double * y_v1k1s;
        double * y_v1k2s;
        double * y_v1k3s;
        double * y_v2k1s;
        double * y_v2k2s;
        double * y_v2k3s;

	double * y_verify;
	//cout << "Allocating space for x and y" << endl;
	
        //x = (double *)malloc(n * sizeof(double)); 
	cudaMallocHost((void**)&x, n * sizeof(double));
	/*
 	y_baseline = (double *)malloc(m * sizeof(double)); 
	y_v1k1 = (double *)malloc(m * sizeof(double)); 
	y_v1k2 = (double *)malloc(m * sizeof(double)); 
	y_v1k3 = (double *)malloc(m * sizeof(double));
        y_v2k1 = (double *)malloc(m * sizeof(double));
        y_v2k2 = (double *)malloc(m * sizeof(double));
        y_v2k3 = (double *)malloc(m * sizeof(double));
        y_verify = (double *)malloc(m * sizeof(double));
	*/
	cudaMallocHost((void**)&y_baseline, m * sizeof(double));

	cudaMallocHost((void**)&y_v1k1, m * sizeof(double));
	cudaMallocHost((void**)&y_v1k2, m * sizeof(double));
	cudaMallocHost((void**)&y_v1k3, m * sizeof(double));
	cudaMallocHost((void**)&y_v2k1, m * sizeof(double));
        cudaMallocHost((void**)&y_v2k2, m * sizeof(double));
        cudaMallocHost((void**)&y_v2k3, m * sizeof(double));	

	cudaMallocHost((void**)&y_v1k1s, m * sizeof(double));
        cudaMallocHost((void**)&y_v1k2s, m * sizeof(double));
        cudaMallocHost((void**)&y_v1k3s, m * sizeof(double));
        cudaMallocHost((void**)&y_v2k1s, m * sizeof(double));
        cudaMallocHost((void**)&y_v2k2s, m * sizeof(double));
        cudaMallocHost((void**)&y_v2k3s, m * sizeof(double));

	cudaMallocHost((void**)&y_verify, m * sizeof(double));
	


	/*
	cudaMallocHost((void **)&x, n * sizeof(double));
	cudaMallocHost((void **)&y1, m * sizeof(double));
	cudaMallocHost((void **)&y2, m * sizeof(double));
	cudaMallocHost((void **)&y3, m * sizeof(double));
	*/
        //cout << "Initializing x" << endl;
	for (int i = 0; i < n; i++)
	{
		x[i] = 1.0; //((double) rand() / (RAND_MAX)); 
	}

	/*
	int zero_count = 0;
	for (int i = 0; i < n; i++){
    if ((double) rand() / (RAND_MAX) < 0.5) {
      x[i] = 0;
	    zero_count++;
    }
  }
  cout << "x zero count: " << zero_count << "/" << n << endl;
	*/
	//cout << "Initializing y" << endl;
	for (int i = 0; i < m; i++)
	{
		y_baseline[i] = 0.0;
                y_v1k1[i] = 0.0;
		y_v1k2[i] = 0.0;
		y_v1k3[i] = 0.0;
                y_v2k1[i] = 0.0;
                y_v2k2[i] = 0.0;
                y_v2k3[i] = 0.0;
	
		y_v1k1s[i] = 0.0;
                y_v1k2s[i] = 0.0;
                y_v1k3s[i] = 0.0;
                y_v2k1s[i] = 0.0;
                y_v2k2s[i] = 0.0;
                y_v2k3s[i] = 0.0;

		y_verify[i] = 0.0;
	}

	double ALPHA = 1.0;//(double) rand() / (RAND_MAX);
	double BETA = 0.0; //(double) rand() / (RAND_MAX);

	double time_baseline = 0.0;
	double time_baseline_part = 0.0;

	double time_v1k1 = 0.0;
  double time_v1k2 = 0.0;
  double time_v1k3 = 0.0;
	double time_v2k1 = 0.0;
  double time_v2k2 = 0.0;
  double time_v2k3 = 0.0;
  
	double time_v1k1s = 0.0;
  double time_v1k2s = 0.0;
  double time_v1k3s = 0.0;
  double time_v2k1s = 0.0;
  double time_v2k2s = 0.0;
  double time_v2k3s = 0.0;


	double time_v1k1_numa_part = 0.0;
	double time_v1k1_part = 0.0;
	double time_v1k1_merg = 0.0;


	double avg_time_baseline = 0.0;
	double avg_time_baseline_part=0.0;

	double avg_time_v1k1 = 0.0;
  double avg_time_v1k2 = 0.0;
  double avg_time_v1k3 = 0.0;
	double avg_time_v2k1 = 0.0;
  double avg_time_v2k2 = 0.0;
  double avg_time_v2k3 = 0.0;

	double avg_time_v1k1_numa_part = 0.0;
	double avg_time_v1k1_part = 0.0;
  double avg_time_v1k1_merg = 0.0;

	double avg_time_v1k1s = 0.0;
  double avg_time_v1k2s = 0.0;
  double avg_time_v1k3s = 0.0;
  double avg_time_v2k1s = 0.0;
  double avg_time_v2k2s = 0.0;
  double avg_time_v2k3s = 0.0;

	double curr_time = 0.0;

	int warm_up_iter = 0;

	cout << "Compute CPU version" << endl;
	csr_spmv_cpu(m, n, nnz,
                     &ALPHA,
                     cooVal, csrRowPtr, cooColIndex,
                     x,
                     &BETA,
                     y_verify);


	ofstream myfile;
        ostringstream o;
        if(input_type == 'g') {
            o << csv_output << "_" << n << ".csv";
        }
        if(input_type == 'f') {
            o << csv_output << ".csv";
        }
        cout <<"Output to file: " << o.str().c_str() << endl;
        myfile.open (o.str().c_str());

        //for (int ngpu = 1; ngpu <= NGPU; ngpu++){
	int ngpu = NGPU;

	cout << "Warming up GPU(s)..." << endl;

	for (int i = 0; i < warm_up_iter; i++) {
		spMV_mgpu_baseline(m, n, nnz, &ALPHA,
					 cooVal, csrRowPtr, cooColIndex, 
					 x, &BETA,
					 y_v1k1,
					 ngpu);

	}




	  
  double profile_time = 0.0;
        
  double min_profile_time_k1 = numeric_limits<double>::max();
  double best_dev_count_k1 = ngpu;
  int best_copy_k1 = 1;
	
	double min_profile_time_k2 = numeric_limits<double>::max();
  double best_dev_count_k2 = ngpu;
  int best_copy_k2 = 1;

	double min_profile_time_k3 = numeric_limits<double>::max();
  double best_dev_count_k3 = ngpu;
  int best_copy_k3 = 1;

	
/*
	for (int d = 1; d <= ngpu; d++) {
          for (int c = 1; c <= 8; c++) {
	    curr_time = get_time();
	    spmv_ret ret =spMV_mgpu_v2(m, n, nnz, &ALPHA,
			 cooVal, csrRowPtr, cooColIndex, 
			 x, &BETA,
			 y_v2k1,
			 d,
		         1, //kernel 1
			 ceil(nnz / (d * c)),
			 c);
	    profile_time = ret.comp_time + ret.comm_time; //get_time() - curr_time;	
	    if (profile_time < min_profile_time_k1) {
	      min_profile_time_k1 = profile_time;
              best_dev_count_k1 = d;
              best_copy_k1 = c;
	    }
	  }
        }
*/
/*
       for (int d = 1; d <= ngpu; d++) {
          for (int c = 1; c <= 8; c++) {
            curr_time = get_time();
            spmv_ret ret = spMV_mgpu_v2(m, n, nnz, &ALPHA,
                         cooVal, csrRowPtr, cooColIndex,  
                         x, &BETA,
                         y_v2k2,
                         d,
                         2, //kernel 2
                         ceil(nnz / (d * c)),
                         c);
            profile_time = ret.comp_time + ret.comm_time;//get_time() - curr_time;  
            if (profile_time < min_profile_time_k2) {
              min_profile_time_k2 = profile_time;
              best_dev_count_k2 = d;
              best_copy_k2 = c;
            }
          }
        }

       for (int d = 1; d <= ngpu; d++) {
          for (int c = 1; c <= 8; c++) {
            curr_time = get_time();
            spmv_ret ret = spMV_mgpu_v2(m, n, nnz, &ALPHA,
                         cooVal, csrRowPtr, cooColIndex,  
                         x, &BETA,
                         y_v2k3,
                         d,
                         3, //kernel 3
                         ceil(nnz / (d * c)),
                         c);
            profile_time = ret.comp_time + ret.comm_time;//get_time() - curr_time;  
            if (profile_time < min_profile_time_k3) {
              min_profile_time_k3 = profile_time;
              best_dev_count_k3 = d;
              best_copy_k3 = c;
            }
          }
        }

*/
	double min_profile_time_k1s = numeric_limits<double>::max();
  double best_dev_count_k1s = ngpu;
  int best_copy_k1s = 1;

  double min_profile_time_k2s = numeric_limits<double>::max();
  double best_dev_count_k2s = ngpu;
  int best_copy_k2s = 1;

  double min_profile_time_k3s = numeric_limits<double>::max();
  double best_dev_count_k3s = ngpu;
  int best_copy_k3s = 1;

/*
	for (int d = 1; d <= ngpu; d++) {
          for (int c = 1; c <= 8; c++) {
            curr_time = get_time();
            spmv_ret ret =spMspV_mgpu_v2(m, n, nnz, &ALPHA,
                         cooVal, csrRowPtr, cooColIndex, 
                         x, &BETA,
                         y_v2k1s,
                         d,
                         1, //kernel 1
                         ceil(nnz / (d * c)),
                         c);
            profile_time = ret.comp_time + ret.comm_time; //get_time() - curr_time;     
            if (profile_time < min_profile_time_k1s) {
              min_profile_time_k1s = profile_time;
              best_dev_count_k1s = d;
              best_copy_k1s = c;
            }
          }
        }
*/
/*
       for (int d = 1; d <= ngpu; d++) {
          for (int c = 1; c <= 8; c++) {
            curr_time = get_time();
            spmv_ret ret = spMspV_mgpu_v2(m, n, nnz, &ALPHA,
                         cooVal, csrRowPtr, cooColIndex,  
                         x, &BETA,
                         y_v2k2s,
                         d,
                         2, //kernel 2
                         ceil(nnz / (d * c)),
                         c);
            profile_time = ret.comp_time + ret.comm_time;//get_time() - curr_time;  
            if (profile_time < min_profile_time_k2s) {
              min_profile_time_k2s = profile_time;
              best_dev_count_k2s = d;
              best_copy_k2s = c;
            }
          }
        }

       for (int d = 1; d <= ngpu; d++) {
          for (int c = 1; c <= 8; c++) {
            curr_time = get_time();
            spmv_ret ret = spMspV_mgpu_v2(m, n, nnz, &ALPHA,
                         cooVal, csrRowPtr, cooColIndex,  
                         x, &BETA,
                         y_v2k3s,
                         d,
                         3, //kernel 3
                         ceil(nnz / (d * c)),
                         c);
            profile_time = ret.comp_time + ret.comm_time;//get_time() - curr_time;  
            if (profile_time < min_profile_time_k3s) {
              min_profile_time_k3s = profile_time;
              best_dev_count_k3s = d;
              best_copy_k3s = c;
            }
          }
        }

*/
	spmv_ret ret_baseline;
	spmv_ret ret_v1k1;
	spmv_ret ret_v1k2;
  spmv_ret ret_v1k3;
  spmv_ret ret_v2k1;
  spmv_ret ret_v2k2;
  spmv_ret ret_v2k3;

	spmv_ret ret_v1k1s;
  spmv_ret ret_v1k2s;
  spmv_ret ret_v1k3s;
  spmv_ret ret_v2k1s;
  spmv_ret ret_v2k2s;
  spmv_ret ret_v2k3s;

  int pass_baseline = 0;
  int pass_v1k1 = 0;
  int pass_v1k2 = 0;
  int pass_v1k3 = 0;
  int pass_v2k1 = 0;
  int pass_v2k2 = 0;
  int pass_v2k3 = 0;

	int pass_v1k1s = 0;
  int pass_v1k2s = 0;
  int pass_v1k3s = 0;
  int pass_v2k1s = 0;
  int pass_v2k2s = 0;
  int pass_v2k3s = 0;
	
	cout << "Starting tests..." << endl;

	//cudaProfilerStart();

	cout << "entering loop\n";
	for (int i = 0; i < repeat_test; i++) {
	  
		cout << "init y\n";
		for (int i = 0; i < m; i++)
	  {
	    y_baseline[i] = 0.0;
	    y_v1k1[i] = 0.0;
	    y_v1k2[i] = 0.0;
      y_v1k3[i] = 0.0;
      y_v2k1[i] = 0.0;
      y_v2k2[i] = 0.0;
      y_v2k3[i] = 0.0;

	    y_v1k1s[i] = 0.0;
      y_v1k2s[i] = 0.0;
      y_v1k3s[i] = 0.0;
      y_v2k1s[i] = 0.0;
      y_v2k2s[i] = 0.0;
      y_v2k3s[i] = 0.0;
	  }


		cout << "start baseline\n";
	  curr_time = get_time();
	  ret_baseline = spMV_mgpu_baseline(m, n, nnz, &ALPHA,
				    cooVal, csrRowPtr, cooColIndex, 
				    x, &BETA,
				    y_baseline,
				    ngpu);
	  time_baseline = ret_baseline.comp_time + ret_baseline.comm_time; //get_time() - curr_time;	
		time_baseline_part = ret_baseline.part_time;

		cout << "end baseline\n";
    int numa_mapping[6] = {0,0,0,1,1,1};    

	
//    cudaProfilerStart();
	  cout << "start v1\n";
		ret_v1k1 = spMV_mgpu_v1_numa(m, n, nnz, &ALPHA,
				  cooVal, csrRowPtr, cooColIndex, 
				  x, &BETA,
				  y_v1k1,
				  ngpu,
				  1,
					numa_mapping); //kernel 1
		cout << "end v1\n";
	  time_v1k1 = ret_v1k1.comp_time + ret_v1k1.comm_time;  //get_time() - curr_time;
		time_v1k1_part = ret_v1k1.part_time;
		time_v1k1_merg = ret_v1k1.merg_time;
		time_v1k1_numa_part = ret_v1k1.numa_part_time;	
//	  cudaProfilerStop();

/*
	  curr_time = get_time();
          ret_v1k2 = spMV_mgpu_v1(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v1k2,
                                  ngpu, 
                                  2); //kernel 2
          time_v1k2 = ret_v1k2.comp_time + ret_v1k2.comm_time; //get_time() - curr_time;

	  curr_time = get_time();
          ret_v1k3 = spMV_mgpu_v1(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v1k3,
                                  ngpu, 
                                  3); //kernel 3
          time_v1k3 = ret_v1k3.comp_time + ret_v1k3.comm_time; //get_time() - curr_time;

		//cudaProfilerStart();


		
	  curr_time = get_time();
	  ret_v2k1 = spMV_mgpu_v2(m, n, nnz, &ALPHA,
				  cooVal, csrRowPtr, cooColIndex, 
			          x, &BETA,
				  y_v2k1,
				  //ngpu,
				  best_dev_count_k1,
				  1, //kernel 1
				  //nnz,
 			          //1
		                  ceil(nnz / (best_dev_count_k1 * best_copy_k1)),
			          best_copy_k1);
	  time_v2k1 = ret_v2k1.comp_time + ret_v2k1.comm_time; //get_time() - curr_time;	


	  curr_time = get_time();
          ret_v2k2 = spMV_mgpu_v2(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v2k2,   
                                  //ngpu,
                                  best_dev_count_k2,
                                  2, //kernel 2
                                  //nnz,
                                  //1
                                  ceil(nnz / (best_dev_count_k2 * best_copy_k2)),
                                  best_copy_k2);
          time_v2k2 = ret_v2k2.comp_time + ret_v2k2.comm_time; //get_time() - curr_time;
	
	  curr_time = get_time();
          ret_v2k3 = spMV_mgpu_v2(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v2k3,   
                                  //ngpu,
                                  best_dev_count_k3,
                                  3, //kernel 3
                                  //nnz,
                                  //1
                                  ceil(nnz / (best_dev_count_k3 * best_copy_k3)),
                                  best_copy_k3);
          time_v2k3 = ret_v2k3.comp_time + ret_v2k3.comm_time; //get_time() - curr_time;

	
	  //cudaProfilerStart();
          ret_v1k1s = spMspV_mgpu_v1_numa(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v1k1s,
                                  ngpu,
                                  1); //kernel 1
          time_v1k1s = ret_v1k1s.comp_time + ret_v1k1s.comm_time;  //get_time() - curr_time;       
      //    cudaProfilerStop();

          curr_time = get_time();
          ret_v1k2s = spMspV_mgpu_v1(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v1k2s,
                                  ngpu,
                                  2); //kernel 2
          time_v1k2s = ret_v1k2s.comp_time + ret_v1k2s.comm_time; //get_time() - curr_time;

          curr_time = get_time();
          ret_v1k3s = spMspV_mgpu_v1(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v1k3s,
                                  ngpu,
                                  3); //kernel 3
          time_v1k3s = ret_v1k3s.comp_time + ret_v1k3s.comm_time; //get_time() - curr_time;
	

	  curr_time = get_time();
          ret_v2k1s = spMspV_mgpu_v2(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v2k1s,
                                  //ngpu,
                                  best_dev_count_k1s,
                                  1, //kernel 1
                                  //nnz,
                                  //1
                                  ceil(nnz / (best_dev_count_k1s * best_copy_k1s)),
                                  best_copy_k1s);
          time_v2k1s = ret_v2k1s.comp_time + ret_v2k1s.comm_time; //get_time() - curr_time;        


          curr_time = get_time();
          ret_v2k2s = spMspV_mgpu_v2(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v2k2s,   
                                  //ngpu,
                                  best_dev_count_k2s,
                                  2, //kernel 2
                                  //nnz,
                                  //1
                                  ceil(nnz / (best_dev_count_k2s * best_copy_k2s)),
                                  best_copy_k2s);
          time_v2k2s = ret_v2k2s.comp_time + ret_v2k2s.comm_time; //get_time() - curr_time;
        
          curr_time = get_time();
          ret_v2k3s = spMspV_mgpu_v2(m, n, nnz, &ALPHA,
                                  cooVal, csrRowPtr, cooColIndex,
                                  x, &BETA,
                                  y_v2k3s,   
                                  //ngpu,
                                  best_dev_count_k3s,
                                  3, //kernel 3
                                  //nnz,
                                  //1
                                  ceil(nnz / (best_dev_count_k3s * best_copy_k3s)),
                                  best_copy_k3s);
          time_v2k3s = ret_v2k3s.comp_time + ret_v2k3s.comm_time; //get_time() - curr_time;
*/


	  avg_time_baseline += time_baseline;
		avg_time_baseline_part += time_baseline_part;

	  avg_time_v1k1  += time_v1k1;
    avg_time_v1k2  += time_v1k2;
    avg_time_v1k3  += time_v1k3;          
    avg_time_v2k1  += time_v2k1;
    avg_time_v2k2  += time_v2k2;
    avg_time_v2k3  += time_v2k3;

		avg_time_v1k1_part += time_v1k1_part;
		avg_time_v1k1_merg += time_v1k1_merg;
		avg_time_v1k1_numa_part += time_v1k1_numa_part;

	  avg_time_v1k1s  += time_v1k1s;
    avg_time_v1k2s  += time_v1k2s;
    avg_time_v1k3s  += time_v1k3s;
    avg_time_v2k1s  += time_v2k1s;
    avg_time_v2k2s  += time_v2k2s;
    avg_time_v2k3s  += time_v2k3s;

	  bool correct_baseline = true;
	  bool correct_v1k1 = true;
	  bool correct_v1k2 = true;
    bool correct_v1k3 = true;
    bool correct_v2k1 = true;
    bool correct_v2k2 = true;
    bool correct_v2k3 = true;

	  bool correct_v1k1s = true;
    bool correct_v1k2s = true;
    bool correct_v1k3s = true;
    bool correct_v2k1s = true;
    bool correct_v2k2s = true;
    bool correct_v2k3s = true;

    double E = 1e-3;
	  for(int i = 0; i < m; i++) {
	    if (abs(y_verify[i] - y_baseline[i]) > E) {
              correct_baseline = false;
            }
            if (abs(y_verify[i] - y_v1k1[i]) > E) {
							cout << "error at: " << i <<" : " << y_verify[i] << " - "<< y_v1k1[i] << endl;
              correct_v1k1 = false;
            }
            if (abs(y_verify[i] - y_v1k2[i]) > E) {
              correct_v1k2 = false;
            }
            if (abs(y_verify[i] - y_v1k3[i]) > E) {
              correct_v1k3 = false;
            }
            if (abs(y_verify[i] - y_v2k1[i]) > E) {
              correct_v2k1 = false;
            }
            if (abs(y_verify[i] - y_v2k2[i]) > E) {
              correct_v2k2 = false;
            }
            if (abs(y_verify[i] - y_v2k3[i]) > E) {
              correct_v2k3 = false;
            }
	
	    if (abs(y_verify[i] - y_v1k1s[i]) > E) {
              correct_v1k1s = false;
            }
            if (abs(y_verify[i] - y_v1k2s[i]) > E) {
              correct_v1k2s = false;
            }
            if (abs(y_verify[i] - y_v1k3s[i]) > E) {
              correct_v1k3s = false;
            }
            if (abs(y_verify[i] - y_v2k1s[i]) > E) {
              correct_v2k1s = false;
            }
            if (abs(y_verify[i] - y_v2k2s[i]) > E) {
              correct_v2k2s = false;
            }
            if (abs(y_verify[i] - y_v2k3s[i]) > E) {
              correct_v2k3s = false;
            }
          }
 
          if (correct_baseline) pass_baseline++;
          if (correct_v1k1) pass_v1k1++;
          if (correct_v1k2) pass_v1k2++;
          if (correct_v1k3) pass_v1k3++;
          if (correct_v2k1) pass_v2k1++;
          if (correct_v2k2) pass_v2k2++;
          if (correct_v2k3) pass_v2k3++;
		
	  if (correct_v1k1s) pass_v1k1s++;
          if (correct_v1k2s) pass_v1k2s++;
          if (correct_v1k3s) pass_v1k3s++;
          if (correct_v2k1s) pass_v2k1s++;
          if (correct_v2k2s) pass_v2k2s++;
          if (correct_v2k3s) pass_v2k3s++;	

	}

	//cudaProfilerStop();
	avg_time_baseline/=repeat_test;
	avg_time_baseline_part/=repeat_test;

	avg_time_v1k1/=repeat_test;
	avg_time_v1k2/=repeat_test;
  avg_time_v1k3/=repeat_test;
  avg_time_v2k1/=repeat_test;
  avg_time_v2k2/=repeat_test;
  avg_time_v2k3/=repeat_test;

	avg_time_v1k1_part /=repeat_test;
  avg_time_v1k1_merg /=repeat_test;
  avg_time_v1k1_numa_part /=repeat_test;
	
	avg_time_v1k1s/=repeat_test;
  avg_time_v1k2s/=repeat_test;
  avg_time_v1k3s/=repeat_test;
  avg_time_v2k1s/=repeat_test;
  avg_time_v2k2s/=repeat_test;
  avg_time_v2k3s/=repeat_test;


	cout << "Baseline: " <<avg_time_baseline_part <<", "<<avg_time_baseline << "s (" << pass_baseline << "/" << repeat_test << ")" << endl;
  cout << "V1K1: " << avg_time_v1k1_numa_part<< ", "<<avg_time_v1k1_part << ", " << avg_time_v1k1 << ", " << avg_time_v1k1_merg << " (" << pass_v1k1 << "/" << repeat_test << ")" << endl;
	cout << "V1K2: " << avg_time_v1k2 << "s (" << pass_v1k2 << "/" << repeat_test << ")" << endl;
	cout << "V1K3: " << avg_time_v1k3 << "s (" << pass_v1k3 << "/" << repeat_test << ")" << endl;
	cout << "V2K1: " << avg_time_v2k1 << "s (" << pass_v2k1 << "/" << repeat_test << ")" << endl;
  cout << "V2K2: " << avg_time_v2k2 << "s (" << pass_v2k2 << "/" << repeat_test << ")" << endl;
  cout << "V2K3: " << avg_time_v2k3 << "s (" << pass_v2k3 << "/" << repeat_test << ")" << endl;

	cout << "V1K1s: " << avg_time_v1k1s << "s (" << pass_v1k1s << "/" << repeat_test << ")" << endl;
  cout << "V1K2s: " << avg_time_v1k2s << "s (" << pass_v1k2s << "/" << repeat_test << ")" << endl;
  cout << "V1K3s: " << avg_time_v1k3s << "s (" << pass_v1k3s << "/" << repeat_test << ")" << endl;
  cout << "V2K1s: " << avg_time_v2k1s << "s (" << pass_v2k1s << "/" << repeat_test << ")" << endl;
  cout << "V2K2s: " << avg_time_v2k2s << "s (" << pass_v2k2s << "/" << repeat_test << ")" << endl;
  cout << "V2K3s: " << avg_time_v2k3s << "s (" << pass_v2k3s << "/" << repeat_test << ")" << endl;

	myfile << avg_time_baseline << "," << avg_time_v1k1 << "," << avg_time_v1k2 << "," << avg_time_v1k3 << "," << avg_time_v2k1 << "," << avg_time_v2k2 << "," << avg_time_v2k3 << "," << avg_time_v1k1s << "," << avg_time_v1k2s << "," << avg_time_v1k3s << "," << avg_time_v2k1s << "," << avg_time_v2k2s << "," << avg_time_v2k3s; 	

//	}

	cudaFreeHost(cooRowIndex);
	cudaFreeHost(cooColIndex);
	cudaFreeHost(cooVal);
	cudaFreeHost(csrRowPtr);
	

/*	
	ofstream myfile;
	ostringstream o;
	if(input_type == 'g') {
	    o << csv_output << "_" << n << ".csv";
	}
 	if(input_type == 'f') {
	    o << csv_output << ".csv";
	}
        cout <<"Output to file: " << o.str().c_str() << endl;
	myfile.open (o.str().c_str());


	myfile << avg_time_baseline << "," << avg_time_v1k1 << "," << avg_time_v1k2 << "," << avg_time_v1k3 << "," << avg_time_v2k1 << "," << avg_time_v2k2 << "," << avg_time_v2k3;
*/

	myfile.close();
}
