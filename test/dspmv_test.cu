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

  int ngpu = atoi(argv[3]);
  int repeat_test = atoi(argv[4]);
  //int kernel_version = atoi(argv[5]);
  
  //int divide = atoi(argv[7]);
  //int copy_of_workspace = atoi(argv[8]);

  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int m, n;
  int nnz;   
  int * cooRowIdx;
  int * cooColIdx;
  double * cooVal;
  

  char * csv_output = argv[5];
  //cout << "csv_output" << csv_output << endl;
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount < ngpu) {
    cout << "Error: Not enough number of GPUs. Only " << deviceCount << "available." << endl;
    return -1;
  }
  if (ngpu <= 0) {
    cout << "Error: Number of GPU(s) needs to be greater than 0." << endl;
    return -1;
  }


  cout << "Using total " << ngpu << " GPU(s)." << endl; 


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
        
    nnz = 100;
     m = 10;
     n = 10;

    cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;
    cudaMallocHost((void **)&cooRowIdx, nnz * sizeof(int));
    cudaMallocHost((void **)&cooColIdx, nnz * sizeof(int));
    cudaMallocHost((void **)&cooVal, nnz * sizeof(double));;
    //Read matrix from file into COO format
    // cout << "Start reading data from file" << endl;
    // if (mm_is_pattern(matcode)) { // binary input
    //   cout << "binary input\n";
    //   for (int i = 0; i < nnz; i++) {
    //     fscanf(f, "%d %d\n", &cooRowIndex[i], &cooColIndex[i]);
    //     cooVal[i] = 1;
    //     cooRowIndex[i]--;
    //     cooColIndex[i]--;
    //   }
    // } else if (mm_is_real(matcode)){ // float input
    //   cout << "float input\n";
    //   for (int i = 0; i < nnz; i++) {
    //     fscanf(f, "%d %d %lg\n", &cooRowIndex[i], &cooColIndex[i], &cooVal[i]);
    //     cooRowIndex[i]--;
    //     cooColIndex[i]--;
    //   }
    // } else if (mm_is_integer(matcode)){ // integer input
    //   cout << "integer input\n";
    //   for (int i = 0; i < nnz; i++) {
    //     int tmp;
    //     fscanf(f, "%d %d %d\n", &cooRowIndex[i], &cooColIndex[i], &tmp);
    //     cooVal[i] = tmp;
    //     cooRowIndex[i]--;
    //     cooColIndex[i]--;
    //   }
    // }
    
    // testing data
    int p = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        cooVal[p] = 1;
        cooRowIdx[p] = i;
        cooColIdx[p] = j;
        p++;
      }
    }
    // for (int i = 0; i < nnz; i++) {
    //   cooVal[i] = 1;
    //   cooRowIndex[i] = ;
    //   cooColIndex[i] = i;
    // }
    
    cout << "Done loading data from file" << endl;
  } else if(input_type == 'g') { // generate data
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
    
    cudaMallocHost((void **)&cooRowIdx, nnz * sizeof(int));
    cudaMallocHost((void **)&cooColIdx, nnz * sizeof(int));
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
          cooRowIdx[p] = ii;
          cooColIdx[p] = j;
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



  sortCOORow(m, n, nnz, cooVal, cooRowIdx, cooColIdx);

  // Convert COO to CSR
  double * csrVal;
  int * csrRowPtr;
  int * csrColIdx;
  cudaMallocHost((void **)&csrVal, nnz * sizeof(double));
  cudaMallocHost((void **)&csrRowPtr, (m+1) * sizeof(int));
  cudaMallocHost((void **)&csrColIdx, nnz * sizeof(int));

  long long matrix_data_space = nnz * sizeof(double) + nnz * sizeof(int) + (m+1) * sizeof(int);

  double matrix_size_in_gb = (double)matrix_data_space / 1e9;
  cout << "Matrix space size: " << matrix_size_in_gb << " GB." << endl;

  // int * counter = new int[m];
  // for (int i = 0; i < m; i++) {
  //   counter[i] = 0;
  // }
  // for (int i = 0; i < nnz; i++) {
  //   counter[cooRowIndex[i]]++;
  // }
  // int t = 0;
  // for (int i = 0; i < m; i++) {
  //   t += counter[i];
  // }
  // csrRowPtr[0] = 0;
  // for (int i = 1; i <= m; i++) {
  //   csrRowPtr[i] = csrRowPtr[i - 1] + counter[i - 1];
  // }

  // csrVal = cooVal;
  // csrColIdx = cooColIndex;

  // CSR to CSC
  double * cscVal;
  int * cscColPtr;
  int * cscRowIdx;
  cudaMallocHost((void **)&cscVal, nnz * sizeof(double));
  cudaMallocHost((void **)&cscColPtr, (n+1) * sizeof(int));
  cudaMallocHost((void **)&cscRowIdx, nnz * sizeof(int));

  // csr2csc(m, n, nnz,
  //         csrVal, csrRowPtr, csrColIdx,
  //         cscVal, cscColPtr, cscRowIdx);

  csr2csrNcsc(m, n, nnz,
             cooVal, cooRowIdx, cooColIdx,
             csrVal, csrRowPtr, csrColIdx,
             cscVal, cscColPtr, cscRowIdx);
  printf("out of conversion function\n");

  print_vec(csrVal, nnz, "csrVal:");
  print_vec(csrRowPtr, m+1, "csrRowPtr:");
  print_vec(csrColIdx, nnz, "csrColIdx:");

  print_vec(cscVal, nnz, "cscVal:");
  print_vec(cscColPtr, n+1, "cscColPtr:");
  print_vec(cscRowIdx, nnz, "cscRowIdx:");

  double * x;

  double * y_baseline_csr;
  double * y_static_csr;
  double * y_dynamic_csr;

  double * y_baseline_csc;
  double * y_static_csc;
  double * y_dynamic_csc;

  double * y_baseline_coo;
  double * y_static_coo;
  double * y_dynamic_coo;

  double * y_verify;

  printf("Allocate y\n");

  cudaMallocHost((void**)&x, n * sizeof(double));

  cudaMallocHost((void**)&y_baseline_csr, m * sizeof(double));
  cudaMallocHost((void**)&y_static_csr, m * sizeof(double));
  cudaMallocHost((void**)&y_dynamic_csr, m * sizeof(double));

  cudaMallocHost((void**)&y_baseline_csc, m * sizeof(double));
  cudaMallocHost((void**)&y_static_csc, m * sizeof(double));
  cudaMallocHost((void**)&y_dynamic_csc, m * sizeof(double));

  cudaMallocHost((void**)&y_baseline_coo, m * sizeof(double));
  cudaMallocHost((void**)&y_static_coo, m * sizeof(double));
  cudaMallocHost((void**)&y_dynamic_coo, m * sizeof(double));

  cudaMallocHost((void**)&y_verify, m * sizeof(double));

  cout << "Initializing x" << endl;
  for (int i = 0; i < n; i++)
  {
    x[i] = 1.0; //((double) rand() / (RAND_MAX)); 
  }
//   /*
//   int zero_count = 0;
//   for (int i = 0; i < n; i++){
//     if ((double) rand() / (RAND_MAX) < 0.5) {
//       x[i] = 0;
//       zero_count++;
//     }
//   }
//   cout << "x zero count: " << zero_count << "/" << n << endl;
//   */

  double ALPHA = 1.0;//(double) rand() / (RAND_MAX);
  double BETA = 0.0; //(double) rand() / (RAND_MAX);

  double time_baseline = 0.0;
  double time_baseline_part = 0.0;

  struct spmv_ret ret_baseline_csr;
  struct spmv_ret ret_static_csr;
  struct spmv_ret ret_dynamic_csr;

  struct spmv_ret ret_baseline_csc;
  struct spmv_ret ret_static_csc;
  struct spmv_ret ret_dynamic_csc;

  struct spmv_ret ret_baseline_coo;
  struct spmv_ret ret_static_coo;
  struct spmv_ret ret_dynamic_coo;

  ret_baseline_csr.init();
  ret_static_csr.init();
  ret_dynamic_csr.init();

  ret_baseline_csc.init();
  ret_static_csc.init();
  ret_dynamic_csc.init();

  ret_baseline_coo.init();
  ret_static_coo.init();
  ret_dynamic_coo.init();
  
  cout << "Compute CPU version" << endl;
  for (int i = 0; i < m; i++) y_verify[i] = 0.0;
  csr_spmv_cpu(m, n, nnz,
               &ALPHA,
               csrVal, csrRowPtr, csrColIdx,
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

    

  int pass_baseline_csr = 0;
  int pass_static_csr = 0;
  int pass_dynamic_csr = 0;

  int pass_baseline_csc = 0;
  int pass_static_csc = 0;
  int pass_dynamic_csc = 0;

  int pass_baseline_coo = 0;
  int pass_static_coo = 0;
  int pass_dynamic_coo = 0;

  struct spmv_ret ret;
  struct spmv_ret ret2;
  ret = ret2;
  int numa_mapping[6] = {0,0,0,1,1,1};
  
  cout << "Starting tests..." << endl;

  for (int i = 0; i < repeat_test; i++) {
    for (int i = 0; i < m; i++) {
      y_baseline_csr[i] = 0.0;
      y_static_csr[i] = 0.0;
      y_dynamic_csr[i] = 0.0;

      y_baseline_csc[i] = 0.0;
      y_static_csc[i] = 0.0;
      y_dynamic_csc[i] = 0.0;

      y_baseline_coo[i] = 0.0;
      y_static_coo[i] = 0.0;
      y_dynamic_coo[i] = 0.0;
    }
    ret = spMV_mgpu_baseline(m, n, nnz, &ALPHA,
                            csrVal, csrRowPtr, csrColIdx, 
                            x, &BETA,
                            y_baseline_csr,
                            ngpu);
    ret_baseline_csr.add(ret);

    ret = spMV_mgpu_v1_numa(m, n, nnz, &ALPHA,
                            csrVal, csrRowPtr, csrColIdx,
                            x, &BETA,
                            y_static_csr,
                            ngpu,
                            1,
                            numa_mapping); //kernel 1
    ret_static_csr.add(ret);

    ret = spMV_mgpu_baseline_csc(m, n, nnz, &ALPHA,
                                cscVal, cscColPtr, cscRowIdx,
                                x, &BETA,
                                y_baseline_csc,
                                ngpu);
    ret_baseline_csc.add(ret);

    ret = spMV_mgpu_v1_numa_csc(m, n, nnz, &ALPHA,
                                cscVal, cscColPtr, cscRowIdx,
                                x, &BETA,
                                y_static_csc,
                                ngpu,
                                1,
                                numa_mapping); //kernel 1
    ret_static_csc.add(ret);

    ret = spMV_mgpu_baseline_coo(m, n, nnz, &ALPHA,
                                cooVal, cooRowIdx, cooColIdx, 
                                x, &BETA,
                                y_baseline_coo,
                                ngpu);
    ret_baseline_coo.add(ret);
    

    

    bool correct_baseline_csr = true;
    bool correct_static_csr = true;
    bool correct_dynamic_csr = true;

    bool correct_baseline_csc = true;
    bool correct_static_csc = true;
    bool correct_dynamic_csc = true;

    bool correct_baseline_coo = true;
    bool correct_static_coo = true;
    bool correct_dynamic_coo = true;
    

    double E = 1e-3;
    for(int i = 0; i < m; i++) {
      if (abs(y_verify[i] - y_baseline_csr[i]) > E) {
        correct_baseline_csr = false;
      }
      if (abs(y_verify[i] - y_static_csr[i]) > E) {
        correct_static_csr = false;
      }
      if (abs(y_verify[i] - y_dynamic_csr[i]) > E) {
        correct_dynamic_csr = false;
      }

      if (abs(y_verify[i] - y_baseline_csc[i]) > E) {
        correct_baseline_csc = false;
      }
      if (abs(y_verify[i] - y_static_csc[i]) > E) {
        correct_static_csc = false;
      }
      if (abs(y_verify[i] - y_dynamic_csc[i]) > E) {
        correct_dynamic_csc = false;
      }

      if (abs(y_verify[i] - y_baseline_coo[i]) > E) {
        correct_baseline_coo = false;
      }
      if (abs(y_verify[i] - y_static_coo[i]) > E) {
        correct_static_coo = false;
      }
      if (abs(y_verify[i] - y_dynamic_coo[i]) > E) {
        correct_dynamic_coo = false;
      }
    }

    if (correct_baseline_csr) pass_baseline_csr++;
    if (correct_static_csr) pass_static_csr++;
    if (correct_dynamic_csr) pass_dynamic_csr++;

    if (correct_baseline_csc) pass_baseline_csc++;
    if (correct_static_csc) pass_static_csc++;
    if (correct_dynamic_csc) pass_dynamic_csc++;

    if (correct_baseline_coo) pass_baseline_coo++;
    if (correct_static_coo) pass_static_coo++;
    if (correct_dynamic_coo) pass_dynamic_coo++;
    

    ret_baseline_csr.avg(repeat_test);
    ret_static_csr.avg(repeat_test);
    ret_dynamic_csr.avg(repeat_test);

    ret_baseline_csc.avg(repeat_test);
    ret_static_csc.avg(repeat_test);
    ret_dynamic_csc.avg(repeat_test);

    ret_baseline_coo.avg(repeat_test);
    ret_static_coo.avg(repeat_test);
    ret_dynamic_coo.avg(repeat_test);
  }

  ret_baseline_csr.print();
  ret_static_csr.print();
  ret_dynamic_csr.print();
  

  ret_baseline_csc.print();
  ret_static_csc.print();
  ret_dynamic_csc.print();
  

  ret_baseline_coo.print();
  ret_static_coo.print();
  ret_dynamic_coo.print();
  

  printf("Check: %d/%d\n", pass_baseline_csr, repeat_test);
  printf("Check: %d/%d\n", pass_static_csr, repeat_test);
  printf("Check: %d/%d\n", pass_dynamic_csr, repeat_test);
  printf("Check: %d/%d\n", pass_baseline_csc, repeat_test);
  printf("Check: %d/%d\n", pass_static_csc, repeat_test);
  printf("Check: %d/%d\n", pass_dynamic_csc, repeat_test);
  printf("Check: %d/%d\n", pass_baseline_coo, repeat_test);
  printf("Check: %d/%d\n", pass_static_coo, repeat_test);
  printf("Check: %d/%d\n", pass_dynamic_coo, repeat_test);

  //myfile << avg_time_baseline << "," << avg_time_v1k1 << "," << avg_time_v1k2 << "," << avg_time_v1k3 << "," << avg_time_v2k1 << "," << avg_time_v2k2 << "," << avg_time_v2k3 << "," << avg_time_v1k1s << "," << avg_time_v1k2s << "," << avg_time_v1k3s << "," << avg_time_v2k1s << "," << avg_time_v2k2s << "," << avg_time_v2k3s;  

  cudaFreeHost(cooRowIndex);
  cudaFreeHost(cooColIndex);
  cudaFreeHost(cooVal);
  cudaFreeHost(csrVal);
  cudaFreeHost(csrRowPtr);
  cudaFreeHost(csrColIdx);
  cudaFreeHost(cscVal);
  cudaFreeHost(cscColPtr);
  cudaFreeHost(cscRowIdx);

  myfile.close();
}
