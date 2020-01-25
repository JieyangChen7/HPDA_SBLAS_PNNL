#ifndef SPMV_KERNEL
#define SPMV_KERNEL

#include <sstream>
#include <string>

using namespace std;

struct spmv_ret {
	double numa_part_time;
	double part_time;
  double comp_time;
  double comm_time;
	double merg_time;
};

int csr5_kernel(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  cudaStream_t stream);

spmv_ret spMV_mgpu_baseline(int m, int n, int nnz, double * alpha,
				 double * csrVal, int * csrRowPtr, int * csrColIndex, 
				 double * x, double * beta,
				 double * y,
				 int ngpu);
spmv_ret spMV_mgpu_v1(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu,
				  int kernel);

spmv_ret spMV_mgpu_v1_numa(int m, int n, int nnz, double * alpha,
          double * csrVal, int * csrRowPtr, int * csrColIndex,
          double * x, double * beta,
          double * y,
          int ngpu,
          int kernel,
					int * numa_mapping);

spmv_ret spMV_mgpu_v2(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  int kernel,
				  int nb,
				  int copy_of_workspace);


spmv_ret spMspV_mgpu_v1(int m, int n, int nnz, double * alpha,
                                  double * csrVal, int * csrRowPtr, int * csrColIndex,
                                  double * x, double * beta,
                                  double * y,
                                  int ngpu,
                                  int kernel);
spmv_ret spMspV_mgpu_v1_numa(int m, int n, int nnz, double * alpha,
                                  double * csrVal, int * csrRowPtr, int * csrColIndex,
                                  double * x, double * beta,
                                  double * y,
                                  int ngpu,
                                  int kernel);

spmv_ret spMspV_mgpu_v2(int m, int n, int nnz, double * alpha,
                                  double * csrVal, int * csrRowPtr, int * csrColIndex,
                                  double * x, double * beta,
                                  double * y,
                                  int ngpu,
                                  int kernel,
                                  int nb,
                                  int copy_of_workspace);



int get_row_from_index(int n, int * a, int idx);

double get_time();

double get_gpu_availble_mem(int ngpu);

void print_vec(double * a, int n, string s);
void print_vec(int * a, int n, string s);
void print_vec_gpu(double * a, int n, string s);
void print_vec_gpu(int * a, int n, string s);

#endif /* SPMV_KERNEL */
