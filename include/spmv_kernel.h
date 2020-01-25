#ifndef SPMV_KERNEL
#define SPMV_KERNEL

#include <sstream>
#include <string>

using namespace std;

void print_vec(double * a, int n, string s);
void print_vec(int * a, int n, string s);
void print_vec_gpu(double * a, int n, string s);
void print_vec_gpu(int * a, int n, string s);

struct spmv_ret {
  double numa_part_time;
  double part_time;
  double comp_time;
  double comm_time;
  double merg_time;
};

struct pCSR {
  double * val;
  int * rowPtr;
  int * colIdx;
  double * x;
  double * y;

  double * dval;
  int * drowPtr;
  int * dcolIdx;
  double * dx;
  double * dy;

  int m;
  int n;
  int nnz;
  int startIdx;
  int endIdx;
  int startRow;
  int endRow;
  bool startFlag;
  bool endFlag;
  double org_y;
};


struct pCSC {
  double * cscVal;
  int * cscColPtr;
  int * cscRowIdx;
  int m;
  int n;
  int nnz;
  int startIdx;
  int endIdx;
  int startRow;
  int endRow;
  bool startFlag;
  bool endFlag;
};

struct NumaContext {
  int * numaMapping;
  int numNumaNodes;
  bool * representiveThreads;
  int * numGPUs;
  int * workload;
  NumaContext(int * numaMapping) {
    int num_numa_nodes = 0;
    for (int i = 0; i < ngpu; i++) {
      if (numaMapping[i] > num_numa_nodes) {
        num_numa_nodes = numaMapping[i];
      }
    }
    num_numa_nodes += 1;
    printf("# of NUMA nodes: %d\n", num_numa_nodes);
    printf("Representive threads: ");
    bool * representive_threads = new bool[ngpu];
    for (int i = 0; i < ngpu; i++) representive_threads = false;
    for (int i = 0; i < num_numa_nodes; i++) {
      for (int j = 0; j < ngpu; j++) {
        if (numaMapping[j] == i) {
          //representive_threads[i] = j;
          representive_threads[j] = true;
          break;
        }
        printf("%d ", j);
      }
    }
    printf("\n");

    printf("# of GPU distribution: ");
    int * num_gpus = new int [num_numa_nodes];
    for (int j = 0; j < num_numa_nodes; j++) {
      num_gpus[j] = 0;
    }
    for (int j = 0; j < ngpu; j++) {
      num_gpus[numaMapping[j]]++;
    }
    for (int i = 0; i < num_numa_nodes; i++) {
      printf("%d ", num_gpus[i]);
    }
    printf("\n");

    int * workload = new int [num_numa_nodes+1];
      workload[0] = 0;
    workload[1] = num_gpus[0];      
    for (int i = 2; i < num_numa_nodes+1; i++) {
      workload[i] = workload[i-1] + num_gpus[i-1];
    }
    print_vec(workload, num_numa_nodes+1, "workload: ");

    numaContext->numaMapping = numaMapping;
    numaContext->numNumaNodes = num_numa_nodes;
    numaContext->representiveThreads = representive_threads;
    numaContext->numGPUs = num_gpus;
    numaContext->workload = workload;

  }
};



}
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



#endif /* SPMV_KERNEL */
