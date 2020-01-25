#! /bin/bash
#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J SP_test 
#BSUB -W 00:20 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"
### End BSUB Options and begin shell commands

export OMP_PLACES="{0},{28},{56},{88},{116},{144}"
export OMP_PROC_BIND=close

JSRUN='jsrun -n 1 -a 1 -c 42 -g 6 -r 1 -l CPU-CPU -d packed -b packed:42 --smpiargs="-disable_gpu_hooks"'


DATA_PREFIX=/gpfs/alpine/world-shared/csc297/test_matrices
RESULT_PREFIX=/gpfs/alpine/scratch/jieyang/csc331/spmv_results

#DATA_PREFIX=./
#NGPU=6
NTEST=1

NVPROF='nvprof --profile-from-start off --print-gpu-trace --export-profile'

# for real type matrices
for matrix_file in MMM
do
  _CSV_SUMMARY=${RESULT_PREFIX}/${matrix_file}
  [ -f ${_CSV_SUMMARY}.csv ] && rm ${_CSV_SUMMARY}.csv
  for NGPU in GGG #1 2 4 #5 6
  do
    _CSV_OUTPUT=${RESULT_PREFIX}/${matrix_file}_${NGPU}
    _PROF_OUTPUT=${RESULT_PREFIX}/${matrix_file}_${NGPU}
    #NVPROF_CMD=${NVPROF}' '${_PROF_OUTPUT}.prof
    [ -f ${_CSV_OUTPUT}.csv ] && rm ${_CSV_OUTPUT}.csv
    [ -f ${_PROF_OUTPUT}.prof ] && rm ${_PROF_OUTPUT}.prof
    $JSRUN ${NVPROF_CMD} ./test_spmv f $DATA_PREFIX/$matrix_file $NGPU $NTEST ${_CSV_OUTPUT} 
    cat ${_CSV_OUTPUT}.csv >> ${_CSV_SUMMARY}.csv
    echo "" >> ${_CSV_SUMMARY}.csv
  done
done
