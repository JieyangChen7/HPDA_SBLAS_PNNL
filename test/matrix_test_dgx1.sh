#! /bin/bash
#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J SP_test 
#BSUB -W 00:20 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"
### End BSUB Options and begin shell commands
NUMA=AAA

if ((${NUMA} == 1))
then
	export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7}"
fi
if ((${NUMA} == 0))
then
  export OMP_PLACES="{0},{1},{2},{3},{20},{21},{22},{23}"
fi


export OMP_PROC_BIND=close

#JSRUN='jsrun -n 1 -a 1 -c 42 -g 6 -r 1 -l CPU-CPU -d packed -b packed:42 --smpiargs="-disable_gpu_hooks"'


DATA_PREFIX=/raid/data/SuiteSparse/jieyang
RESULT_PREFIX=.

#DATA_PREFIX=./
#NGPU=6
NTEST=10

PART_OPT=PPP
MERG_OPT=EEE

NVPROF='nvprof --profile-from-start off --print-gpu-trace --export-profile'

# for real type matrices
for matrix_file in MMM
do
  for NGPU in GGG #1 2 4 #5 6
  do
    _CSV_OUTPUT=${RESULT_PREFIX}/${matrix_file}_${NGPU}_${PART_OPT}_${MERG_OPT}_${NUMA}
    _PROF_OUTPUT=${RESULT_PREFIX}/${matrix_file}_${NGPU}_${PART_OPT}_${MERG_OPT}_${NUMA}
    #NVPROF_CMD=${NVPROF}' '${_PROF_OUTPUT}.prof
    [ -f ${_CSV_OUTPUT}.csv ] && rm ${_CSV_OUTPUT}.csv
    [ -f ${_PROF_OUTPUT}.prof ] && rm ${_PROF_OUTPUT}.prof
    $JSRUN ${NVPROF_CMD} ./test_spmv_dgx1 f $DATA_PREFIX/$matrix_file $NGPU $NTEST ${_CSV_OUTPUT} ${PART_OPT} ${MERG_OPT}
  done
done
