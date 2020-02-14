NUMA=AAA

if ((${NUMA} == 1))
then
  export OMP_PLACES="{0},{2},{4},{8},{60},{62},{64},{66}"
fi
if ((${NUMA} == 0))
then
  export OMP_PLACES="{0},{2},{4},{6},{8},{10},{12},{14}"
fi


export OMP_PROC_BIND=close

DATA_PREFIX=../data
RESULT_PREFIX=.

NTEST=10

PART_OPT=PPP
MERG_OPT=EEE

NVPROF='nvprof --profile-from-start off --print-gpu-trace --export-profile'

for matrix_file in MMM
do
  for NGPU in GGG
  do
    _CSV_OUTPUT=${RESULT_PREFIX}/${matrix_file}_${NGPU}_${PART_OPT}_${MERG_OPT}_${NUMA}
    _PROF_OUTPUT=${RESULT_PREFIX}/${matrix_file}_${NGPU}_${PART_OPT}_${MERG_OPT}_${NUMA}
    #NVPROF_CMD=${NVPROF}' '${_PROF_OUTPUT}.prof
    [ -f ${_CSV_OUTPUT}.csv ] && rm ${_CSV_OUTPUT}.csv
    [ -f ${_PROF_OUTPUT}.prof ] && rm ${_PROF_OUTPUT}.prof
    $JSRUN ${NVPROF_CMD} ./test_spmv_dgx2 f $DATA_PREFIX/$matrix_file $NGPU $NTEST ${_CSV_OUTPUT} ${PART_OPT} ${MERG_OPT}
  done
done
